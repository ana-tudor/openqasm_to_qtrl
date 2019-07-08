# Copyright (c) 2018-2019, UC Regents

import copy
import numpy as np
import qtrl
from .ManagerBase import ManagerBase
from ..sequencer import Sequence
from ..processing import process_lookup

import logging
log = logging.getLogger('qtrl.ADCManager')


class ADCManager(ManagerBase):
    """Parent class for all ADC cards:

    Formatting:
        Kept (mostly) consistent with old formatting for comparability reasons.
        The biggest change is elements and triggers are being delineated, which
        is an improvement.

        Raw traces from the ADCs will be assumed to be of this form:
            [IQ (2 chan), repetitions, elements, triggers, time points]

        After raw processing the assumed shape will be:
            [Frequency, IQ (2 chan), repetitions, elements, triggers]


    The steps which need to happen in this class are as follows:
        Manage config file - this entails a config file which should contain all
            of the information needed in order to recreate a measurement in full,
            given a set of raw traces from the ADC.

        Processing will be recorded in a dictionary being used as a 2 level tree:
            the first level of the processing dictionary can be used to
            differentiate processing classes, IE: common, fitting, etc as keys

    The next level is an ordered dictionary of ADCProcess objects, the keys for
    this level are to be used as descriptors for what is happening at that point,
    IE: ExpSinFit, etc.
    """

    _default_config = {'points': 1024,
                       'n_reps': 1,
                       'n_elements': 1,
                       'n_triggers': 1,
                       'n_batches': 1,
                       'keep_raw': False}

    _default_hardware_config = {}

    def __init__(self, config_file='ADC.yaml', variables={}):
        super().__init__(config_file, variables)

        self._settings = {}
        self._processing = {}
        self.load()
        self.raw_batch = None
        self.measurement = {}
        qtrl._ADC = self

    def _arm(self):
        raise NotImplementedError()

    def _acquire(self):
        raise NotImplementedError()

    def _dev_config(self, **settings):
        """This needs to be replaced by each ADCs implementation.
        It will be called after the config file has been loaded.
        It should be used to set ADC hardware settings, such as sample rate.
        """
        super().load()

        self._settings = copy.deepcopy(self._default_config)
        self._settings.update(self.get("acquisition_settings", {}))
        self._settings.update(settings)

        self.set('acquisition_settings', self._settings)

    def load(self):
        super().load()

        process_dict = self.get('processing', {})

        # if there is no acquisition settings in the config file, add the default
        if 'acquisition_settings' not in self.keys(''):
            log.warning("No acquisition_settings found in config, adding default.")
            self.set('acquisition_settings', self._default_hardware_config)
        else:
            current_keys = self.keys('acquisition_settings')
            for key in self._default_config:
                if key not in current_keys:
                    self.set(f'acquisition_settings/{key}', self._default_config[key])
        self._dev_config()

        processes = {}
        # Build a dictionary tree of all processing to be done
        for process_cat in process_dict:
            processes[process_cat] = {}
            for process in process_dict.get(process_cat, {}):
                if is_process(process_dict[process_cat][process]):
                    processes[process_cat][process] = \
                        construct_process(process_dict[process_cat][process])
                elif process == 'enable':
                    processes[process_cat][process] = process_dict[process_cat][process]
        self._processing = processes

        # if there is no hardware_settings in the config file, add the default
        if 'hardware_settings' not in self.keys(''):
            self.set('hardware_settings', self._default_hardware_config)
        else:
            current_keys = self.keys('hardware_settings')
            for key in self._default_hardware_config:
                if key not in current_keys:
                    self.set(f'hardware_settings/{key}', self._default_hardware_config[key])
        self.save()

    def acquire(self, seq=None, points=None, n_reps=None, n_elements=None,
                n_triggers=None, n_batches=None, **settings):
        """Acquire data
            Accepts:
                shape - a list of the form [repetitions, elements, triggers, batches, time points]
                        where a partial list is allowed
                        also accepts an envelope sequence and pull the n_elements from that
                **settings - any settings passed here will over-write the settings
                        this has a higher priority than the shape

            Every measurement results in a measurement dictionary being created.
            This dictionary should be restricted in structure.
            - raw_trace - this will be the full raw trace of the acquisition
            - R{} - R0, R1 etc, this will be processed results from the measurement
            - Joint - this is for joint measurements, IE: correlators, confusion matrix etc.
            - Settings - dictionary containing the settings for the measurement
        """

        if points is not None:
            settings['points'] = points
        if n_reps is not None:
            settings['n_reps'] = n_reps
        if n_elements is not None:
            settings['n_elements'] = n_elements
        if n_triggers is not None:
            settings['n_triggers'] = n_triggers
        if n_batches is not None:
            settings['n_batches'] = n_batches
        if isinstance(seq, Sequence):
            settings['n_elements'] = seq.n_elements
        elif n_elements is not None:
            settings['n_elements'] = n_elements

        self._dev_config(**settings)

        self.measurement = {'raw_trace': None,
                            'joint': {},
                            'settings': {'ADC_settings': copy.deepcopy(self._config_raw)}}

        # run the prep functions
        for proc_category in self._processing:
            keys = list(self._processing[proc_category].keys())
            if 'enable' in keys:
                if not self._processing[proc_category]['enable']:
                    continue
                else:
                    keys.remove('enable')
            for process in keys:
                if self._processing[proc_category][process][1]:
                    self._processing[proc_category][process][0].prep()

        # start batching
        for batch in range(self._settings['n_batches']):
            if self._settings['n_batches'] > 1:
                log.info("Batch {}".format(batch + 1))
            # arm the ADC
            self._arm()

            # run the batch_start functions
            for proc_category in self._processing:
                keys = list(self._processing[proc_category].keys())
                if 'enable' in keys:
                    if not self._processing[proc_category]['enable']:
                        continue
                    else:
                        keys.remove('enable')
                for process in keys:
                    if self._processing[proc_category][process][1]:
                        self._processing[proc_category][process][0].batch_start()

            self.raw_batch = self._acquire()
            if self.measurement['raw_trace'] is None:
                self.measurement['raw_trace'] = self.raw_batch
            else:
                self.measurement['raw_trace'] = \
                    np.concatenate([self.measurement['raw_trace'], self.raw_batch], 1)

            keep_raw = self._settings['keep_raw']
            if keep_raw is True:
                raw_trace = self.measurement['raw_trace']
                if batch == 0:
                    raw_data = raw_trace
                else:
                    raw_data = np.append(raw_data, raw_trace, axis=1)

            # run the batch_end functions
            for proc_category in self._processing:
                keys = list(self._processing[proc_category].keys())
                if 'enable' in keys:
                    if not self._processing[proc_category]['enable']:
                        continue
                    else:
                        keys.remove('enable')
                for process in keys:
                    if self._processing[proc_category][process][1]:
                        self._processing[proc_category][process][0].batch_end(self.measurement)

        if keep_raw is True:
            self.measurement['raw_trace'] = raw_data

        self._post(seq)

        return self.measurement

    def _post(self, seq=None):
        """Run all of the post processing """
        # run the post functions
        for proc_category in self._processing:
            keys = list(self._processing[proc_category].keys())
            if 'enable' in keys:
                if not self._processing[proc_category]['enable']:
                    continue
                else:
                    keys.remove('enable')
            for process in keys:
                if self._processing[proc_category][process][1]:
                    self._processing[proc_category][process][0].post(self.measurement, seq)


def is_process(config):
    """test to see if there is a valid process configuration in a given dictionary-like object."""

    try:
        keys = list(config.keys())
        keys.remove('type')
        keys.remove('enable')
        keys.remove('kwargs')
    except AttributeError:
        return False

    return True


def construct_process(config):
    """Construct a process from a given config dictionary like object"""

    assert is_process(config), "This doesn't appear to be a valid process"
    pro_type = config['type']

    config = copy.deepcopy(config)
    del config['type']
    enabled = config['enable']

    return [process_lookup[pro_type](**config['kwargs']), enabled]
