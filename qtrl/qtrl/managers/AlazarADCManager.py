# Copyright (c) 2018-2019, UC Regents

import copy
from .ADCManager import ADCManager
from ..alazar import Alazar


class AlazarADCManager(ADCManager):
    """"""
    _default_config = {'points': 1024,
                       'n_reps': 1,
                       'n_elements': 1,
                       'n_triggers': 1,
                       'n_batches': 1,
                       'keep_raw': False}

    _default_hardware_config = {'clock_source': 'EXTERNAL_CLOCK_10MHz_REF',
                                'sample_rate': 1e+9,
                                'trigger_level': 150}

    def __init__(self, config_file='ADC.yaml', variables={}):
        self._alazar = Alazar()
        super().__init__(config_file, variables)

    def _arm(self):
        self._alazar.stop()
        self._dev_config()
        self._alazar.start_acquire(\
            samples=self._settings['points'],
            n_repetitions=self._settings['n_reps'],
            mean=False,
            expected_triggers=self._settings['n_triggers']*self._settings['n_elements'])

    def _dev_config(self, **settings):
        """Loads keysight specific settings from the config"""

        super()._dev_config(**settings)
        hw_settings = copy.deepcopy(self.get('hardware_settings'))
        self._alazar.capture_clock_settings(clock_source=hw_settings['clock_source'],
                                            sample_rate=hw_settings['sample_rate'])
        self._alazar.trigger_settings(trig_source='EXTERNAL',
                                      trig_slope='POSITIVE',
                                      trig_threshold=int(hw_settings['trigger_level']))

    def _acquire(self):
        """Acquire data after the cards have been armed.
        Important: Card have to have been armed"""

        acquire_shape = [2,
                         self._settings['n_reps'],
                         self._settings['n_elements'],
                         self._settings['n_triggers'],
                         self._settings['points']]

        result = self._alazar.acquire()

        return result.reshape(acquire_shape)

