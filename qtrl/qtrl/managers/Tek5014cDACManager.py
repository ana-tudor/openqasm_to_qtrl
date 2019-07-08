# Copyright (c) 2018-2019, UC Regents

import logging, sys
import numpy as np
import qtrl
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
from .DACManager import DACManager

log = logging.getLogger('qtrl.Tek5014cDAC')


class Tek5014cDACManager(DACManager):
    def __init__(self, config_file='DAC.yaml', variables={}):
        """
        Config file should be laid out like so:
        '
        Channel_0:
            amplitude: 1.5
            dc_offset: 0.0
        Channel_1:
            amplitude: 1.5
            dc_offset: 0.0
        Channel_2:
            amplitude: 1.5
            dc_offset: 0.0
        clock_delay: 100
        '
        etc...

        amplitude and dc_offset are in units of Volts, with maximum being 1.5V
            note that since the DAC can only put out 1.5 max, having a peak
            amplitude of a pulse at 1.5V with a DC offset of 0.5 volts will lead
            to a new peak of 2V, which is outside the range of the DAC.
            As a result of this, there is clipping when there is a non-zero
            dc_offset and pulses which approach the voltage limit.

        clock_delay is the delay between each sequence element to be played.
            it is defined in units of ns, and will be rounded to the nearest
            100ns step. I suspect there is a finite offset of this value as well
            be ~100ns but have not exactly calculated it.
        """

        self._n_channels = 0
        self.channels = []
        self._sample_rate = float(1e9)
        super().__init__(config_file, variables)
        self.name = self._config_dict['name']
        self.awg = getattr(sys.modules[__name__],
                           self._config_dict['model'])(self.name, self._config_dict['address'])
        self._n_channels = self.awg.num_channels
        self.channels = np.arange(1, self._n_channels+1)
        self._n_markers = 2
        self.markers = {}

        # Timing information
        self.sample_rate(self._sample_rate)
        self._seq_length = self.awg.sequence_length()

        qtrl._DAC = self

    def load(self):
        """Inherited from the Config load, this loads the config file, and
        additionally sets the valuesin to the device.
        """

        super().load()

        for channel in self.channels:
            c_str = 'Channel_{0:02d}'.format(channel)
            if c_str not in self:
                log.info(f'{c_str} not found in config yaml, adding it now with defaults')
                self.set(c_str, {'amplitude': 1.5, 'dc_offset': 0.0}, save_config=True)

            val = self.get(c_str, reload_config=False)
            self.amplitude(channel, val['amplitude'])
            self.offset(channel, val['dc_offset'])

            if 'markers' in val.keys():
                self.markers[f'ch_{channel}'] = {}
                for m in val['markers']:
                    self.markers[f'ch_{channel}']['m{m}'] = {}
                    #set marker to high value
                    self.marker_value(channel, m, val['markers'][m]['high'], value_type='high')
                    self.marker_value(channel, m, val['markers'][m]['low'], value_type='low')
        if hasattr(self, 'awg'):
            self.trigger_interval(self._config_dict['clock_delay'])
            self._trigger_interval = self.trigger_interval()


    @property
    def n_channels(self):
        """Return the number of available channels.
        """

        return self._n_channels

    def flush(self):
        """
        paper-thin wrapper for Qcodes awg driver:

        Delete all user-defined waveforms in the list in a single
        action. Note that there is no “UNDO” action once the waveforms
        are deleted. Use caution before issuing this command.

        If the deleted waveform(s) is (are) currently loaded into
        waveform memory, it (they) is (are) unloaded. If the RUN state
        of the instrument is ON, the state is turned OFF. If the
        channel is on, it will be switched off.
        """

        self.awg.delete_all_waveforms_from_list()

    def marker_value(self, channel, marker, value=None, value_type='high'):
        if value is not None:
            self.markers[f'ch_{channel}']['m{m}']['high'] = \
                getattr(self.awg, f'ch{channel}_m{marker}_{value_type}').set(value)
            return value
        return getattr(self.awg, f'ch{channel}_m{marker}_{value_type}').get()

    def amplitude(self, channel, amp):
        """Set the amplitude of the DAC channels, must be supplied as voltage.
        """

        # assert abs(amp) <= 1.5, 'Amplitude has to be less than 1.5V'
        getattr(self.awg, f'ch{channel}_amp').set(amp)

    def offset(self, channel, offset):
        """Set the DC offset of the DAC channels, must be supplied as voltage.
        """

        getattr(self.awg, f'ch{channel}_offset').set(offset)

    def trigger_interval(self, wait_time=None):
        """setter/getter for trigger interval. wait_time is in nanoseconds.
        """

        if wait_time is None:
            return self.awg.trigger_seq_timer.get()
        return self.awg.trigger_seq_timer.set(wait_time)

    def sample_rate(self, sr=None):
        """Get or set the sample rate of the box.
        """

        if sr is None:
            return self.awg.clock_freq()
        return self.awg.clock_freq(sr)

    def write_sequence(self, seq):
        """Write a Sequence to the DAC channels.
        """

        # num_chan, num_elem, num_time_points, num_subchannels = seq.array.shape

        channels_used = np.argwhere(np.array(np.max(np.abs(seq.array), (2, 1)) != 0))
        analog_channels_used = \
            np.array([channel[0] for channel in channels_used if not channel[1]])

        # Pull used analog waveforms (everything that isn't a marker). We choose
        # 1-indexing (hence [1:,...] to match the Tektronix channel numbering
        # convention, so chan 0 will always be empty
        analog_waveforms = \
            np.array([seq.array[a, :, :, 0] for a in analog_channels_used]).astype(np.float64)
        marker1s = \
            np.array([seq.array[a, :, :, 1] for a in analog_channels_used]).astype(np.float64)
        marker2s = np.zeros_like(marker1s).astype(np.float64)

        # See QCoDeS Tek 5014C example notebook
        # http://qcodes.github.io/Qcodes/examples/driver_examples/Qcodes%20example%20with%20Tektronix%20AWG5014C.html
        nreps = [1]*seq.n_elements
        trig_waits = [1] * seq.n_elements
        jump_tos = [2] * seq.n_elements
        goto_states = np.roll(np.arange(seq.n_elements) + 1, -1)

        # make, send, and load awg
        self.awg.make_send_and_load_awg_file(analog_waveforms, marker1s, marker2s,
                                             nreps, trig_waits, goto_states,
                                             jump_tos, channels=analog_channels_used)
        # turn on channels!
        self.awg.trigger_seq_timer.set(self._trigger_interval)
        self.awg.all_channels_on()
        self.awg.trigger_seq_timer.set(self._trigger_interval)
        self._seq_length = self.awg.sequence_length()

    def run(self, state=1):
        """Start playing out of the sequence from waveform memory.
        """

        self.load()

        if state:
            self.run(0)
            self.trigger_interval(self.get('clock_delay'))
            # run it!
            self.awg.start()

        else:
            self.awg.stop()

    def close(self):
        """Disconnect and irreversibly tear down the instrument.
        """

        self.awg.close()

    # def __del__(self):
    #     self.close()
