# Copyright (c) 2018-2019, UC Regents

import pkg_resources
import logging
import numpy as np
import qtrl
from .ManagerBase import ManagerBase
from ..keysight.utils import *
from ..keysight import keysightSD1 as sd1

log = logging.getLogger('qtrl.KeysightDAC')


class KeysightDACManager(ManagerBase):
    def __init__(self, config_file='DAC.yaml', variables={}):
        """Finds and open connections to all cards on a keysight PXI chassis.
        DAC channels are enumerated from top to bottom -> left to right, 0 indexed

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

        self.channels = {}
        self._n_channels = 0
        self._n_markers = 0

        # Timing information
        self._sample_rate = 1e9
        self._seq_length = 0

        # waveform SD objects
        self._wave_sds = []

        # Instantiate and connect cards
        # get cards and sort by chassis and slot
        self.cards = []
        self.cards = [x for x in get_cards() if x.type == 'DAC']
        self.cards = sorted(self.cards, key=lambda x: x.chassis)
        self.cards = sorted(self.cards, key=lambda x: x.slot)

        # connect to cards
        self._connect()

        super().__init__(config_file, variables)

        qtrl._DAC = self

    def load(self):
        """Inherited from the Config load, this loads the config file, and additionally sets the values
        in to the device."""
        super().load()
        for channel in range(self.n_channels):
            c_str = 'Channel_{0:02d}'.format(channel)
            if c_str not in self:
                log.info(f'{c_str} not found in config yaml, adding it now with defaults')
                self.set(c_str, {'amplitude': 1.5, 'dc_offset': 0.0}, save_config=True)

            val = self.get(c_str)
            self.amplitude(channel, val['amplitude'])
            self.offset(channel, val['dc_offset'])
        self._set_register(0, self.get('clock_delay', 1000)//100 + self._seq_length//100 - 1)

    def run(self, state=1):
        """Start playing out of the sequence from waveform memory."""
        self.load()
        if state:
            self.run(0)
            self._set_register(0, self.get('clock_delay', 1000)//100 + self._seq_length//100 - 1)

            for card in self.cards:
                card.connection.AWGstartMultiple(2**card.channels - 1)
            self._hvi.start()

        else:
            for card in self.cards:
                card.connection.AWGstopMultiple(2**card.channels - 1)
            self._hvi.stop()

    @property
    def n_channels(self):
        """Return the number of available channels"""
        return self._n_channels

    def flush(self):
        """Flush the memory of all of the cards"""
        for card in self.cards:
            card.connection.waveformFlush()
            for chan in range(card.channels):
                card.connection.AWGflush(chan)
                card.connection.AWGstop(chan)
                card.connection.channelWaveShape(chan+1, sd1.SD_Waveshapes.AOU_AWG)

    def amplitude(self, channel, amp):
        """Set the amplitude of the DAC channels, must be supplied as voltage"""
        chan = self.channels[channel]

        assert abs(amp) <= 1.5, 'Amplitude has to be less than 1.5V'
        err = chan.connection.channelAmplitude(chan.channel+1, amp)
        assert err >= 0, 'Failed to set amplitude {}'.format(err)

    def offset(self, channel, offset):
        """Set the DC offset of the DAC channels, must be supplied as voltage"""
        chan = self.channels[channel]

        assert abs(offset) <= 1.5, 'Amplitude has to be less than 1.5V'
        err = chan.connection.channelOffset(chan.channel+1, offset)
        assert err >= 0, 'Failed to set amplitude {}'.format(err)

    @property
    def sample_rate(self, sr=None):
        """Returns the sample rate of the box, currently defaults to 1e9"""
        return self._sample_rate

    def write_sequence(self, seq):
        """Write a Sequence to the DAC channels"""
        for card in self.cards:
            card.connection.AWGstopMultiple(2**card.channels - 1)
        self.flush()
        waveforms, seq_table = seq.generate_seq_table()

        # Ensure we have the correct formatting
        waveforms = np.array(np.clip(waveforms, -1., 1.)[..., 0], dtype='float32')

        assert np.shape(waveforms)[0] < 1024, "Keysight can only have 1024 waveforms at a time"

        for chan, conn in self.channels.items():
            if chan >= seq_table.shape[0]:
                continue
            for i, waveform in enumerate(waveforms):

                # only write waveforms to the channels which will play them
                if i not in np.unique(seq_table[chan]):
                    continue

                wave = sd1.SD_Wave()

                # load waveform into local ram
                error = wave.newFromArrayDouble(sd1.SD_WaveformTypes.WAVE_ANALOG, waveform)
                assert error >= 0, "Error newFromArrayDouble {}".format(ks_errors.get(error, error))

                error = conn.connection.waveformLoad(wave, i+1)
                assert error >= 0, "Error waveformLoad {}".format(ks_errors.get(error, error))

                del wave  # Delete the waveform object to decrement keysight SD1 memory counter
        for chan, sequence in enumerate(seq_table):
            for aseq in sequence:
                self._queue_waveform(chan, aseq[0]+1, repeat=1)
                
        for chan, sequence in enumerate(seq_table):
            # This sets us to cyclic mode, queue has to be made already
            assert self.channels[chan].connection.AWGqueueConfig(self.channels[chan].channel+1, 1) >= 0
            assert self.channels[chan].connection.AWGqueueSyncMode(self.channels[chan].channel+1, 1) >= 0
        self._seq_length = len(waveforms[0, :])

    def close(self):
        """Close all connections to cards"""
        for i, card in enumerate(self.cards):
            if card.connection is not None:
                card.connection.close()
            self.cards[i] = card._replace(connection=None)

    def _connect(self):
        """Instantiate connections to each card in the keysight chassis.
        """

        log.info("Loading HVI")

        self._hvi = sd1.SD_HVI()
        hvi_file = pkg_resources.resource_filename("qtrl.keysight", 'sequencer.hvi')
        log.info(hvi_file)
        self._hvi.open(hvi_file)
        # for some unknown reason, this has to be run twice before it will not error
        self._hvi.assignHardwareWithIndexAndSlot(nChassis=1, nSlot=3, index=0)
        self._hvi.assignHardwareWithIndexAndSlot(nChassis=1, nSlot=4, index=1)
        self._hvi.assignHardwareWithIndexAndSlot(nChassis=1, nSlot=5, index=2)

        assert self._hvi.open(hvi_file) >= 0, 'Failed to load HVI'
        assert self._hvi.assignHardwareWithIndexAndSlot(nChassis=1, nSlot=3, index=0) >= 0, 'Failed to load HVI'
        assert self._hvi.assignHardwareWithIndexAndSlot(nChassis=1, nSlot=4, index=1) >= 0, 'Failed to load HVI'
        assert self._hvi.assignHardwareWithIndexAndSlot(nChassis=1, nSlot=5, index=2) >= 0, 'Failed to load HVI'

        assert self._hvi.compile() >= 0, 'Failed to load HVI'

        assert self._hvi.load() >= 0, 'Failed to load HVI'
        self._hvi.reset()

        cur_chan = 0
        for i, card in enumerate(self.cards):
            if card.connection is not None:
                self.close()

            card_cxn = sd1.SD_AOU()
            assert card_cxn.openWithSlot("", card.chassis, card.slot) > 0, 'Failed to connect to slot'

            self.cards[i] = card._replace(connection=card_cxn)
            # self.cards[i].connection.triggerIOconfig(sd1.SD_TriggerDirections.AOU_TRG_IN)

            for channel in range(card.channels):
                self.channels[cur_chan] = KeysightChannel(channel=channel,
                                                          chassis=card.chassis,
                                                          slot=card.slot,
                                                          model=card.model,
                                                          type=card.type,
                                                          connection=card_cxn)

                self.channels[cur_chan].connection.channelWaveShape(channel+1, sd1.SD_Waveshapes.AOU_AWG)

                self.channels[cur_chan].connection.clockResetPhase(3, 0)

                # ext trig config, 0 is external source, 3 is rising edge
                # self.channels[cur_chan].connection.AWGtriggerExternalConfig(channel+1, 0, 3)

                cur_chan += 1
        self._hvi.start()
        self._hvi.stop()

        self._n_channels = cur_chan


    def _queue_waveform(self, channel, w_id, repeat=1):
        card_chan = self.channels[channel]

        # Trigger setting for external trigger mode
        # trig = sd1.SD_TriggerModes.EXTTRIG
        trig = sd1.SD_TriggerModes.SWHVITRIG

        err = card_chan.connection.AWGqueueWaveform(nAWG=card_chan.channel+1,
                                                    waveformNumber=int(w_id),
                                                    triggerMode=int(trig),
                                                    startDelay=0,
                                                    cycles=repeat,
                                                    prescaler=0)
        assert err >= 0, f'{ks_errors[err]}'

    def _set_register(self, reg, value):
        for card in self.cards:
            card.connection.writeRegisterByNumber(reg, int(value))

    # def __del__(self):
    #     self.close()
