# Copyright (c) 2018-2019, UC Regents

import pkg_resources
import numpy as np
from qtrl.keysight.utils import *
from .ADCManager import ADCManager


class KeysightADCManager(ADCManager):
    _default_config = {'pp_voltage': 1.5,
                       'points': 1024,
                       'n_reps': 1,
                       'n_elements': 1,
                       'n_triggers': 1,
                       'n_batches': 1,
                       'delay': 1000,
                       'keep_raw': False}

    def __init__(self, config_file='ADC.yaml', variables={}):
        super().__init__(config_file, variables)

        self.channels = {}

        # Instantiate and connect cards
        # get cards and sort by chassis and slot
        self.cards = [x for x in get_cards() if x.type == 'ADC']
        self.cards = sorted(self.cards, key=lambda x: x.chassis)
        self.cards = sorted(self.cards, key=lambda x: x.slot)

        self._connect()
        self.load()

    def _run(self, state):
        for card in self.cards:
            card.connection.FPGAwritePCport(
                0, np.array([int(not bool(state))], dtype=np.int32), 7, 0, 1)

    def _dev_config(self, **settings):
        """Loads keysight specific settings from the config"""
        super()._dev_config(**settings)
        self._sample_rate = 500e6

    def _arm(self):
        """Tell the ADC the shape of the data it will be collected, and tell it to start
        collecting.  It will only actually collect data when it sees a trigger."""
        self._flush()

        # +24 tick offset to get 0 delay from trigger start
        self._timing = self._settings.get('timing', self._settings['n_elements']*[0])
        self._timing = np.array(self._timing, dtype=np.int32) + 24

        for card in self.cards:
            # set the coupling and impedance
            for chan in range(card.channels):
                card.connection.channelInputConfig(\
                    chan + 1,
                    self._settings['pp_voltage'],
                    sd1.AIN_Impedance.AIN_IMPEDANCE_50,
                    sd1.AIN_Coupling.AIN_COUPLING_DC)

                n_cycles = self._settings['n_reps'] * \
                           self._settings['n_elements'] * \
                           self._settings['n_triggers']

                # tell the channels how much data they are collecting
                card.connection.DAQconfig(chan + 1,
                                          pointsPerCycle=self._settings['points'],
                                          nCycles=n_cycles,
                                          triggerDelay=0,
                                          triggerMode=3)

            # Write the timing table
            card.connection.FPGAwritePCport(1, self._timing, 0, 0, 1)

            # Write the sequence size to the ADC
            # -1 delay and n_elements to get counting correct
            buff = np.array([self._settings['delay'] - 1,
                             self._settings['n_elements'],
                             self._settings['n_triggers'],
                             0, 0, 0, 0, 1],
                            dtype=np.int32)
            card.connection.FPGAwritePCport(0, buff, 0, 0, 1)

            # start them all, this is a binary mask
            card.connection.DAQstartMultiple(2**card.channels-1)

        for card in self.cards:
            # Start the triggering by writing 0 to byte 7
            card.connection.FPGAwritePCport(0, np.array([0], dtype=np.int32), 7, 0, 1)

    def _acquire(self):
        """Acquire data after the cards have been armed.
        Important: Card have to have been armed"""

        results = []
        for card in self.cards:
            card_results = []

            acquire_shape = [self._settings['n_reps'],
                             self._settings['n_elements'],
                             self._settings['n_triggers'],
                             self._settings['points']]

            acquire_size = int(np.product(acquire_shape))

            # last argument of the daqread is a timeout, I think it is ms,
            # 0 will never time out, not ideal
            card_results.append(
                card.connection.DAQread(1, acquire_size, 1000).reshape(acquire_shape))
            card_results.append(
                card.connection.DAQread(2, acquire_size, 1000).reshape(acquire_shape))
            results.append(card_results)

        if len(results) == 1:
            results = results[0]

        for card in self.cards:
            # Stop the triggering by writing 1 to byte 7
            card.connection.FPGAwritePCport(0, np.array([1], dtype=np.int32), 7, 0, 1)

        self._flush()

        return np.array(results)

    def _flush(self):
        """Flush the memory of all of the cards"""
        for card in self.cards:
            card.connection.DAQstopMultiple(2**card.channels-1)
            card.connection.DAQflushMultiple(2**card.channels-1)

    def close(self):
        """Close all connections to cards"""
        for i, card in enumerate(self.cards):
            if card.connection is not None:
                card.connection.close()
            self.cards[i] = card._replace(connection=None)

    def _connect(self):
        """Instantiate connections to each card in the keysight chassis"""
        cur_chan = 0
        for i, card in enumerate(self.cards):
            if card.connection is not None:
                self.close()

            card_cxn = sd1.SD_AIN()
            assert card_cxn.openWithSlot("", card.chassis, card.slot) > 0, 'Failed to connect to slot'

            self.cards[i] = card._replace(connection=card_cxn)

            for channel in range(card.channels):
                self.channels[cur_chan] = KeysightChannel(channel=channel,
                                                          chassis=card.chassis,
                                                          slot=card.slot,
                                                          model=card.model,
                                                          type=card.type,
                                                          connection=card_cxn)

                cur_chan += 1
            card_cxn.FPGAload(pkg_resources.resource_filename("qtrl.keysight", 'ADC_fpga.sbp'))
        print('Connected to: Keysight ADC')

        self._n_channels = cur_chan

    def __del__(self):
        self.close()

