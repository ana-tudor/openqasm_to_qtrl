# Copyright (c) 2018-2019, UC Regents

import qtrl
from .ManagerBase import ManagerBase


class DACManager(ManagerBase):
    """Template class for a DAC device"""

    def __init__(self, config_file='DAC.yaml', variables={}):
        super().__init__(config_file, variables)
        self._n_channels = 0
        self._n_markers = 0

        # Timing information
        if 'sample_rate' in self._config_dict.keys():
            self._sample_rate = self._config_dict['sample_rate']
        else:
            self._sample_rate = 1e9
        self._seq_length = 0
        qtrl._DAC = self

    def load(self):
        super().load()   # TODO: useless, why keep?

    @property
    def n_channels(self):
        """Return the number of available channels"""
        return self._n_channels

    def flush(self):
        raise NotImplementedError("Subclasses should implement this!")

    def sample_rate(self, sr=None):
        """Returns the sample rate of the box, currently defaults to 1e9"""
        return self._sample_rate

    def write_sequence(self, seq):
        """Write a Sequence to the DAC channels"""
        raise NotImplementedError("Subclasses should implement this!")

    def run(self, state=1):
        """Start playing out of the sequence from waveform memory."""
        raise NotImplementedError("Subclasses should implement this!")

    def close(self):
        raise NotImplementedError("Subclasses should implement this!")
