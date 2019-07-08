# Copyright (c) 2018-2019, UC Regents

from .ManagerBase import ManagerBase


class VariableManager(ManagerBase):
    """Thin wrapper over a config_file to help with modulation management.
    """

    def __init__(self, varfile='Variables.yaml'):
        super().__init__(varfile)

    def load(self):
        super().load()

        try:
            qubit_lo = self.get('hardware/qubit_LO')
            readout_lo = self.get('hardware/readout_LO')
        except KeyError:
            return

        for k in self.keys(''):
            if 'Q' in k and 'freq' in self.get(k):
                freq = self.get(f'{k}/freq')
                self.set(f'{k}/mod_freq', -qubit_lo+freq)
                anharm = self.get(f'{k}/anharmonicity')
                self.set(f'{k}/mod_freq_ef', -qubit_lo + freq + anharm)

        for k in self.keys(''):
            if 'Q' in k and 'res_freq' in self.get(k):
                freq = self.get(f'{k}/res_freq')
                self.set(f'{k}/res_mod_freq', -readout_lo+freq)
