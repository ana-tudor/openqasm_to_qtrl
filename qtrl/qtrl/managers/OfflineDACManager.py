# Copyright (c) 2018-2019, UC Regents

from .DACManager import DACManager


class OfflineDACManager(DACManager):

    def __init__(self, config_file='DAC.yaml', variables={}):
        super().__init__(config_file, variables)

