# Copyright (c) 2018-2019, UC Regents

from .ADCManager import ADCManager


class OfflineADCManager(ADCManager):

    def __init__(self, config_file='ADC.yaml', variables={}):
        super().__init__(config_file, variables)
