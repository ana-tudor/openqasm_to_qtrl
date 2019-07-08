# Copyright (c) 2018-2019, UC Regents

import copy
import qtrl
from .ManagerBase import ManagerBase
from .ADCManager import ADCManager
from ..remote import Connection


class RemoteManager(Connection):
    """This is a holder to provide standardization of a full configuration"""

    def __init__(self, manager_dict, ip='localhost', port=9000):
        """Accepts a dictionary containing a set of managers.
        These managers are then made accessible through RemoteManager.*manager_name*."""

        self._managers = manager_dict
        for manager in manager_dict:
            assert isinstance(manager_dict[manager], ManagerBase)
            self.__setattr__(manager, manager_dict[manager])

        if 'ADC' in manager_dict and isinstance(self._managers['ADC'], ADCManager):
            self.acquire = self._managers['ADC'].acquire

        # enable this to be accessible to the rest of qtrl
        qtrl._cfg = self
        super().__init__(self, ip, port)

    def load(self):
        """Call all of the managers load functions"""
        if 'variables' in self._managers:
            self._managers['variables'].load()

        for manager in self._managers:
            self._managers[manager].load()

    def save(self):
        """Call all of the managers save functions"""
        for manager in self._managers:
            self._managers[manager].save()

    @property
    def config(self):
        """Traverse all of the managers and record the config files in their original format.
        Returns a dictionary of all the entries."""
        config_dict = dict()
        for manager in self._managers:
            key = (self._managers[manager]._config_file, manager)
            config_dict[key] = copy.deepcopy(self._managers[manager]._config_raw)

        return config_dict
