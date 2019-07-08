# Copyright (c) 2018-2019, UC Regents

import copy
import qtrl
from .ManagerBase import ManagerBase


class MetaManager:
    """This is a holder to provide standardization of a full configuration"""

    def __init__(self, settings):
        """Accepts a dictionary containing a set of managers.
        These managers are then made accessible through MetaManager.*manager_name*.
        If DAC/ADC are provided as keys, make write_sequence/acquire accessible
        through MetaManager.acquire/write_sequence.
        """

        if isinstance(settings, dict):
            # for backwards compatibility
            from qtrl.settings import Settings
            Settings.setup = Settings.ONLINE     # fills in online defaults
            for key, value in settings.items():
                setattr(Settings, key, value)
            settings = Settings

        self._managers = settings.managers()
        for name, manager in self._managers.items():
            from qtrl.settings import Settings
            assert isinstance(manager, ManagerBase)
            self.__setattr__(name, manager)

        if 'ADC' in self._managers and hasattr(self._managers['ADC'], 'acquire'):
            self.acquire = self._managers['ADC'].acquire

        if 'DAC' in self._managers and hasattr(self._managers['DAC'], 'write_sequence'):
            self.write_sequence = self._managers['DAC'].write_sequence

        # enable this to be accessible to the rest of qtrl
        qtrl._cfg = self

    def load(self):
        """Call all of the managers load functions.
        """

        if 'variables' in self._managers:
            self._managers['variables'].load()

        for manager in self._managers:
            self._managers[manager].load()

    def save(self):
        """Call all of the managers save functions.
        """

        for manager in self._managers:
            self._managers[manager].save()

    @property
    def config(self):
        """Traverse all of the managers and record the config files in their original
        format. Returns a dictionary of all the entries.
        """

        config_dict = dict()
        for manager in self._managers:
            key = (self._managers[manager]._config_file, manager)
            config_dict[key] = copy.deepcopy(self._managers[manager]._config_raw)

        return config_dict
