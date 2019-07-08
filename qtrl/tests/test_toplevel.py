# Copyright (c) 2018-2019, UC Regents

import py, os, sys
import common
from pytest import raises

CONFIG_PATH = os.path.join(os.path.pardir, 'examples', 'Config_Example')


class TestTOPLEVEL:
    def setup_class(cls):
        pass

    def test01_managers(self):
        """Test importing from qtrl.managers"""

        import qtrl.managers as mgrs
        from qtrl.managers import __all__

        # the following should now (at minimum) be available
        for module, name in mgrs.COMMON_MANAGERS:
            assert name in __all__

    def test02_meta_manager_from_settings_online(self):
        """Initializing the meta manager from settings (online)"""

        common.setup_managers('online')
        from qtrl.settings import Settings
        Settings.config_path = CONFIG_PATH

        from qtrl.managers import MetaManager

        cfg = MetaManager(Settings)

    def test03_meta_manager_from_settings_offline(self):
        """Initializing the meta manager from settings (offline)"""

        common.setup_managers('offline')
        from qtrl.settings import Settings
        Settings.config_path = CONFIG_PATH

        from qtrl.managers import MetaManager

        cfg = MetaManager(Settings)

    def test04_meta_manager_explicit_offline(self):
        """Explicit initialization of the meta manager (offline)"""

        common.setup_managers('offline')
        from qtrl.settings import Settings
        Settings.config_path = CONFIG_PATH

        from qtrl.managers import MetaManager, VariableManager, InstrumentManager, \
             PulseManager, DataManager, OfflineADCManager, OfflineDACManager

        var = VariableManager()
        cfg = MetaManager({'variables': var,
                   'devices': InstrumentManager('Devices.yaml', var),
                   'ADC': OfflineADCManager('ADC.yaml', var),
                   'DAC': OfflineDACManager('DAC.yaml', var),
                   'pulses': PulseManager('Pulses.yaml', var)
                   })
