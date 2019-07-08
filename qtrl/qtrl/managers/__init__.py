# Copyright (c) 2018-2019, UC Regents

import warnings
import importlib, warnings
from qtrl.settings import Settings

__all__ = list()


# for backwards compatibility, default to online
if Settings.setup is None:
    Settings.setup = Settings.ONLINE

def _errmsg(what, why):
    warnings.warn("Cannot import {what} [{why}].".format(what=what, why=why))

COMMON_MANAGERS = [
    ('VariableManager',       'VariableManager'),
    ('MetaManager',           'MetaManager'),
    ('InstrumentManager',     'InstrumentManager'),
    ('PulseManager',          'PulseManager'),
    ('DataManager',           'DataManager'),
]

ONLINE_MANAGERS = [
    ('KeysightADCManager',    'KeysightADCManager'),
    ('KeysightDACManager',    'KeysightDACManager'),
    ('AlazarADCManager',      'AlazarADCManager'),
]

OFFLINE_MANAGERS = [
    ('OfflineADCManager',     'OfflineADCManager'),
    ('OfflineDACManager',     'OfflineDACManager'),
]

QUBIC_MANAGERS =  [
    ('QubicADCManager',       'QubicADCManager'),
    ('QubicDACManager',       'QubicDACManager'),
]

# TODO: it doesn't seem that the set of functions from InstrumentManager should
# be loaded into here?
try:
    from .InstrumentManager import qcodes_inst, load_device, is_device, find_dev_config
except (ImportError, OSError, ModuleNotFoundError) as e:
    _errmsg('InstrumentManager', str(e))

def _import_managers(managers, modall, silent=False):
    for smod, smgr in managers:
        try:
            mod = importlib.import_module('.'+smod, __name__)
            mgr = getattr(mod, smgr)
            globals()[smgr] = mgr
            modall.append(smgr)
        except (ImportError, OSError, ModuleNotFoundError) as e:
            if not silent:
                _errmsg(smod, str(e))

_import_managers(COMMON_MANAGERS, __all__)
if Settings.setup == Settings.ONLINE:
    _import_managers(ONLINE_MANAGERS, __all__)
elif Settings.setup == Settings.OFFLINE:
    _import_managers(OFFLINE_MANAGERS, __all__)
    _import_managers(ONLINE_MANAGERS, __all__, silent=True)
elif Settings.setup == Settings.QUBIC:
    _import_managers(QUBIC_MANAGERS, __all__)
