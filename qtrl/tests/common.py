# Copyright (c) 2018-2019, UC Regents

import sys


def setup_managers(which):
    # Settings currently determine the class hierarchy, as the configuration
    # in qtrl is hard-wired that way. Thus, for differently configured code (in
    # particular, qubic), those classes need to be fully reloaded.

    if 'qtrl.managers' in sys.modules:
        import qtrl.settings
        if qtrl.settings.Settings.setup != which:
            import importlib
            importlib.reload(sys.modules['qtrl.settings'])
            qtrl.settings.Settings.setup = which
            importlib.reload(sys.modules['qtrl.utils.config'])
            for modname in ['ManagerBase', 'ADCManager', 'DACManager',
                    'VariableManager', 'PulseManager', 'InstrumentManager', 'MetaManager']:
                try:
                    importlib.reload(sys.modules['qtrl.managers.'+modname])
                except KeyError:
                    pass
            importlib.reload(sys.modules['qtrl.managers'])
    else:
        import qtrl.settings
        qtrl.settings.Settings.setup = which
