# Copyright (c) 2018-2019, UC Regents

import os


"""
Top-level settings.

 - setup: online, offline, or qubic
     online:  existing experimental setup
     offline: testing harness
     qubic:   experimental qubic hookups

 - config_path: Config directory to load from
     Can be either absolute or relative (under qtrl/server,
     or qtrl/projects)

 - set of managers:
     variables
     devices
     ADC
     DAC
     pulses

"""

__all__ = ['Settings']

VARIABLES = 'variables'

class _Settings(object):
    default = -1

    __config = {
        'setup'       : None,
        'config_path' : ''
    }

    __managers = {
        VARIABLES     : 'VariableManager',
        'devices'     : 'InstrumentManager',
        'ADC'         : default,
        'DAC'         : default,
        'pulses'      : 'PulseManager',
    }

    OFFLINE = 'offline'
    ONLINE  = 'online'
    QUBIC   = 'qubic'

    __slots__ = list(__config.keys()) + list(__managers.keys())

    def __init__(self, **kwds):
        for dct in [self.__config, self.__managers]:
            for key, value in dct.items():
                super().__setattr__(key, value)

    def __setattr__(self, attr, value):
        if attr == 'setup':
            if not value in ['online', 'offline', 'qubic']:
                raise ValueError('%s not a valid choice for %s' % (str(value), name))

            # override default values based on configuration choice
            if value == self.ONLINE:
                if self.ADC == self.default:
                    self.ADC = 'AlazarADCManager'
                if self.DAC == self.default:
                    self.DAC = 'KeysightDACManager'
            elif value == self.OFFLINE:
                if self.ADC == self.default:
                    self.ADC = 'OfflineADCManager'
                if self.DAC == self.default:
                    self.DAC = 'OfflineDACManager'
            elif value == self.QUBIC:
                if self.ADC == self.default:
                    self.ADC = 'QubicADCManager'
                if self.DAC == self.default:
                    self.DAC = 'QubicDACManager'

        elif attr == 'config_path':
            if not os.path.isabs(value):
                base = os.path.join(os.path.dirname(__file__), os.path.pardir)
                for loc in ['projects', 'server']:
                    p = os.path.join(base, loc, value, 'Config')
                    if os.path.exists(p):
                        value = os.path.abspath(p)

        super().__setattr__(attr, value)

    def __getitem__(self, item):   # for backwards compatability
        return getattr(self, item)

    def managers(self):            # for backwards compatability
        if self.setup is None:
            self.setup = self.ONLINE

        import importlib
        mgrs = importlib.import_module('qtrl.managers')

        managers = self.__managers.copy()

        mgr = managers.pop(VARIABLES)
        varmgr = getattr(self, VARIABLES)
        if type(varmgr) == str:
            varmgr = getattr(mgrs, mgr)()

        dct = dict()
        dct[VARIABLES] = varmgr
        for mkey in managers.keys():
            mgr = getattr(self, mkey)
            if type(mgr) == str:
                try:
                    dct[mkey] = getattr(mgrs, mgr)(variables=varmgr)
                    setattr(self, mkey, dct[mkey])
                except (AttributeError, RuntimeError):
                    pass           # warnings will have been issued
            else:
                dct[mkey] = mgr
        return dct

Settings = _Settings()
