# Copyright (C) 2018-2019  UC Regents

import os, re
import logging
import warnings
import json
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from ruamel.yaml import load, dump, CSafeLoader, CSafeDumper, error
try:
    # This import is for pretty printing in Jupyter
    from IPython.lib.pretty import for_type, _ordereddict_pprint
    _has_ipython = True
except ImportError:
    _has_ipython = False
from .util import replace_vars
from qtrl.settings import Settings
from qtrl.sequencer import pulse_library

warnings.simplefilter('ignore', error.MantissaNoDotYAML1_1Warning)

log = logging.getLogger('qtrl.config')

# This class exists to provide a unique label so that we can use None as an
# optional input for the get function
class NoSetting:
    pass


class Config:
    pass


#-----------------------------------------------------------------------------
# Original qtrl configuration using YAML files.

class YAMLConfig(Config):
    """Loads and stores a YAML _config_dict file, provides a dictionary like
    interface to the _config_dict, with the additions of load, save, and variables
    """

    def __init__(self, config_file, variables={}):
        self._config_file = os.path.join(Settings.config_path, config_file)
        self._variables   = variables
        self._config_dict = None
        self._config_raw  = None
        self.load()

    def load(self):
        # Try to open the _config_dict file, if not found, create it and warn the user
        try:
            with open(self._config_file, 'r') as f:
                self._config_raw = load(f, CSafeLoader) or {}
        except FileNotFoundError:
            self._config_raw = dict()

        self._config_dict = deepcopy(self._config_raw)

        if 'Variables' in self._config_dict:
            self._variables = self._config_dict.pop('Variables')

        replace_vars(self._config_dict, self._variables)

    def get(self, key, default=NoSetting, reload_config=False):
        """Get a key from the _config_dict, the _config_dict can be traversed with '/'
        a default response can be given, and if the key not found will be returned.
        Otherwise this raises a KeyError if the key is not present.
        
        Examples:
            conf = Config("Example.yaml")
            conf.get("description")
            conf.get("Alazar/Key_not_here", default=0)
            conf.get("Hardware/ADCS/Some/Deep/Value")
        
        """

        if reload_config:
            self.load()

        try:
            keys = key.split('/')
        except AttributeError:
            raise KeyError()

        if keys[-1] is '':
            keys = keys[:-1]

        result = self._config_dict
        for i, key in enumerate(keys):
            try:
                result = result[key]
            except KeyError:
                if default != NoSetting:
                    return default
                else:
                    raise KeyError('/'.join(keys[i:]))

        return result

    def save(self):
        conf_str = dump(self._config_raw, Dumper=CSafeDumper, default_flow_style=False)
        with open(self._config_file, 'w') as f:
            f.write(conf_str)
    
    def set(self, key, value, reload_config=False, save_config=False):
        """Set values in the _config_dict; can be traversed with '/'.
        """

        if reload_config:
            self.load()

        try:
            keys = key.split('/')
        except AttributeError:
            raise KeyError()

        if keys[-1] is '':
            keys = keys[:-1]

        # We need to traverse both the raw and variable replaced 
        # _config_dict dictionary tree, and replace both simultaneously
        config = self._config_dict
        config_raw = self._config_raw
        for key in keys[:-1]:
            try:
                if not isinstance(config[key], dict):
                    config[key] = {}
                config = config[key]
            except KeyError:  # build the tree if it isn't present
                config[key] = {}
                config = config[key]
        
            try:
                if not isinstance(config_raw[key], dict):
                    config_raw[key] = {}
                config_raw = config_raw[key]
            except KeyError:  # build the tree if it isn't present
                config_raw[key] = {}
                config_raw = config_raw[key]

        # there is a bug in pyyaml load which fails for ndarrays
        # we can just cast them to a list, which improves readability as well
        if isinstance(value, np.ndarray):
            value = value.tolist()

        # finally update the value
        config[keys[-1]] = value
        config_raw[keys[-1]] = value

        replace_vars(self._config_dict, self._variables)

        if save_config:
            self.save()

    def keys(self, key='', reload_config=False):
        """Get the keys at the given location; can be traversed with '/'.
        """

        if reload_config:
            self.load()

        try:
            keys = key.split('/')
        except AttributeError:
            raise KeyError()

        if keys[-1] is '':
            keys = keys[:-1]

        result = self._config_dict
        for i, key in enumerate(keys):
            try:
                result = result[key]
            except KeyError:
                    raise KeyError('/'.join(keys[i:]))

        return result.keys()

    def __delitem__(self, key):
        """Delete values in the Config; can be traversed with '/'.
        """

        self.load()

        try:
            keys = key.split('/')
        except AttributeError:
            raise KeyError()

        if keys[-1] is '':
            keys = keys[:-1]

        # We need to traverse both the raw and variable replaced
        # _config_dict dictionary tree, and replace both simultaneously
        config = self._config_dict
        config_raw = self._config_raw
        for key in keys[:-1]:
            config = config[key]

            config_raw = config_raw[key]

        # finally update the value
        del config[keys[-1]]
        del config_raw[keys[-1]]

        self.save()

    # All function below reference the _config_dict as though the Config is this dictionary
    def __iter__(self):
        return self._config_dict.__iter__()

    def __next__(self):
        return self._config_dict.__next__()

    def __repr__(self):
        return self._config_dict.__repr__()

    def __str__(self):
        return self._config_dict.__str__()

    def __len__(self):
        return len(self._config_dict)

    def items(self):
        return self._config_dict.items()

    def __getitem__(self, key):
        return self.get(key, reload_config=False)

    def __setitem__(self, key, item):
        return self.set(key, item)


# Adding this to enable pretty printing in Jupyter
if _has_ipython:
    for_type(YAMLConfig, _ordereddict_pprint)


#-----------------------------------------------------------------------------
# Proposed Qubic configuration based on JSON files
# TODO: this, too, mixes configuration with implementation
# TODO: drop the sorting and use direct insertion instead

class _ConfigQubit:
    def __init__(self, paradict):
        for k in paradict:
            if k in ['freq','freq_ef','readfreq']:
                setattr(self, k, paradict[k])
            # TODO: why do these terms make sense/why hard-wire them?
            elif k in ['rabi','rabi_ef','180','90','180_ef','90_ef','readout','readoutdrv']:
                setattr(self, k, _ConfigQGate(
                    paradict[k] if isinstance(paradict[k], list) else [paradict[k]], qubit=self))

        self.qdrv = []
        self.rdrv = []
        self.read = []
        self.mark = []

    def get(self, key):       # backwards compatibility
        return getattr(self, key)


class _ConfigQGate:
    def __init__(self, all_paradict, chip, name):
        self.chip     = chip
        self.name     = name
        self.pulses   = []
        self.paralist = []
        for paradict in all_paradict:
            pulse = _ConfigQGatePulse(chip=self.chip, paradict=paradict)
            pulse.setfreq(chip=self.chip)
            self.pulses.append(pulse)
            self.paralist.append(pulse.paradict)
        #self.paralist.sort()     # TODO: what's the point of this sort?
        self.tstart = 0.0

    def settstart(self, tstart):
        self.tstart = tstart
        for pulse in self.pulses:
            pulse.settstart(tstart)

    def getval(self, dt, flo=0, mod=True):
        tval = None
        for pulse in self.pulses:
            newtval = pulse.val(dt=dt, flo=flo, mod=mod)
            if tval == None:
                tval = newtval
            else:
                tval.add(newtval)
        return tval

    def get_pulses(self):
        return self.pulses

    def tlength(self):
        return max([pulse.tlength() for pulse in self.pulses])

    def tend(self):
        return self.tstart+self.tlength()

    def modify(self, paradict=None):
        newlist = copy.deepcopy(self.paralist)
        if paradict:
            for pd in newlist:
                pd.update(paradict)
        return _ConfigQGate(newlist, self.chip, name=self.name)

    def dup(self):
        return self.modify()

    def pcalc(self, dt=0, padd=0, freq=None):
        return np.array([p.pcarrier+2*np.pi*(freq if freq else p.fcarrier)*dt+padd for p in self.pulses])


class _ConfigQChip:
    def __init__(self, paradict):
        self.qubits = {}
        for k in paradict['Qubits']:
            self.qubits.update({k : _ConfigQubit(paradict['Qubits'][k])})
        self.gates = {}
        for k in paradict['Gates']:
            self.gates.update({k : _ConfigQGate(paradict['Gates'][k], chip=self, name=k)})

        log.info('configured chip for %d qubits' % len(self.qubits))

    def updatefreq(self, freqname, val):
        q, f = freqname.split('.')
        setattr(self.qubits[q], f, val)

    def getfreq(self, freqname):
        freq = freqname
        if isinstance(freqname, str):
            m = re.match(r'(?P<qname>\S+)\.(?P<fname>\S+)', freqname)
            if m:
                q = m.group('qname')
                f = m.group('fname')
                if q in self.qubits:
                    if hasattr(self.qubits[q], f):
                        freq = getattr(self.qubits[q], f)
        return freq

    def getdest(self, destname):
        q, d = destname.split('.')
        return getattr(self.qubits[q], d)

    def get(self, key):       # backwards compatibility
        try:
            return self.qubits[key]
        except KeyError:
            return self.gates[key]


class _ConfigQGatePulse:
    def __init__(self, chip, paradict):
        self.paradict = {}
        self.chip = chip
        self.target = None
        for k in paradict:
            if k in ['amp', 'twidth', 't0', 'pcarrier', 'dest', 'fcarrier']:
                try:
                    v = eval(paradict[k])
                except Exception:
                    v = paradict[k]
                setattr(self, k, v)
                self.paradict[k] = paradict[k]
                if k == 'fcarrier' and type(self.fcarrier) == str:
                  # todo: why does fcarrier come in str and float flavors?
                    self.target = self.fcarrier[:self.fcarrier.find('.')]
            elif k in ['env']:
                setattr(self, k, _ConfigQEnvelop(paradict[k]))
                self.paradict[k] = sorted(paradict[k], key=lambda x: x['env_func'])
            else:
                warnings.warn("unknown key for _ConfigQGatePulse (%s)" % k)

      # TODO: width of a pulse is taken from the envelope in qtrl, which
      # seems to be not the right place (?); for now store it here so
      # that it is available to the sequencer
        try:
            self.env.width = self.twidth
        except AttributeError:
            pass

    def settstart(self, tstart):
        self.tstart = tstart+self.t0

    def setfreq(self, chip):
        self.fcarrier = chip.getfreq(self.fcarrier)

    def val(self, dt, flo=0, mod=True):
        fcarrier = self.chip.getfreq(self.fcarrier)
        fif = 0 if self.fcarrier == 0 else (fcarrier-flo)
        tv = self.env.env_val(dt=dt, twidth=self.twidth, fif=fif,
                              pini=self.pcarrier, amp=self.amp, mod=mod)
        tv.tstart(self.tstart)
        return tv

    def tend(self):
        return self.tstart+self.twidth

    def tlength(self):
        return self.twidth

    def __ne__(self,other):
        return not self.__eq__(other)

    def __eq__(self,other):
        return self.amp == other.amp and self.twidth == other.twidth and \
               self.dest == other.dest and self.env == other.env

    def __hash__(self):
        return hash(str({k:self.paradict[k] for \
                         k in ("amp", "twidth", "t0", "dest") if k in self.paradict}))

    def timeoverlap(self, other):
        overlap = False
        if self.tstart < other.tstart:
            overlap = self.tend() < other.start
        else:
            overlap = other.tend() < self.tstart
        return overlap

    def __getattr__(self, attr):   # for backwards compatibility
        try:
            return self.paradict[attr]
        except KeyError:
          # a couple of special cases
          if attr == 'freq':
              return self.chip.getfreq(self.target+'.freq')
          elif attr == 'channel' or attr == 'subchannel':
            # TODO: is this supported by qubic??
              return 0
          elif attr == 'envelope':
              return self.env

        return object.__getattribute__(self, attr)

    def _replace(self, **kwds):    # for backwards compatibility
      # TODO: should this copy like the namedtuples do?
        for key, value in kwds.items():
            setattr(self, key, value)
        return self


class _ConfigQEnvelop:
    def __init__(self, env_desc):
        if not isinstance(env_desc, list):
            env_desc = [env_desc]
        self.env_desc = env_desc

    def env_val(self, dt, twidth, fif=0, pini=0, amp=1.0, mod=True):
        vbase = None
        ti = None
        for env in self.env_desc:
            if vbase is None:
                ti, vbase = getattr(pulse_library, 'tv_'+env['env_func'])(
                    dt=dt, twidth=twidth, **env['paradict'])
            else:
                ti1, vbasenew = (getattr(pulse_library, 'tv_'+env['env_func'])(
                    dt=dt, twidth=twidth, **env['paradict']))
                if any(ti != ti1):
                    warnigns.warn('different time!!?')
                vbase = vbase*vbasenew
        if mod:
            vlo = np.cos(2*np.pi*fif*ti+pini)
        else:
            vlo = 1
        val = amp*vlo*vbase
        tv = _ConfigQTV(ti, val)
        return tv

    def __eq__(self, other):
        return sorted(self.env_desc) == sorted(other.env_desc)

    def __getattr__(self, attr):   # for backwards compatibility
        try:
            return self.env_desc[0][attr]
        except KeyError:
            pass
        if attr == 'kwargs':
            return {}
        return object.__getattribute__(self, attr)

    def _asdict(self):        # backwards compatibility
        return OrderedDict()  # TODO: fill


class _ConfigQTV:
    def __init__(self, t, val):
        self.tv = {}
        if len(t) == len(val):
            for i in range(len(t)):
                self.append(t[i], val[i])
        self.val = val

    def append(self, t, val):
        if t in self.tv:
            self.tv[t] += val
        else:
            self.tv[t] = val

    def add(self, tval):
        for it in tval.tv:
            self.append(it, tval.tv[it])

    def tstart(self, tstart):
        newtv = dict((t+tstart, val) for (t, val) in self.tv.items())
        self.tv = newtv

    def tvval(self):
        self.t = []
        self.val = []
        for i in sorted(self.tv):
            self.t.append(i)
            self.val.append(self.tv[i])
        return np.array(self.t), np.array(self.val)

    def nvval(self, dt):
        t, v = self.tvval()
        n = np.array([int(round(it/dt)) for it in t])
        return n, v


class JSONConfig(Config):
    def __init__(self, config_file, variables=None):
        with open('qubitcfg.json') as jfile:
            chipcfg = json.load(jfile)
        self._qchip = _ConfigQChip(chipcfg)

    def load(self):
      # load already occurred in constructor
        pass

    def get(self, key):
        """Lookup the configuration value associated with 'key'.
        """

      # Since the origincal qtrl code split the configuration across one
      # file per manager, but the qubic code does not, _all_ configuration
      # from _all_ managers will come here. Assume for now that all keys
      # are unique. If not, can still use the class name to arbitrate.
        try:
            current = self._qchip
            for kp in key.split('/'):
                current = current.get(kp)
            return current
        except (AttributeError, KeyError):
            pass

      # difference of convention for gates (eg. Q0/rabi v.s. Q0rabi)
        gr = self._qchip.get(key.replace('/', ''))
        if hasattr(gr, 'get_pulses'):
            return gr.get_pulses()[0]
        return gr

_json_configs = dict()
def load_json_config(config_file, variables=None):
    if not config_file in _json_configs:
        _json_configs[config_file] = JSONConfig(config_file, variables)
    return _json_configs[config_file]
