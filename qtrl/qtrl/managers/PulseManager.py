# Copyright (c) 2018-2019, UC Regents

import numpy as np
from .ManagerBase import ManagerBase
from qtrl.utils.config import NoSetting
from qtrl.sequencer import VirtualEnvelope, UniquePulse, pulse_library_constructor
from ..sequencer.yaml_loader import recursive_env_constructor

import logging
log = logging.getLogger('qtrl.PulseManager')


class PulseManager(ManagerBase):
    """Thin wrapper over a config_file to help with pulse management.
    This defaults to not reloading from the config file unless a load()
    is called explicitly. This is a huge performance hit. TODO<wlav>: why?
    """

    def __init__(self, config_file='Pulses.yaml', variables={}):
        super().__init__(config_file, variables)

    def load(self):
      # TODO: why reload, given that the base constructor already loads?
        super().load()

      # the _config_dict is a private variable of the Config base class,
      # which is only true if using a backwards compatible settings
        try:
            recursive_env_constructor(self._config_dict)
            pulse_library_constructor(self._config_dict, prefix='')
            pulse_library_constructor(self._config_dict, suffix='_ef')
        except AttributeError:
            from qtrl.settings import Settings
            if Settings.setup != Settings.ONLINE:
                log.error('failed to update base class _config_dict')
            # else, we're good

    def get(self, key, default=NoSetting, reload_config=False):

        # Test for z gates:

        key_pair = key.split('/')
        # if it looks like 'Q*/Z(float)' assume it is a z gate
        if len(key_pair) > 1 and key_pair[1][0] == 'Z' and \
               key_pair[0] in self.keys(reload_config=False):
            try:
                phase = float(key_pair[1][1:]) / 180 * np.pi
                freq = self._variables[key_pair[0]]['mod_freq']
                return [z_pulse(freq, phase)]
            except (ValueError, KeyError):
                pass

        # if it is not a z gate, try to get the usual pulses
        return super().get(key, default, reload_config)

    def __getitem__(self, key):
        return self.get(key, reload_config=False)


def z_pulse(freq, phase):
    """Construct a virtual pulse - Z gate.
    """

    env = VirtualEnvelope(phase, None)
    return UniquePulse(env, freq=freq, channel=0, subchannel=0)
