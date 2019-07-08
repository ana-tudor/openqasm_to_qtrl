# Copyright (c) 2018-2019, UC Regents

from .sequencer import *
import copy
from ..utils.util import replace_vars
from ruamel.yaml import load, dump


envelope_types = {'PulseEnvelope': PulseEnvelope,
                  'VirtualEnvelope': VirtualEnvelope}


def pulse_to_dict(pulse):
    """Convert a pulse object into a dictionary"""
    # convert the pulse into a list of pulses if not already
    pulse = [pulse] if not isinstance(pulse, list) else pulse
    res = []
    for p in pulse:
        d = {}
        d.update(p._asdict())
        d['envelope'] = {'type': type(d['envelope']).__name__,
                         'properties':d['envelope']._asdict()}
        res.append(d)
    if len(res) == 1:
        res = res[0]
    return copy.deepcopy(res)


def dict_to_pulse(pulse_dict):
    """Convert a dictionary containing one pulse definition into a pulse"""
    pulse_dict = copy.deepcopy(pulse_dict)
    env_desc = pulse_dict.pop("envelope")
    envelope = envelope_types[env_desc['type']](**env_desc['properties'])
    pulse = UniquePulse(envelope=envelope, **pulse_dict)
    return pulse


def pulses_to_yaml(pulse_dict):
    """Converts a list of pulses into a yaml string"""
    yaml_dict = {}
    for pulse in pulse_dict:
        yaml_dict[pulse] = pulse_to_dict(pulse_dict[pulse])

    return dump(yaml_dict)


def yaml_to_pulses(yaml_str):
    """Convert from a yaml string to a list of pulses"""
    pulses = load(yaml_str)
    output = {}
    for pulse in pulses:
        output[pulse] = dict_to_pulse(pulses[pulse])

    return output


def load_pulse_yaml(filename, variables):
    """Loads a yaml file containing pulse definitions.
    Returns a dictionary which contains all pulses found
    in the YAML file.
    """

    with open(filename, 'rb') as f:
        pulses = load(f.read())

    if pulses is None:
        raise Exception('Pulse file {} appears empty.'.format(filename))

    replace_vars(pulses, variables)
    recursive_env_constructor(pulses)

    return pulses


def recursive_env_constructor(dic, node_key=None, node=None):
    """Traverse a dictionary and look for anything which looks like a pulse envelope
    This is decided by whether is has an 'envelope' key and is a dictionary.
    If it passes this simple test we attempt to build a pulse envelope out of it.

    This changes a dictionary in place, user beware.

    Typical use:
    recursive_env_constructor(pulse_dictionary)
    """
    # 3 cases we care about, if we recieved a dictionary that looks like a pulse
    if isinstance(dic, dict) and 'envelope' in dic:
        node[node_key] = dict_to_pulse(dic)

    # a dictionary that doesn't look like a pulse, we just go deeper
    elif isinstance(dic, dict):
        for key in dic:
            recursive_env_constructor(dic[key], key, dic)

    # or a list, which we need to iterate over
    elif isinstance(dic, list):
        for i, item in enumerate(dic):
            recursive_env_constructor(item, i, dic)

