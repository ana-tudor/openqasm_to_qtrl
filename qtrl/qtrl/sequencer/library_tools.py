# Copyright (c) 2018-2019, UC Regents

from collections import OrderedDict
from ..sequencer import sequencer
import copy
import numpy as np
from itertools import product


def pulse_library_constructor(pulses, prefix='', suffix=''):
    # Construct the pulse list using the prototype 90 and 180 pulses
    if pulses is not None:
        for qubit in [q for q in pulses.keys() if 'Q' in q]:
            q90_name = '{}90{}'.format(prefix,suffix) if (prefix is not '' or suffix is not '') else 90
            q180_name = '{}180{}'.format(prefix,suffix) if (prefix is not '' or suffix is not '') else 180
            if q90_name in pulses[qubit] and q180_name in pulses[qubit]:
                p90 = pulses[qubit][q90_name]
                p180 = pulses[qubit][q180_name]
                pulses[qubit].update(pulse_constructor(p90, p180, prefix=prefix, suffix=suffix))


def pulse_constructor(pulse_prototype_90, pulse_prototype_180, prefix="", suffix=""):
    """This takes two pulses, one representative of a 90 degree rotation, and 180,
    it then uses these as a basis to construct a full set of pulses using our standard
    contruction, returning X/Y/Z 90/180/270 and the Identity as an ordered dictionary.

    Additionally accepts a prefix so that a custom prefix can be added to pulse names"""

    # composite pulses are treated seperately
    if isinstance(pulse_prototype_90, list):
        assert isinstance(pulse_prototype_180,
                          list), "Either both pulses have to be lists or neither, combinations not allowed"
        assert len(pulse_prototype_90) == len(pulse_prototype_180), "Both pulse lists have to be the same length"

        pulse_lists = [_pulse_constructor_helper_(pulse_prototype_90[i], pulse_prototype_180[i], prefix,suffix) for i in
                       range(len(pulse_prototype_90))]

        pulses = pulse_lists[0]

        # convert all entries to lists
        for k in pulses:
            pulses[k] = [pulses[k]]

        for dic in pulse_lists[1:]:
            for key in dic:
                if 'Z' not in key:
                    # if not isinstance(dic[key].envelope, sequencer.VirtualEnvelope):
                    pulses[key].append(dic[key])
        return pulses

    # if just a normal pulse directly call the constructor helper
    else:
        return _pulse_constructor_helper_(pulse_prototype_90, pulse_prototype_180, prefix, suffix)


def _pulse_constructor_helper_(pulse_prototype_90, pulse_prototype_180, prefix="", suffix=""):
    y_phase = -np.pi / 2.
    mod_freq = pulse_prototype_90.freq
    channel = pulse_prototype_90.channel

    # make the dict we will fill up
    pulses = OrderedDict()

    # First we construct an identity operation
    env = sequencer.VirtualEnvelope(0.0, None)
    new_pulse = sequencer.UniquePulse(env,
                                           freq=mod_freq,
                                           channel=channel,
                                           subchannel=0)
    pulses[prefix + 'I'+suffix] = new_pulse

    # Now we go through all combinations of XY 90,180,270 and build the rest
    for rotation, angle in product(['X', 'Y'], [-90, 90, 180, 270]):

        # Construct the name of the pulse
        p_name = '{}{}{}{}'.format(prefix, rotation, angle,suffix)

        if angle != 180:
            new_pulse = copy.deepcopy(pulse_prototype_90)
        else:
            new_pulse = copy.deepcopy(pulse_prototype_180)

        if not isinstance(new_pulse.envelope, sequencer.VirtualEnvelope):
            phase = 0.0
            phase += y_phase if rotation == 'Y' else 0.0
            phase += np.pi if angle == 270 else 0.0
            phase += np.pi if angle == -90 else 0.0

            # if 'phase' isn't in the pulse kwargs, add it as a list
            if 'phase' not in new_pulse.envelope.kwargs:
                new_pulse.envelope.kwargs['phase'] = []
            # if 'phase' IS in the kwargs, but is not a list, make it a list
            elif not isinstance(new_pulse.envelope.kwargs['phase'], list):
                new_pulse.envelope.kwargs['phase'] = [new_pulse.envelope.kwargs['phase']]

            new_pulse.envelope.kwargs['phase'].append(float(phase))

        # Add the pulses to the dictionary
        pulses[p_name] = new_pulse
    return pulses
