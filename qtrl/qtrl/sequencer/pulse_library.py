# Copyright (c) 2018-2019, UC Regents

import numpy as np
from collections.abc import Sequence


"""
Pulses function have 3 required parameters:
   o) width - the width in samples of the pulse
   o) amplitude - the peak amplitude of the pulse (this need not be used,
        but must be available); if amplitude is entered as an iterable,
        such as a list, all entries will be multiplied together and the
        result will be used as the actual amplitude
   o) phase - the complex phase of the pulse
        if phase is entered as an Iterable, such as a list, all entries will
        be summed together and the result will be used as the actual phase
    
The conversion of lists of amplitudes or phases is immensely useful for
record keeping.

An example of this:

Y pulses have a -pi/2 offset in phase, however if you wish to create an IQ
pulse using 2 channels of a DAC, you need to not only offset the phase by
-pi/2 for the Y pulse, but also offset one of the channel's phase such that
the output of the IQ mixed is properly nulled.

When a new pulse function is added, it should be included in the look-up
dictionary at the bottom of this file.
Additionally, a unit test should be added to qtrl/tests/test_pulse_library.py
"""

import logging
log = logging.getLogger('qtrl')


def _calc_amp_phase(amplitude, phase, **kwds):
  # if amplitude is included as an iterable, product all of the values
    amplitude = np.product(np.array(amplitude))
    amplitude *= kwds.pop('global_amp', 1.0)

  # if phase is included as an iterable, sum all of the values
    phase = np.sum(np.array(phase))
    phase += kwds.pop('global_phase', 0.0)

    if kwds:
        log.warning("unused keyword args: %s", str(list(kwds.keys())))

    return amplitude, phase


def square(width, amplitude=1.0, phase=0.0, **kwds):
    """A simple square pulse.
    """

    amplitude, phase = _calc_amp_phase(amplitude, phase, **kwds)
    return np.array(np.ones(width) * np.exp(1j*phase) * amplitude, dtype=np.complex64)

# TODO: consistent order of amp/phase/other args
def sin(width, amplitude=1.0, frequency=0, phase=0.0, sample_rate=2.5e9, **kwds):
    """A sin pulse, this is modulated independently from the modulation added
    by the sequencer. This is primarily used to construct pulses which do not
    sample from the global phase of the sequence.
    """

    amplitude, phase = _calc_amp_phase(amplitude, phase, **kwds)
    unit_less = 2 * np.pi * np.array(frequency / sample_rate)
    return np.array(amplitude * np.exp(1j * (phase + np.arange(width) * unit_less - np.pi / 2.)), dtype=np.complex64)

def half_sin(width, amplitude=1.0, phase=0.0, **kwds):
    """sin(x) for pi in the given width.
    """

    step = 1.0/(len(t)-1)
    return np.sin(np.pi*np.arange(0, 1. + step/2.0, step)).astype(np.complex64)

def cos_env(width, amplitude=1.0, phase=0.0, **kwds):
    """This is an envelope which is one half of a cosine wave.
    This has the benefit of starting and stopping at true 0 without
    the truncation required from gaussian pulses.
    """

    x = np.arange(0, width)

    amplitude, phase = _calc_amp_phase(amplitude, phase, **kwds)
    unit_less = np.pi * np.array(1. / (x[-1]))
    return np.exp(1j * phase) * np.array(amplitude * (0.5 + 0.5 * np.cos(2. * x * unit_less)), dtype=np.complex64)

def gaussian(width, amplitude=1.0, phase=0.0, sigmas=3, **kwds):
    """Width is the exact width, not the width of the sigmas.
    """

    sigma = width / (2. * sigmas)

    amplitude, phase = _calc_amp_phase(amplitude, phase, **kwds)
    gaus = np.exp(-(np.arange(0, width) - width / 2.) ** 2. / (2 * sigma ** 2))
    return (amplitude * gaus * np.exp(1j * phase)).astype(np.complex64)


def DRAG(width, amplitude=1.0, phase=0.0, sigmas=3, alpha=0.5, delta=260e6, sample_rate=1.0e9, df=0,
         **kwds):
    """Standard DRAG pulse as defined in https://arxiv.org/pdf/0901.0534.pdf

    Args:
        width: the total width of the pulse in points
        alpha: alpha parameter of the DRAG pulse
        sigmas: number of standard deviations
        delta: the anharmonicity of the qubit
        amplitude: the amplitude of the pulse, clipped to between -1 and 1.
        phase: the phase applied as an overall multipler to the pulse,
               i.e. pulse_out = pulse * np.exp(1j*phase)
        sample_rate: AWG sample rate (default 2.5 GHz), and not used in this method.
        df: additional detuning from target frequency
    """

    delta = kwds.pop('sample_rate', delta)

    amplitude, phase = _calc_amp_phase(amplitude, phase, **kwds)

    sigma = width / (2. * sigmas)

    df = float(df / sample_rate)
    delta = float(delta * 2 * np.pi / sample_rate)

    x = np.arange(0, width)

    gaus = np.exp(-(x - width / 2.) ** 2. / (2 * sigma ** 2))

    dgaus = -(x - width / 2.) / (sigma ** 2) * gaus

    return (np.exp(x * df * 1j * 2 * np.pi) * amplitude * (gaus - 1j * alpha * dgaus / delta) * np.exp(
            1j * phase)).astype(np.complex64)


def cos_square(width, amplitude=1.0, phase=0.0, ramp_fraction=0.25, sample_rate=2.5e9, df=0, **kwds):
    """Return a square pulse shape of the specified amplitude with cosine edges.

    Args:
        width: the total width of the pulse in points
        amplitude: the amplitude of the pulse, clipped to between -1 and 1.  
        phase: the phase applied as an overall multipler to the pulse,
               i.e. pulse_out = pulse * np.exp(1j*phase)
        ramp_fraction: duration of the cosine ramp on either side of the pulse, \
        expressed as a fraction of the total pulse length
        sample_rate: AWG sample rate (default 2.5 GHz), and not used in this method.
        df: additional detuning from target frequency
    """

    amplitude, phase = _calc_amp_phase(amplitude, phase, **kwds)

    # 1. create the initial square pulse
    env = np.ones(width)

    # 2.  calculate the length, in points, of the cosine ramp, create the ramp shape, ...
    #   and replace the front/back of the pulse with the appropriate ramp
    ramp_fraction = np.clip(ramp_fraction, 0, 0.5)
    ramp_length = int(np.floor(ramp_fraction * width))

    # if there's actually a ramp, do the ramp
    if ramp_length > 0:
        ramp_up = -(np.cos(np.arange(0, ramp_length) * np.pi / ramp_length) - 1) / 2.
        env[:ramp_length] = ramp_up
        env[-ramp_length:] = ramp_up[::-1]

    # 3.  clip amplitude to -1.0 to +1.0...
    #   and scale the pulse
    amplitude = np.clip(amplitude, -1.0, 1.0)
    env = env * amplitude

    # 4.  multiply by the phase...
    env = env * np.exp(1j * phase)

    # 5.  multiply by a detuning df offset detuning...
    df = float(df / sample_rate)
    x = np.arange(0, width)
    env = env * np.exp(-x * df * 1j * 2 * np.pi)

    return env


def gauss_square(width, amplitude=1.0, phase=0.0, ramp_fraction=0.1, **kwds):
    """Return a square pulse shape of the specified amplitude with gaussian edges.

    Args:
        width: the total width of the pulse in points
        amplitude: the amplitude of the pulse, clipped to between -1 and 1.  
        phase: the phase applied as an overall multipler to the pulse, i.e. pulse_out = pulse * np.exp(1j*phase)
        ramp_fraction: duration of the cosine ramp on either side of the pulse,\
         expressed as a fraction of the total pulse length
    """

    amplitude, phase = _calc_amp_phase(amplitude, phase, **kwds)

    # 1. create the initial square pulse
    env = np.ones(width)

    # 2.  calculate the length, in points, of the cosine ramp, create the ramp shape, ...
    #   and replace the front/back of the pulse with the appropriate ramp
    ramp_fraction = np.clip(ramp_fraction, 0, 0.5)
    ramp_length = int(np.floor(ramp_fraction * width))

    # if there's actually a ramp, do the ramp
    if ramp_length > 0:
        ramp_up = -(np.cos(np.arange(0, ramp_length) * np.pi / (ramp_length)) - 1) / 2.
        env[:ramp_length] = ramp_up
        env[-ramp_length:] = ramp_up[::-1]

    # 3. scale the pulse
    env = env * amplitude

    # 4.  multiply by the phaase)
    return env * np.exp(1j * phase)


def hann(width, amplitude=1.0, phase=0.0, **kwds):
    """A Hann window function, see https://en.wikipedia.org/wiki/Window_function#Hann_window
    """

    amplitude, phase = _calc_amp_phase(amplitude, phase, **kwds)
    window = amplitude * np.hanning(width).astype(complex) * np.exp(1j * phase)
    return window


def arb(width, amplitude=1.0, points=np.zeros(0), phase=0.0, **kwds):
    """An arbitrary pulse definition, allows for arbitrary pulses
    to be added to the sequencer without the creation of new functions.
    """

    amplitude, phase = _calc_amp_phase(amplitude, phase, **kwds)
    return amplitude * np.exp(1j * phase) * np.array(points[:width], dtype=complex)


def mark(width, amplitude=1.0, phase=0.0, **kwds):
    return np.ones(width, dtype=np.int64)


envelope_lookup = {
    'square'            : square,
    'gaussian'          : gaussian,
    'sin'               : sin,
    'half_sin'          : half_sin,
    'cos_env'           : cos_env,
    'DRAG'              : DRAG,
    'cos_square'        : cos_square,
    'cos_square_fixed'  : cos_square,
    'hann'              : hann,
    'arb'               : arb,
    'mark'              : mark,
}


# generate time-value equivalents
def _generate_tv_envelopes():
    for name, func in envelope_lookup.items():
        def tv_func(twidth, dt, *args):
            t = np.arange(0, twidth, dt)
            width = len(t)
            return t, func(width, *args)

        globals()['tv_'+name] = tv_func
_generate_tv_envelopes()
del _generate_tv_envelopes


# the following functions did not exist in pulse_library (only in qubic) and rely on twidth
def tv_cos_edge_square(twidth, dt, ramp_fraction=0.25):
    t = np.arange(0, twidth, dt)
    f = 1.0/(2*ramp_fraction*twidth)
    tedge = np.arange(0,2*twidth*ramp_fraction, dt)
    width = len(t)
    if ramp_fraction > 0 and ramp_fraction <= 0.5:
        edges = (np.cos(2*np.pi*f*tedge-np.pi) + 1.0)/2.0
        nramp = len(edges)/2
        nflat = width-len(edges)
        env = np.concatenate((edges[:nramp], np.ones(nflat), edges[nramp:]))
    else:
        log.error('ramp_fraction should be 0 < ramp_function <= 0.5')
        env = np.ones(width)
    return t, env.astype(np.complex64)

def tv_sin_edge_square(twidth, dt, ramp_fraction=0.25):
    """Return a square pulse shape of the specified amplitude with cosine edges.

    Args:
        twidth: the total width of the pulse
        ramp_fraction: duration of the cosine ramp on either side of the pulse,
            expressed as a fraction of the total pulse length
    """

    t = np.arange(0, twidth, dt)
    width = len(t)
    if ramp_fraction > 0 and ramp_fraction <= 0.5:
        _,edges = tv_half_sin(twidth=2*ramp_fraction*twidth, dt=dt, phase0=0.0)
        nramp = int(len(edges)/2)
        nflat = width-len(edges)
        env = np.concatenate((edges[:nramp], np.ones(nflat), edges[nramp:]))
    else:
        log.error('ramp_fraction should be 0 < ramp_function <= 0.5')
        env = np.ones(width)

    return t, env.astype(np.complex64)

def tv_gauss_edge_square(twidth, dt, ramp_fraction=0.1, sigmas=3):
    """Return a square pulse shape of the specified amplitude with gaussian edges.

    Args:
        twidth: the total width of the pulse in points
        ramp_fraction: duration of the cosine ramp on either side of the pulse,
            expressed as a fraction of the total pulse length
    """

    t = np.arange(0,twidth,dt)
    width = len(t)

    if ramp_fraction > 0 and ramp_fraction <= 0.5:
        _, edges = tv_gaussian(twidth=2.0*ramp_fraction*twidth, dt=dt, sigmas=sigmas)
        nramp = len(edges)/2
        nflat = width-len(edges)
        env = np.concatenate((edges[:nramp],np.ones(nflat),edges[nramp:]))
    else:
        log.error('ramp_fraction should be 0 < ramp_function <= 0.5')
        env = np.ones(width)

    return t, env.astype(np.complex64)
