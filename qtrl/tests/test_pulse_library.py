# Copyright (c) 2018-2019, UC Regents

import py, os, sys
from pytest import raises

import numpy as np

"""Unit tests for the qtrl/sequencer/pulse_library.py function"""


def freq_present(signal, sample_rate, target_freq, within=0.05):
    """Rough test to see if a signal where some percentage of the signal's power
    is contained within a specific frequency. This is a helper function for
    many of the tests below.

    Args:
        signal - one dimensional, ndarray
        sample_rate - rate at which the samples are taken, float
        target_freq - what is the desired frequency, float
        within - what percentile the power of the signal must be in before function will
                 return True, float [0-1], defaults 0.25

    Returns:
        boolean - based on whether the signal contains mostly the desired frequency
    """

    # remove the DC component
    signal -= np.mean(signal)

    fft = np.abs(np.fft.fft(signal)) ** 2

    # Sort the signals by strength, largest to smallest
    sorted_fft = np.argsort(fft)[::-1]

    # what index are we looking for
    index = int(len(signal)/sample_rate*target_freq/2)
    allowed_index = [index - 1, index, index + 1]

    # what index is the last to contain our within variable
    percentile_index = int(np.ceil(len(signal) * within))

    # return true if any frequency is within 1 step of the target one
    return any([x in sorted_fft[:percentile_index] for x in allowed_index])


class TestPULSE_LIBRARY:

    def standard_test(self, pulse_func, phase_offset=0.0, **kwargs):
        """Standard set of tests for a generic pulse function.
        optional kwargs are passed to the function as well.

        Pulse functions are assumed to take 4 standard inputs:
            -width
            -amplitude
            -phase
            -offset_phase

        Args:
            pulse_func - the function which constructs the pulse envelope
            phase_offset - expected offset in complex phase of the pulse.
                           Some pulses you expect to have a finite offset
            kwargs - dictionary of values which will be passed to the pulse_func
                     Excluding the 4 standard inputs
        """
        # Test simple construction
        s = pulse_func(width=100,
                       amplitude=1.0,
                       phase=0.0,
                       global_phase=0.0,
                       **kwargs)
        assert len(s) == 100
        assert np.max(np.abs(s)) >= 0.95
        assert np.isclose(np.angle(s[1]), 0.0 + phase_offset)

        # Test phase
        s = pulse_func(width=1000,
                       amplitude=0.5,
                       phase=0.25,
                       global_phase=0.0,
                       **kwargs)
        assert len(s) == 1000
        assert np.max(np.abs(s)) >= 0.45
        assert np.isclose(np.angle(s[1]), 0.25 + phase_offset)

        # Test Phase with offset
        s = pulse_func(width=25,
                       amplitude=0.5,
                       phase=0.25,
                       global_phase=0.50,
                       **kwargs)
        assert len(s) == 25
        assert np.max(np.abs(s)) >= 0.45
        assert np.isclose(np.angle(s[1]), 0.75 + phase_offset)

    def frequency_tests(self, pulse_func, phase_offset=0.0):
        """Standard tests to see if pulses which generate their
        own modulation have the correct frequency content.
        Has the same assumptions as standard_tests, with
        the additional assumptions of accepting keywords:
            - frequency
            - sample_rate

        Args:
            pulse_func - the function which constructs the pulse envelope
            phase_offset - expected offset in complex phase of the pulse.
                           Some pulses you expect to have a finite offset
        """

        # Test the frequency is correct
        s = pulse_func(width=100,
                       amplitude=1.0,
                       phase=0.0,
                       global_phase=0.0,
                       frequency=1e6,
                       sample_rate=1e9)

        # Test Frequency and sample_rate interaction
        assert freq_present(signal=s,
                            sample_rate=1e9,
                            target_freq=1e6)

        s = pulse_func(width=7111,
                       amplitude=0.5,
                       frequency=7e6,
                       phase=0.25,
                       global_phase=0.50,
                       sample_rate=3e9)
        assert len(s) == 7111
        assert np.isclose(np.angle(s[0]), 0.75 + phase_offset, 0.03), np.angle(s[0])

        # Test Frequency and sample_rate interaction
        assert freq_present(signal=s,
                            sample_rate=3e9,
                            target_freq=7e6)

    def test01_square(self):
        from qtrl.sequencer import square
        self.standard_test(square)

    def test02_sin(self):
        from qtrl.sequencer import sin
        self.standard_test(sin, phase_offset=-np.pi/2.)
        self.frequency_tests(sin, phase_offset=-np.pi/2.)

    def test03_cos_env(self):
        from qtrl.sequencer import cos_env
        self.standard_test(cos_env)

    def test04_gaussian(self):
        from qtrl.sequencer import gaussian
        self.standard_test(gaussian)

    def test05_DRAG(self):
        from qtrl.sequencer import DRAG
        self.standard_test(DRAG, alpha=0.0)
        self.frequency_tests(DRAG)

    def test06_hann(self):
        from qtrl.sequencer import hann
        self.standard_test(hann)

    def test07_gauss_square(self):
        from qtrl.sequencer import gauss_square
        self.standard_test(gauss_square)

    def test08_cos_square(self):
        from qtrl.sequencer import cos_square
        self.standard_test(cos_square)

