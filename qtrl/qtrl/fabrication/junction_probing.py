# Copyright (c) 2018-2019, UC Regents

import numpy as np
# Author: Brad Mitchell, bradmitchell@berkeley.edu, Dar Dahlen
# This is a direct port of the Mathematica 
# notebook that John Mark Kreikebaum & Kevin O'brien passed on to me.

# Add exact frequency estimates given Ec, Ej a la Kevin O'Brien's code

hplanck = 6.626e-34
electron_charge = 1.6e-19
al_bandgap = 0.170e-3 * electron_charge


def junction_resistance(voltage, r_bias=100e3, v_div=1 / 100., v_short=0.0):
    """Calculate the resistance of the junction
        Args:
            voltage - voltage measured from probing (Volts)

            r_bias - bias voltage (Ohms), defaults 100e3
            v_div - measured voltage when probes are not touching anything (Volts), defaults 1/100.
            v_short - measured voltage when touching probes together (Volts), defaults 0.0
    """
    if v_short == 0:
        return r_bias * (v_div / voltage - 1.) ** (-1.)
    else:
        return r_bias * (v_div / voltage - 1.) ** (-1.) - r_bias * (v_div / v_short - 1.) ** (-1.)


def critical_current(voltage, r_bias=100e3, v_div=1 / 100., v_short=0.0):
    """Calculate the critical current of the junction
        Args:
            voltage - volts measured from probing (Volts)

            r_bias - bias volts (Ohms), defaults 100e3
            v_div - measured volts when probes are not touching anything (Volts), defaults 1/100.
            v_short - measured volts when touching probes together (Volts), defaults 0.0
    """
    return al_bandgap * np.pi / (2 * electron_charge * junction_resistance(voltage, r_bias, v_div, v_short))


def josephson_inductance(voltage, r_bias=100e3, v_div=1 / 100., v_short=0.0):
    """Calculate the josephson inductance of the junction
        Args:
            voltage - volts measured from probing (Volts)

            r_bias - bias volts (Ohms), defaults 100e3
            v_div - measured volts when probes are not touching anything (Volts), defaults 1/100.
            v_short - measured volts when touching probes together (Volts), defaults 0.0
    """
    return hplanck / (4 * electron_charge * np.pi) / critical_current(voltage, r_bias, v_div, v_short)


def josephson_energy(voltage, r_bias=100e3, v_div=1 / 100., v_short=0.0):
    """Calculate the josephson energy of the junction
        Args:
            voltage - volts measured from probing (Volts)

            r_bias - bias volts (Ohms), defaults 100e3
            v_div - measured volts when probes are not touching anything (Volts), defaults 1/100.
            v_short - measured volts when touching probes together (Volts), defaults 0.0
    """
    return hplanck / (4 * electron_charge * np.pi) * critical_current(voltage, r_bias, v_div, v_short)


def frequency(voltage, e_c, r_bias=100e3, v_div=1 / 100., v_short=0.0):
    """Calculate the frequency of the junction
        Args:
            voltage - volts measured from probing (Volts)
            e_c - Charging energy of the transmon e**2/C_sum (Frequency)

            r_bias - bias volts (Ohms), defaults 100e3
            v_div - measured volts when probes are not touching anything (Volts), defaults 1/100.
            v_short - measured volts when touching probes together (Volts), defaults 0.0
    """
    ec_energy = hplanck * e_c
    return (np.sqrt(8 * ec_energy * josephson_energy(voltage, r_bias, v_div, v_short)) - ec_energy) / hplanck
