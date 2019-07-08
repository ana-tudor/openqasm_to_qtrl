# Author Ravi Naik: rnaik24@gmail.com
import numpy as np
import pandas as pd
from .transmon_numerics import transmon_solver, two_body_JC_solver, qubit_res_dressed_frequencies


def asym_squid(phi, E_J_sum, asymm, phi_0=1, phi_off=0):
    """
    Given an E_J maximum (E_J_sum=E_J1+E_J2) and junction asymmetry (asymm=E_J1/E_J2) for an asymmetric SQuID, returns
    E_J(phi) for a given flux phi. phi_0 and phi_offset are the flux quantum and flux offset in appropriate units, respectively
    """

    d = (asymm-1.0)/(1.0+asymm)
    return E_J_sum*abs(np.cos(np.pi*(phi-phi_off)/phi_0))*np.sqrt(1.0+d**2*(np.tan(np.pi*(phi-phi_off)/phi_0))**2)


def asym_transmon_full(phi, E_J_sum, E_C, asymm, phi_0=1, phi_off=0):
    """
    Solves the Hamiltonian for a transmon with an asymmetric SQuID.
    """
    return transmon_solver(asym_squid(phi, E_J_sum, asymm, phi_0, phi_off), E_C)


def dressed_frequencies_vs_flux(flux, qubit, chip):
    """
    Takes a flux value (units of flux quanta) and the chip_parameters config file and returns the dressed state properties.
    """

    chip.load()
    EJ_sum = chip[f'Q{qubit}']['EJ_Sum']
    EC = chip[f'Q{qubit}']['EC']
    asymm = chip[f'Q{qubit}']['asymm']
    f_r_bare = chip[f'Q{qubit}']['f_r_bare']
    g = chip[f'Q{qubit}']['g']
    transmon_spec = asym_transmon_full(flux, EJ_sum, EC, asymm)['absolute_freqs']
    resonator_spec = np.arange(5)*f_r_bare
    solver_props = two_body_JC_solver(transmon_spec,resonator_spec,g)
    dressed_props = qubit_res_dressed_frequencies(solver_props)
    return dressed_props


def interpolate_flux_from_freq(flux_array, freq_array, target_freq):
    """
    Interpolate a flux value for a target frequency by performing a
    linear fit between the two closest frequency values produced by
    the model
    """
    df = pd.DataFrame({'Flux':flux_array,'Freq':freq_array})
    below = df[df.Freq <= target_freq].head(1)
    above = df[df.Freq >= target_freq].tail(1)
    linear_fit = np.polyfit(np.concatenate((below.Freq.values,above.Freq.values)),
                            np.concatenate((below.Flux.values,above.Flux.values)),
                            1)
    linear_fit_fn = np.poly1d(linear_fit)
    return linear_fit_fn(target_freq)


def qubit_max(qubit, chip):
    return dressed_frequencies_vs_flux(0.0, qubit, chip=chip)['dressed_qubit_frequency']


def qubit_min(qubit, chip):
    return dressed_frequencies_vs_flux(0.5, qubit, chip=chip)['dressed_qubit_frequency']


def resonator_max(qubit, chip):
    return dressed_frequencies_vs_flux(0.0, qubit, chip=chip)['dressed_resonator_frequency']


def resonator_min(qubit, chip):
    return dressed_frequencies_vs_flux(0.5, qubit, chip=chip)['dressed_resonator_frequency']