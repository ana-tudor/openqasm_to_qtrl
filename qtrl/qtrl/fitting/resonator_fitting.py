# Copyright (c) 2018-2019, UC Regents

"""Script for resonator data acqusition and characterization

Reference for if bandwidth and averaging:
http://anlage.umd.edu/Microwave%20Measurements%20for%20Personal%20Web%20Site/5980-2778EN.pdf


TODO: 
-calculate the confidence interval for qint
-add weighting to fit
-load fridge attenuation from file 
    what kind of file? csv, mat file? maybe mat file in standard format
-load fridge temp
-plot result for a given photon number

-auto peak find and auto-electrical delay. give a range to search over and a threshold Q. 

change smith data to complex
add option to specify how wide the span should be in terms of linewidths

-give option to plan a sequence of power steps based on the estimated photon number. will require the min and max vna powers. is there a command to get this?
-JM suggests 0.5 photons to 1e6 photons in 3dB steps. 
-

#### SETTINGS FOR AUTOMATICALLY SETTING POWER RANGE ######
powerStep = 3 #size of the power step in dB
minPhoton = 0.5 # minimum power in photons
maxPhoton = 1e6 # maximum power in photons
photons = np.power(10,1/20.0*np.linspace(20*np.log10(minPhoton),20*np.log10(maxPhoton),np.ceil(20*np.log10(maxPhoton/minPhoton)/powerStep))


FIX AGILENT VNA SCRIPT TO REMOVE %0.03E

"""
import lmfit as lm
import numpy as np


def powerToPhotons(power=None, qext=None, q=None, f0=None, fridgeAttenuation=None):
    """Calculate the power in the resonator. ref? what is that constant? derive"""
    h = 6.626e-34
    pwatts = 0.001 * np.power(10, (power - np.abs(fridgeAttenuation)) / 10.0)
    photons = pwatts / np.pi * np.square(q) / (qext * h * np.square(f0))
    return photons


def hanger_resonator_params(A, f_inf, f_zero, tau=0, covar=None):
    """Given generic resonator parameters,

        Arguements:
        A: Overall complex amplitude
        f_inf: point in the frequency doamin mapping to infinity in the
                complex scattering plane. (aka a pole)
        f_zero: point in the frequency doamin mapping to zero in the
                complex scattering plane. (aka a zero)
        tau: real electrical delay
        covar: optional covariance matrix

        Returns:
        Dictionary containing relevent resonator data. if covar is specified,
            the returned dictionary will also contain standard deviations."""
    f0 = np.real(f_inf)
    f_int = np.real(f_zero)
    kappa = 2 * np.imag(f_inf)
    kappa_int = 2 * np.imag(f_zero)
    kappa_ext_complex = 2 * (f_zero - f_inf) * 1j
    param_dict = dict()
    param_dict['a'] = np.real(A)
    param_dict['b'] = np.imag(A)
    param_dict['f0'] = f0
    param_dict['kappa'] = kappa
    param_dict['kappa_int'] = kappa_int
    param_dict['Q'] = f0 / kappa
    param_dict['Qi'] = f0 / kappa_int
    param_dict['Qe'] = f0 / abs(kappa_ext_complex)
    param_dict['phi'] = np.angle(kappa_ext_complex + np.pi) % (2 * np.pi) - np.pi
    param_dict['tau'] = tau
    if covar is None:
        return param_dict
    # compute the standard deviation for the various quantities
    transform_vectors = dict()
    transform_vectors['a'] = np.array([1, 0, 0, 0, 0, 0, 0]) # real(A)
    transform_vectors['b'] = np.array([0, 1, 0, 0, 0, 0, 0]) # imag(A)
    transform_vectors['f0'] = np.array([0, 0, 1, 0, 0, 0, 0]) # real(f_inf)
    transform_vectors['kappa'] = np.array([0, 0, 0, 2, 0, 0, 0]) # imag(f_inf)
    transform_vectors['kappa_int'] = np.array([0, 0, 0, 0, 0, 2, 0]) # imag(f_zero)
    transform_vectors['Q'] = np.array([0, 0, 1 / kappa, -2 * f0 / kappa ** 2, 0, 0, 0])
    transform_vectors['Qi'] = np.array([0, 0, 1 / kappa_int, 0, 0, -2 * f0 / kappa_int ** 2, 0])
    transform_vectors['Qe'] = np.array([0, 0,
                                        (kappa - kappa_int) ** 2 + 4 * f_int * (f_int - f0),
                                        2 * f0 * (kappa_int - kappa),
                                        4 * f0 * (f0 - f_int),
                                        2 * f0 * (kappa - kappa_int),
                                        0]) / abs(kappa_ext_complex) ** 3
    transform_vectors['phi'] = np.array([0, 0,
                                         2 * (kappa_int - kappa),
                                         4 * (f0 - f_int),
                                         2 * (kappa - kappa_int),
                                         4 * (f_int - f0),
                                         0]) / abs(kappa_ext_complex) ** 2

    transform_vectors['tau'] = np.array([0, 0, 0, 0, 0, 0, 1])

    # take v.covar.v to get the variance of the new variable for gradient v
    for key in transform_vectors:
        vec = transform_vectors[key]
        param_dict[key + '_sd'] = np.sqrt(np.dot(np.dot(covar, vec), vec))
    return param_dict


def reflection_resonator_params(A, f_inf, f_zero, tau=0, covar=None):
    """Given generic resonator parameters,

        Arguments:
        A: Overall complex amplitude
        f_inf: point in the frequency doamin mapping to infinity in the
                complex scattering plane. (aka a pole)
        f_zero: point in the frequency doamin mapping to zero in the
                complex scattering plane. (aka a zero)
        tau: real electrical delay
        covar: optional covariance matrix

        Returns:
        Dictionary containing relevant resonator data. if covar is specified,
            the returned dictionary will also contain standard deviations."""
    f0 = np.real(f_inf)
    f_int = np.real(f_zero)
    kappa = 2 * np.imag(f_inf)
    kappa_int = np.imag(f_inf) + np.imag(f_zero)
    kappa_ext_complex = np.imag(f_inf) - np.imag(f_zero)
    param_dict = dict()
    param_dict['a'] = np.real(A)
    param_dict['b'] = np.imag(A)
    param_dict['f0'] = f0
    param_dict['kappa'] = kappa
    param_dict['kappa_int'] = kappa_int
    param_dict['kappa_ext'] = np.abs(kappa_ext_complex)
    param_dict['Q'] = f0 / kappa
    param_dict['Qi'] = f0 / kappa_int
    param_dict['Qe'] = f0 / abs(kappa_ext_complex)
    param_dict['phi'] = np.angle(kappa_ext_complex + np.pi) % (2 * np.pi) - np.pi
    param_dict['tau'] = tau
    if covar is None:
        return param_dict
    # compute the standard deviation for the various quantities TBD
    # array basis is [A_r, A_i, f_inf_r, f_inf_i, f_zero_r, f_zero_i, tau]
    transform_vectors = dict()
    transform_vectors['a'] = np.array([1, 0, 0, 0, 0, 0, 0])
    transform_vectors['b'] = np.array([0, 1, 0, 0, 0, 0, 0])
    transform_vectors['f0'] = np.array([0, 0, 1, 0, 0, 0, 0])
    transform_vectors['kappa'] = np.array([0, 0, 0, 2, 0, 0, 0])
    transform_vectors['kappa_int'] = np.array([0, 0, 0, 1, 0, 1, 0])
    transform_vectors['kappa_ext'] = np.array([0, 0, 0, 1, 0, -1, 0])
    transform_vectors['Q'] = np.array([0,0,1/kappa, -2*f0/kappa**2,0,0,0])
    transform_vectors['Qi'] = np.array([0,0,1/kappa_int,0,0,-2*f0/kappa_int**2,0])
    transform_vectors['Qe'] = np.array([0,0,
                                (kappa-kappa_int)**2 + 4*f_int*(f_int-f0),
                                2*f0*(kappa_int-kappa),
                                4*f0*(f0-f_int),
                                2*f0*(kappa-kappa_int),
                                0])/abs(kappa_ext_complex)**3
    transform_vectors['phi'] = np.array([0,0,
                                        2*(kappa_int-kappa),
                                        4*(f0-f_int),
                                        2*(kappa-kappa_int),
                                        4*(f_int-f0),
                                        0])/abs(kappa_ext_complex)**2

    transform_vectors['tau'] = np.array([0,0,0,0,0,0,1])

    # take v.covar.v to get the variance of the new variable
    for key in transform_vectors:
        vec = transform_vectors[key]
        param_dict[key + '_sd'] = np.sqrt(np.dot(np.dot(covar, vec), vec))
    return param_dict


def _params_to_complex_vals(params):
    """internal use converting dictionary to complex values."""
    A_r = params['A_r'].value
    A_i = params['A_i'].value
    f_inf_r = params['f_inf_r'].value
    f_inf_i = params['f_inf_i'].value
    f_zero_r = params['f_zero_r'].value
    f_zero_i = params['f_zero_i'].value
    tau = params['tau'].value

    A = A_r + A_i * 1j  # scaling Amplitude
    f_inf = f_inf_r + f_inf_i * 1j  # pole in the complex scattering plane
    f_zero = f_zero_r + f_zero_i * 1j  # zero of the complex scattering plane
    return (A, f_inf, f_zero, tau)


def resonator_f_to_S(f, A, f_inf, f_zero, tau=0):
    """Converts frequency data to scattering data given generic resonator
        parameters.

        Arguements:
        f: frequency or frequencies to be converted
        A: Overall complex amplitude
        f_inf: point in the frequency domain mapping to infinity in the
                complex scattering plane. (aka a pole)
        f_zero: point in the frequency doamin mapping to zero in the
                complex scattering plane. (aka a zero)
        tau: real electrical delay"""
    return A * (f - f_zero) / (f - f_inf) * np.exp(1j * (f - np.real(f_inf)) * tau)


def residual(params, f, data):
    """params is the lmfit object. x is the frequency in Hz. For reference: https://lmfit.github.io/lmfit-py/fitting.html"""
    S21 = resonator_f_to_S(f, *_params_to_complex_vals(params))
    return (S21.view(np.float) - data.view(np.float))


def resonator_regression(frequency, smith):
    """Resonator regression for dimensionless units.
        Arguements:
            frequency: list of frequencies
            smith: list of complex scattering data

        Returns a tuple containing:
            0) list of generalized resonator parameters:
                0) A, overall complex amplitude
                1) f_inf, complex frequency where smith(f_inf) = infinity
                2) f_0, complex frequency where smith(f_0) = 0
                3) electrical delay
            1) covariance matrix with the following basis
                ['A_r', 'A_i','f_inf_r','f_inf_i','f_zero_r','f_zero_i', 'tau']
            2) reduced chi value"""

    # f_scaled is a rescaled frequency so matrix inversion doesn't get too singular during the initial guess

    df = frequency[-1] - frequency[0]
    f_scaled = (frequency - frequency[0]) / df
    smith = np.array(smith)

    # autoguess some starting values
    A = np.ones((len(frequency), 3), dtype=complex)
    A[:, 0] = f_scaled
    A[:, 2] = -smith
    y = np.multiply(f_scaled, smith)
    mobius_fit = np.linalg.lstsq(A, y)[0]
    A = mobius_fit[0]
    f_inf = -mobius_fit[2] * df + frequency[0]
    f_zero = -mobius_fit[1] / mobius_fit[0] * df + frequency[0]

    # create a set of Parameters
    params = lm.Parameters()
    params.add('A_r', value=np.real(A))
    params.add('A_i', value=np.imag(A))
    params.add('f_inf_r', value=np.real(f_inf))
    params.add('f_inf_i', value=np.imag(f_inf))
    params.add('f_zero_r', value=np.real(f_zero))
    params.add('f_zero_i', value=np.imag(f_zero))
    params.add('tau', value=0)

    # best Minimizer class documentation i've found so far
    # https://github.com/lmfit/lmfit-py/blob/master/doc/fitting.rst#id81
    mini = lm.Minimizer(residual, params, fcn_args=(frequency, smith.view(np.float)))
    minimized = mini.minimize()

    #return minimized
    return (_params_to_complex_vals(minimized.params), minimized.covar, minimized.redchi)


# useful strings for datastore server
resonator_description = '''A: Overall complex amplitude
        f_inf: point in the frequency doamin mapping to infinity in the
                complex scattering plane. (aka a pole)
        f_zero: point in the frequency doamin mapping to zero in the
                complex scattering plane. (aka a zero)
        tau: real electrical delay'''
resonator_variables = ['Amplitude', 'f_inf', 'f_zero', 'tau']
resonator_covar_variables = ['A_r', 'A_i', 'f_inf_r', 'f_inf_i', 'f_zero_r', 'f_zero_i', 'tau']
