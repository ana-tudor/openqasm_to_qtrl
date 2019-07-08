# Copyright (c) 2018-2019, UC Regents

import numpy as np
from . import fit


#  common fitfunctions

def line(b,m):
    fitfunc_str = 'm * x + b'

    b = fit.Parameter(b, 'b')
    m = fit.Parameter(m, 'm')

    p0 = [b, m]  # Note: If you do not want to use a fit argument set fixed when using in fit1d

    def fitfunc(x):
        return b() + m() * x

    return p0, fitfunc, fitfunc_str

def parabola(a,k,x0):
    fitfunc_str = 'a + k * ( x - x0 )**2'

    a = fit.Parameter(a, 'a')
    k = fit.Parameter(k, 'k')
    x0 = fit.Parameter(x0,'x0')
    p0 = [a, k, x0]  # Note: If you do not want to use a fit argument set fixed when using in fit1d

    def fitfunc(x):
        return a() + k() * (x - x0())**2.0

    return p0, fitfunc, fitfunc_str

def fit_cos(g_f, g_a, g_A, g_x0, *arg):
    fitfunc_str = 'A * cos(2pi * (f*x + phi/360) ) + a'

    f = fit.Parameter(g_f, 'f')
    a = fit.Parameter(g_a, 'a')
    A = fit.Parameter(g_A, 'A')
    x0 = fit.Parameter(g_x0, 'x0')

    p0 = [f, a, A, x0]  # Note: If you do not want to use a fit argument set fixed when using in fit1d

    def fitfunc(x):
        return a() + A() * np.cos(2 * np.pi * f()*(x - x0()) )

    return p0, fitfunc, fitfunc_str


def fit_sum_2cos(g_avg, g_A, g_f_a, g_phi_a, g_B, g_f_b, g_phi_b, *arg):
    fitfunc_str = '(A*cos(2pi * (fa*x + phi_a/360) )+ B*cos(2pi * (f_b*x + phi_b/360)))/2 + avg'

    avg = fit.Parameter(g_avg, 'avg')

    A = fit.Parameter(g_A, 'A')
    f_a = fit.Parameter(g_f_a, 'f_a')
    phi_a = fit.Parameter(g_phi_a, 'phi_a')

    B = fit.Parameter(g_B, 'B')
    f_b = fit.Parameter(g_f_b, 'f_b')
    phi_b = fit.Parameter(g_phi_b, 'phi_b')

    p0 = [avg, A, f_a, phi_a, B, f_b,
          phi_b]  # Note: If you do not want to use a fit argument set fixed when using in fit1d

    def fitfunc(x):
        return avg() + (A() * np.cos(2 * np.pi * (f_a() * x + phi_a() / 360.)) + B() * np.cos(
            2 * np.pi * (f_b() * x + phi_b() / 360.))) / 2.0

    return p0, fitfunc, fitfunc_str


def fit_decaying_cos(g_f, g_a, g_A, g_phi, g_t, *arg):
    fitfunc_str = 'A *exp(-x/t) cos(2pi * (f*x + phi/360) ) + a'

    f = fit.Parameter(g_f, 'f')
    a = fit.Parameter(g_a, 'a')
    A = fit.Parameter(g_A, 'A')
    phi = fit.Parameter(g_phi, 'phi')
    t = fit.Parameter(g_t, 't')
    # print 'guessed frequency is '+str(g_f)
    p0 = [f, a, A, phi, t]

    def fitfunc(x):
        return a() + A() * np.exp(-x / t()) * np.cos(2 * np.pi * (f() * x + phi() / 360.))

    return p0, fitfunc, fitfunc_str


def fit_gaussian_decaying_cos(g_f, g_a, g_A, g_phi, g_t, *arg):
    fitfunc_str = 'A *exp(-(x/t)**2) cos(2pi * (f*x + phi/360) ) + a'

    f = fit.Parameter(g_f, 'f')
    a = fit.Parameter(g_a, 'a')
    A = fit.Parameter(g_A, 'A')
    phi = fit.Parameter(g_phi, 'phi')
    t = fit.Parameter(g_t, 't')
    print('guessed frequency is ' + str(g_f))
    p0 = [f, a, A, phi, t]

    def fitfunc(x):
        return a() + A() * np.exp(-(x / t()) ** 2) * np.cos(2 * np.pi * (f() * x + phi() / 360.))

    return p0, fitfunc, fitfunc_str


def fit_gaussian_decaying_2cos(g_avg, g_C, g_t, g_A, g_f_a, g_phi_a, g_B, g_f_b, g_phi_b, *arg):
    """ Fits Gaussian decaying sum of two cosines.
    To be used for e.g. dynamicaldecoupling.Carbon_Ramsey_noDD"""

    fitfunc_str = 'avg + C * exp(-(x/t)**2) [ A*cos(2pi * (fa*x + phi_a/360) )+ B*cos(2pi * (f_b*x + phi_b/360)))/2 ]'

    avg = fit.Parameter(g_avg, 'avg')

    C = fit.Parameter(g_C, 'C')
    t = fit.Parameter(g_t, 't')

    A = fit.Parameter(g_A, 'A')
    f_a = fit.Parameter(g_f_a, 'f_a')
    phi_a = fit.Parameter(g_phi_a, 'phi_a')

    B = fit.Parameter(g_B, 'B')
    f_b = fit.Parameter(g_f_b, 'f_b')
    phi_b = fit.Parameter(g_phi_b, 'phi_b')

    p0 = [avg, C, t, A, f_a, phi_a, B, f_b,
          phi_b]  # Note: If you do not want to use a fit argument set fixed when using in fit1d

    def fitfunc(x):
        return avg() + C() * np.exp(-(x / t()) ** 2) * (
                A() * np.cos(2 * np.pi * (f_a() * x + phi_a() / 360.)) + B() * np.cos(
            2 * np.pi * (f_b() * x + phi_b() / 360.))) / 2.0

    return p0, fitfunc, fitfunc_str


def fit_double_decaying_cos(g_f1, g_A1, g_phi1, g_t1, g_f2, g_A2, g_phi2, g_t2, g_o, *arg):
    """ quite a specific function, for electron nuclear control, maybe place somewhere else """
    fitfunc_str = '''(A1 *exp(-x/t1) cos(2pi * (f1*x + phi1/360) ) + a1)*
                     (A2 *exp(-x/t2) cos(2pi * (f2*x + phi2/360) ) + a2)/2+ o '''

    f1 = fit.Parameter(g_f1, 'f1')
    # a1 = fit.Parameter(g_a1, 'a1')
    A1 = fit.Parameter(g_A1, 'A1')
    phi1 = fit.Parameter(g_phi1, 'phi1')
    t1 = fit.Parameter(g_t1, 't1')

    f2 = fit.Parameter(g_f2, 'f2')
    # a2 = fit.Parameter(g_a2, 'a2')
    A2 = fit.Parameter(g_A2, 'A2')
    phi2 = fit.Parameter(g_phi2, 'phi2')
    t2 = fit.Parameter(g_t2, 't2')
    o = fit.Parameter(g_o, 'o')

    # p0 = [f1, a1, A1, phi1, t1, f2, a2, A2, phi2, t2]
    p0 = [f1, A1, phi1, t1, f2, A2, phi2, t2, o]

    def fitfunc(x):
        return (1 - A1() + A1() * np.exp(-x / t1()) * np.cos(2 * np.pi * (f1() * x + phi1() / 360.))) * (
                1 - A2() + A2() * np.exp(-x / t2()) * np.cos(2 * np.pi * (f2() * x + phi2() / 360.))) / 2 + o()

    return p0, fitfunc, fitfunc_str


def fit_exp_decay_with_offset(g_a, g_A, g_tau, *arg):
    """
    fitfunction for an exponential decay,
        y(x) = A * exp(-x/tau) + a

    Initial guesses (in this order):
        g_a : offset
        g_A : initial Amplitude
        g_tau : decay constant

    """
    fitfunc_str = 'A * exp(-x/tau) + a'

    a = fit.Parameter(g_a, 'a')
    A = fit.Parameter(g_A, 'A')
    tau = fit.Parameter(g_tau, 'tau')
    p0 = [a, A, tau]

    def fitfunc(x):
        return a() + A() * np.exp(-x / tau())

    return p0, fitfunc, fitfunc_str


def fit_exp_decay_with_offset_linslope(g_a, g_A, g_tau, g_b, *arg):
    """
    fitfunction for an exponential decay,
        y(x) = A * exp(-x/tau) + a

    Initial guesses (in this order):
        g_a : offset
        g_A : initial Amplitude
        g_tau : decay constant

    """
    fitfunc_str = 'A * exp(-x/tau) + a + b*x'

    a = fit.Parameter(g_a, 'a')
    A = fit.Parameter(g_A, 'A')
    tau = fit.Parameter(g_tau, 'tau')
    b = fit.Parameter(g_b, 'b')
    p0 = [a, A, tau, b]

    def fitfunc(x):
        return a() + A() * np.exp(-x / tau()) + x * b()

    return p0, fitfunc, fitfunc_str


def fit_double_exp_decay_with_offset(g_a, g_A, g_tau, g_A2, g_tau2, *arg):
    """
    fitfunction for an exponential decay,
        y(x) = A * exp(-x/tau)+ A2 * exp(-x/tau2) + a

    Initial guesses (in this order):
        g_a : offset
        g_A : initial Amplitude
        g_tau : decay constant
        g_A2 : initial Amplitude 2
        g_tau2 : decay constant 2
    """
    fitfunc_str = 'A * exp(-x/tau)+ A2 * exp(-x/tau2) + a'

    a = fit.Parameter(g_a, 'a')
    A = fit.Parameter(g_A, 'A')
    tau = fit.Parameter(g_tau, 'tau')
    A2 = fit.Parameter(g_A2, 'A2')
    tau2 = fit.Parameter(g_tau2, 'tau2')
    p0 = [a, A, tau, A2, tau2]

    def fitfunc(x):
        return a() + A() * np.exp(-x / tau()) + A2() * np.exp(-x / tau2())

    return p0, fitfunc, fitfunc_str


def fit_exp_decay_shifted_with_offset(g_a, g_A, g_tau, g_x0, *arg):
    """
    fitfunction for an exponential decay,
        y(x) = A * exp(-(x-x0)/tau) + a

    Initial guesses (in this order):
        g_a : offset
        g_A : initial Amplitude
        g_tau : decay constant
        g_x0 : x offset
    """

    fitfunc_str = 'A * exp(-(x-x0)/tau) + a'

    a = fit.Parameter(g_a, 'a')
    A = fit.Parameter(g_A, 'A')
    tau = fit.Parameter(g_tau, 'tau')
    x0 = fit.Parameter(g_x0, 'x0')
    p0 = [a, A, tau, x0]

    def fitfunc(x):
        return a() + A() * np.exp(-(x - x0()) / tau())

    return p0, fitfunc, fitfunc_str


def fit_saturation(g_A, g_xsat, *arg):
    """
    fitfunction for a saturation (e.g., the NV PL)
        y(x) = A * x / (x + xsat)

    I.g.:
        g_A : maximum signal (at x=infinity)
        g_xsat : saturation point
    """

    fitfunc_str = 'A * x / (x + x_sat)'

    A = fit.Parameter(g_A, 'A')
    xsat = fit.Parameter(g_xsat, 'xsat')
    p0 = [A, xsat]

    def fitfunc(x):
        return A() * x / (x + xsat())

    return p0, fitfunc, fitfunc_str


def fit_saturation_with_offset_linslope(g_a, g_b, g_A, g_xsat, *arg):
    """
    fitfunction for a saturation (e.g., the NV PL)
        y(x) = a + b*x + A * x / (x + xsat)

    I.g.:
        g_a : offset
        g_b : linear slope
        g_A : maximum signal (at x=infinity)
        g_xsat : saturation point
    """

    fitfunc_str = 'a + b*x + A * x / (x + x_sat)'

    a = fit.Parameter(g_a, 'a')
    b = fit.Parameter(g_b, 'b')
    A = fit.Parameter(g_A, 'A')
    xsat = fit.Parameter(g_xsat, 'xsat')
    p0 = [a, b, A, xsat]

    def fitfunc(x):
        return a() + b() * x + A() * x / (x + xsat())

    return p0, fitfunc, fitfunc_str


def fit_poly(*arg):
    fitfunc_str = 'sum_n ( a[n] * x**n )'

    idx = 0
    p0 = []
    for i, a in enumerate(arg):
        p0.append(fit.Parameter(a, 'a%d' % i))
        idx = i

    def fitfunc(x):
        val = 0
        for i in range(idx):
            val += p0[i]() * x ** i
        return val

    return p0, fitfunc, fitfunc_str


def fit_poly_shifted(g_x0, *arg):
    fitfunc_str = 'sum_n ( a[n] * (x-x0)**n )'

    idx = 0
    p0 = []
    x0 = fit.Parameter(g_x0, 'x0')
    p0.append(x0)
    for i, a in enumerate(arg):
        p0.append(fit.Parameter(a, 'a%d' % i))
        idx = i

    def fitfunc(x):
        val = 0
        for i in range(idx):
            val += p0[i + 1]() * (x - x0()) ** i
        return val

    return p0, fitfunc, fitfunc_str


def fit_parabole(g_o, g_A, g_c, *arg):
    fitfunc_str = 'o + A * (x-c)**2'

    o = fit.Parameter(g_o, 'o')
    A = fit.Parameter(g_A, 'A')
    c = fit.Parameter(g_c, 'c')
    p0 = [o, A, c]

    def fitfunc(x):
        return o() + A() * (x - c()) ** 2

    return p0, fitfunc, fitfunc_str


def fit_AOM_powerdependence(g_a, g_xc, g_k, *arg):
    fitfunc_str = 'a * exp(-exp(-k*(x-xc)))'

    a = fit.Parameter(g_a, 'a')
    xc = fit.Parameter(g_xc, 'xc')
    k = fit.Parameter(g_k, 'k')

    p0 = [a, xc, k]

    def fitfunc(x):
        return a() * np.exp(-np.exp(-k() * (x - xc())))

    return p0, fitfunc, fitfunc_str


def fit_gauss(g_a, g_A, g_x0, g_sigma):
    ### i think there should be a factor 2 infront of the sigma
    fitfunc_str = 'a + A * exp(-(x-x0)**2/(2*sigma**2))'

    a = fit.Parameter(g_a, 'a')
    x0 = fit.Parameter(g_x0, 'x0')
    A = fit.Parameter(g_A, 'A')
    sigma = fit.Parameter(g_sigma, 'sigma')

    p0 = [a, x0, A, sigma]

    def fitfunc(x):
        return a() + A() * np.exp(-(x - x0()) ** 2 / (2 * sigma() ** 2))

    return p0, fitfunc, fitfunc_str


def fit_gauss_pos(g_a, g_A, g_x0, g_sigma):
    ### i think there should be a factor 2 infront of the sigma
    fitfunc_str = 'a + |A| * exp(-(x-x0)**2/(2*sigma**2))'

    a = fit.Parameter(g_a, 'a')
    x0 = fit.Parameter(g_x0, 'x0')
    A = fit.Parameter(g_A, 'A')
    sigma = fit.Parameter(g_sigma, 'sigma')

    p0 = [a, x0, A, sigma]

    def fitfunc(x):
        return a() + np.abs(A()) * np.exp(-(x - x0()) ** 2 / (2 * sigma() ** 2))

    return p0, fitfunc, fitfunc_str


def fit_general_exponential(g_a, g_A, g_x0, g_T, g_n):
    fitfunc_str = 'a + A * exp(-((x-x0)/T )**n)'

    a = fit.Parameter(g_a, 'a')
    A = fit.Parameter(g_A, 'A')
    x0 = fit.Parameter(g_x0, 'x0')
    T = fit.Parameter(g_T, 'T')
    n = fit.Parameter(g_n, 'n')

    p0 = [a, A, x0, T, n]

    def fitfunc(x):
        return a() + A() * np.exp(np.round(-(x - x0()) ** n() / (T() ** n()), 8))

    return p0, fitfunc, fitfunc_str


def fit_general_exponential_fixed_offset(f_a, g_A, g_x0, g_T, g_n):
    fitfunc_str = 'a + A * exp(-((x-x0)/T )**n)'

    A = fit.Parameter(g_A, 'A')
    x0 = fit.Parameter(g_x0, 'x0')
    T = fit.Parameter(g_T, 'T')
    n = fit.Parameter(g_n, 'n')

    p0 = [A, x0, T, n]

    def fitfunc(x):
        return f_a + A() * np.exp(-(x - x0()) ** n() / (T() ** n()))

    return p0, fitfunc, fitfunc_str


def fit_2gauss(g_a1, g_A1, g_x01, g_sigma1, g_A2, g_x02, g_sigma2):
    ### i think there should be a factor 2 infront of the sigma
    fitfunc_str = 'a1 + A1 * exp(-(x-x01)**2/(2*sigma1**2)) +\
            A2 * exp(-(x-x02)**2/(2*sigma2**2))'

    a1 = fit.Parameter(g_a1, 'a1')
    x01 = fit.Parameter(g_x01, 'x01')
    A1 = fit.Parameter(g_A1, 'A1')
    sigma1 = fit.Parameter(g_sigma1, 'sigma1')

    x02 = fit.Parameter(g_x02, 'x02')
    A2 = fit.Parameter(g_A2, 'A2')
    sigma2 = fit.Parameter(g_sigma2, 'sigma2')

    p0 = [a1, x01, A1, sigma1, x02, A2, sigma2]

    def fitfunc(x):
        return a1() + A1() * np.exp(-(x - x01()) ** 2 / (2 * sigma1() ** 2)) + \
               A2() * np.exp(-(x - x02()) ** 2 / (2 * sigma2() ** 2))

    return p0, fitfunc, fitfunc_str


def fit_lorentz(g_a, g_A, g_x0, g_gamma):
    fitfunc_str = 'a + 2*A/np.pi*gamma/(4*(x-x0)**2+gamma**2)'

    a = fit.Parameter(g_a, 'a')
    A = fit.Parameter(g_A, 'A')
    x0 = fit.Parameter(g_x0, 'x0')
    gamma = fit.Parameter(g_gamma, 'gamma')

    p0 = [a, A, x0, gamma]

    def fitfunc(x):
        return a() + 2 * A() / np.pi * gamma() / (4 * (x - x0()) ** 2 + gamma() ** 2)

    return p0, fitfunc, fitfunc_str


def fit_2lorentz(g_a1, g_A1, g_x01, g_gamma1, g_A2, g_x02, g_gamma2):
    fitfunc_str = 'a1 + 2*A1/np.pi*gamma1/(4*(x-x01)**2+gamma1**2) \
            + 2*A2/np.pi*gamma2/(4*(x-x02)**2+gamma2**2)'

    a1 = fit.Parameter(g_a1, 'a1')
    A1 = fit.Parameter(g_A1, 'A1')
    x01 = fit.Parameter(g_x01, 'x01')
    gamma1 = fit.Parameter(g_gamma1, 'gamma1')

    A2 = fit.Parameter(g_A2, 'A2')
    x02 = fit.Parameter(g_x02, 'x02')
    gamma2 = fit.Parameter(g_gamma2, 'gamma2')

    p0 = [a1, A1, x01, gamma1, A2, x02, gamma2]

    def fitfunc(x):
        return a1() + 2 * A1() / np.pi * gamma1() / (4 * (x - x01()) ** 2 + gamma1() ** 2) + \
               2 * A2() / np.pi * gamma2() / (4 * (x - x02()) ** 2 + gamma2() ** 2)

    return p0, fitfunc, fitfunc_str


def fit_3lorentz(g_a1, g_A1, g_x01, g_gamma1, g_A2, g_x02, g_gamma2, g_A3, g_x03,
                 g_gamma3):  # fit for 3 lorentzians for EOM drive
    fitfunc_str = 'a1 + 2*A1/np.pi*gamma1/(4*(x-x01)**2+gamma1**2) \
            + 2*A2/np.pi*gamma2/(4*(x-x02)**2+gamma2**2) + 2*A3/np.pi*gamma3/(4*(x-x03)**2+gamma3**2)'

    a1 = fit.Parameter(g_a1, 'a1')
    A1 = fit.Parameter(g_A1, 'A1')
    x01 = fit.Parameter(g_x01, 'x01')
    gamma1 = fit.Parameter(g_gamma1, 'gamma1')

    A2 = fit.Parameter(g_A2, 'A2')
    x02 = fit.Parameter(g_x02, 'x02')
    gamma2 = fit.Parameter(g_gamma2, 'gamma2')

    A3 = fit.Parameter(g_A3, 'A3')
    x03 = fit.Parameter(g_x03, 'x03')
    gamma3 = fit.Parameter(g_gamma3, 'gamma3')

    p0 = [a1, A1, x01, gamma1, A2, x02, gamma2, A3, x03, gamma3]

    def fitfunc(x):
        return a1() + 2 * A1() / np.pi * gamma1() / (4 * (x - x01()) ** 2 + gamma1() ** 2) + \
               2 * A2() / np.pi * gamma2() / (4 * (x - x02()) ** 2 + gamma2() ** 2) + 2 * A3() / np.pi * gamma3() / (
                       4 * (x - x03()) ** 2 + gamma3() ** 2)

    return p0, fitfunc, fitfunc_str


def fit_2lorentz_splitting(g_a1, g_A1, g_x01, g_gamma1, g_A2, g_dx2, g_gamma2):
    fitfunc_str = 'a1 + 2*A1/np.pi*gamma1/(4*(x-x01)**2+gamma1**2) \
            + 2*A2/np.pi*gamma2/(4*(x-x01-dx2)**2+gamma2**2)'

    a1 = fit.Parameter(g_a1, 'a1')
    A1 = fit.Parameter(g_A1, 'A1')
    x01 = fit.Parameter(g_x01, 'x01')
    gamma1 = fit.Parameter(g_gamma1, 'gamma1')

    dx2 = fit.Parameter(g_dx2, 'dx2')
    A2 = fit.Parameter(g_A2, 'A2')
    gamma2 = fit.Parameter(g_gamma2, 'gamma2')

    p0 = [a1, A1, x01, gamma1, A2, dx2, gamma2]

    def fitfunc(x):
        return a1() + 2 * A1() / np.pi * gamma1() / (4 * (x - x01()) ** 2 + gamma1() ** 2) + \
               2 * A2() / np.pi * gamma2() / (4 * (x - (x01() + dx2())) ** 2 + gamma2() ** 2)

    return p0, fitfunc, fitfunc_str


def fit_line(g_a, g_b, *arg):
    """
    fitfunction for a line
        y(x) = a + b*x

    I.g.:
        g_a : offset
        g_b : linear slope
    """

    fitfunc_str = 'a + b*x'

    a = fit.Parameter(g_a, 'a')
    b = fit.Parameter(g_b, 'b')
    # xsat = fit.Parameter(g_xsat, 'xsat')
    p0 = [a, b]

    def fitfunc(x):
        return a() + b() * x

    return p0, fitfunc, fitfunc_str


def fit_general_exponential_dec_cos(g_a, g_A, g_x0, g_T, g_n, g_f, g_phi):
    # NOTE: Order of arguments has changed to remain consistent with fitting params order
    # NOTE: removed g_x0=0 as a default argument. This should be handed explicitly to the function to prevent confusion
    # Fits with a general exponential modulated by a cosine
    fitfunc_str = 'a + A * exp(-((x-x0)/T )**n*cos(2pi *(f*x+phi/360) )'

    a = fit.Parameter(g_a, 'a')
    A = fit.Parameter(g_A, 'A')
    x0 = fit.Parameter(g_x0, 'x0')
    T = fit.Parameter(g_T, 'T')
    n = fit.Parameter(g_n, 'n')
    f = fit.Parameter(g_f, 'f')
    phi = fit.Parameter(g_phi, 'phi')

    p0 = [a, A, x0, T, n, f, phi]

    def fitfunc(x):
        return a() + A() * np.exp(-((x - x0()) / T()) ** n()) * np.cos(2 * np.pi * (f() * x + phi() / 360.))

    return p0, fitfunc, fitfunc_str


def fit_exp_cos(g_a, g_A, g_x0, g_T, g_n, g_f, g_phi):
    fitfunc_str = 'a + A * exp(-((x-x0)/T )**n  *  cos(2pi *(f*x+phi/360) )'

    a = fit.Parameter(g_a, 'a')
    A = fit.Parameter(g_A, 'A')
    x0 = fit.Parameter(g_x0, 'x0')
    T = fit.Parameter(g_T, 'T')
    n = fit.Parameter(g_n, 'n')
    f = fit.Parameter(g_f, 'f')
    phi = fit.Parameter(g_phi, 'phi')

    if g_n == 0:
        p0 = [a, A, x0, f, phi]

        def fitfunc(x):
            return a() + A() * np.cos(2 * np.pi * (f() * x + phi() / 360.))
    else:
        p0 = [a, A, x0, T, n, f, phi]

        def fitfunc(x):
            return a() + A() * np.exp(-((x - x0()) / T()) ** n()) * np.cos(2 * np.pi * (f() * x + phi() / 360.))

    return p0, fitfunc, fitfunc_str


def fit_hyperbola(g_a, g_n, g_o):
    fitfunc_str = 'a/(x**n)=o'

    a = fit.Parameter(g_a, 'a')
    n = fit.Parameter(g_n, 'n')
    o = fit.Parameter(g_o, 'o')
    p0 = [a, n, o]

    def fitfunc(x):
        return a() / (x ** n()) + o()

    return p0, fitfunc, fitfunc_str


def fit_dephasing_coupl(g_a, g_tau, g_coup_offs):
    fitfunc_str = '-a/(ln2-ln(1+e**(-0.5* (2*pi*(x+coup_off)*tau)**2)))'

    a = fit.Parameter(g_a, 'a')
    tau = fit.Parameter(g_tau, 'tau')
    coup_offs = fit.Parameter(g_coup_offs, 'coup_offs')
    p0 = [a, tau, coup_offs]

    def fitfunc(coup):
        # return o()-a()/np.log((1.+np.exp(-0.5*(2*np.pi*x*tau())**2.))/2.)
        return a() / (np.log(2) - np.log(1 + np.exp(-0.5 * (2 * np.pi * (coup_offs() + coup) * tau()) ** 2)))

    return p0, fitfunc, fitfunc_str


def fit_dephasing_tau(g_a, g_coup, g_tau_offs):
    fitfunc_str = '-a/(ln2-ln(1+e**(-0.5* (2*pi*coup*(tau+tau_offs))**2)))'

    a = fit.Parameter(g_a, 'a')
    coup = fit.Parameter(g_coup, 'coup')
    tau_offs = fit.Parameter(g_tau_offs, 'tau_offs')
    p0 = [a, coup, tau_offs]

    def fitfunc(tau):
        return a() / (np.log(2) - np.log(1 + np.exp(-0.5 * (2 * np.pi * coup() * (tau + tau_offs())) ** 2.)))

    return p0, fitfunc, fitfunc_str


def fit_dephasing_constant_offset(g_a, g_coup, g_offs):
    fitfunc_str = '-a/(ln2-ln(1+e**(-0.5* (2*pi*coup*tau+offs)**2)))'

    a = fit.Parameter(g_a, 'a')
    coup = fit.Parameter(g_coup, 'coup')
    offs = fit.Parameter(g_offs, 'offs')
    p0 = [a, coup, offs]

    def fitfunc(x):
        return a() / (np.log(2.) - np.log(1. + np.exp(- 0.5 * (offs() + x * 2 * np.pi * coup()) ** 2.)))

    return p0, fitfunc, fitfunc_str


def fit_repumping(g_a, g_A, g_tau, g_tau2, g_offs_x, *arg):
    """
    fitfunction for an exponential decay,
        y(x) = A * exp(-x/tau)+ A2 * exp(-x/tau2) + a

    Initial guesses (in this order):
        g_a : offset
        g_A : initial Amplitude
        g_tau : decay constant
        g_tau2 : decay constant 2
        g_offs_x : x offset
    """
    fitfunc_str = 'A * exp(-(x-offs_x)/tau)+ A2 * exp(-(x-offs_x)/tau2) + a'

    a = fit.Parameter(g_a, 'a')
    A = fit.Parameter(g_A, 'A')
    tau = fit.Parameter(g_tau, 'tau')
    tau2 = fit.Parameter(g_tau2, 'tau2')
    offs_x = fit.Parameter(g_offs_x, 'offs_x')
    p0 = [a, A, tau, tau2, offs_x]

    def fitfunc(x):
        return a() + A() * np.exp(-(x - offs_x()) / tau()) + (1 - A()) * np.exp(-(x - offs_x()) / tau2())

    return p0, fitfunc, fitfunc_str


def fit_repumping_p1(g_a, g_A1, g_A2, g_tau, g_tau2, g_offs_x, *arg):
    """
    fitfunction for an exponential decay,
        y(x) = A * exp(-x/tau)+ A2 * exp(-x/tau2) + a

    Initial guesses (in this order):
        g_a : offset
        g_A : initial Amplitude
        g_tau : decay constant
        g_tau2 : decay constant 2
        g_offs_x : x offset
    """
    fitfunc_str = 'A * exp(-(x-offs_x)/tau)+ A2 * exp(-(x-offs_x)/tau2) + a'

    a = fit.Parameter(g_a, 'a')
    A1 = fit.Parameter(g_A1, 'A1')
    A2 = fit.Parameter(g_A2, 'A2')
    tau = fit.Parameter(g_tau, 'tau')
    tau2 = fit.Parameter(g_tau2, 'tau2')
    offs_x = fit.Parameter(g_offs_x, 'offs_x')
    p0 = [a, A1, A2, tau, tau2, offs_x]

    def fitfunc(x):
        return a() - A1() * np.exp(-(x - offs_x()) / tau()) + A2() * np.exp(-(x - offs_x()) / tau2())

    return p0, fitfunc, fitfunc_str


def rb_curve(A, p, C, B, *arg):
    fitfunc_str = 'A * p ** x + C * (x-1) * p **(x-2) + B'
    A = fit.Parameter(A, 'A')
    p = fit.Parameter(p, 'p')
    C = fit.Parameter(C, 'C')
    B = fit.Parameter(B, 'B')

    p0 = [A, p, C, B]

    def fitfunc(x):
        x = np.array(x)
        return A() * p() ** x + C() * (x - 1) * p() ** (x - 2) + B()

    return p0, fitfunc, fitfunc_str


def leakage_saturation(sat, leak_rate, init, *arg):
    fitfunc_str = 'sat*(1-exp(-leak_rate*x)) + init*exp(-leak_rate*x)'

    sat = fit.Parameter(sat, 'sat')
    leak_rate = fit.Parameter(leak_rate, 'leak_rate')
    init = fit.Parameter(init, 'init')

    p0 = [sat, leak_rate, init]

    def fitfunc(x):
        x = np.array(x)
        return sat() * (1 - np.exp(-leak_rate() * x)) + init() * np.exp(-leak_rate() * x)

    return p0, fitfunc, fitfunc_str
