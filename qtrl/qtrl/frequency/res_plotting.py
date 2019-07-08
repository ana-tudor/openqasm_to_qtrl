import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal
from qtrl.fitting import resonator_fitting as res_fit

def plot_resonator(freqs, trace, filename=None,fit_errors=True):
    """Plot a resonator with fitting from a VNA measurement
    Args:
        freqs - list of frequencies in GHz
        trace - complex trace from the VNA, same length as freq
    """

    fit_params = res_fit.resonator_regression(freqs, trace)

    plt.figure(figsize=(10, 5.3))
    plt.subplot(2,2,1)
    plt.plot(freqs, (np.angle(trace)), '.')
    plt.plot(freqs, (np.angle(res_fit.resonator_f_to_S(freqs, *fit_params[0]))))
    plt.ylabel("Phase")
    plt.xticks([])
    plt.subplot(2,2,3)
    plt.plot(freqs, 20*np.log10(np.abs(trace)), '.', label='data')
    plt.plot(freqs, 20*np.log10(np.abs(res_fit.resonator_f_to_S(freqs, *fit_params[0]))), label='fit')
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency (GHz)")
    for tick in plt.gca().get_xticklabels():
        tick.set_rotation(30)
    plt.legend()
    if fit_errors:
        fit_result = res_fit.reflection_resonator_params(*fit_params[0],covar=fit_params[1])
        # fit_results.append(fit_result)
        s = ''
        for k in fit_result:
            if k in ['a', 'b', 'phi','tau']:
                continue
            if '_sd' not in k:
                s = s + '{:<22}{} ({})\n'.format(k, np.format_float_scientific(fit_result[k], precision=6),
                                                      np.format_float_scientific(fit_result[k+'_sd'], precision=2))
    else:
        fit_result = res_fit.reflection_resonator_params(*fit_params[0])
        # fit_results.append(fit_result)
        s = 'Fit Parameters:\n'
        for k in fit_result:
            s = s + '{:<22}{:<18.7}\n'.format(k, fit_result[k])

    plt.subplot(2,2,2)
    plt.box(False)
    plt.figtext(0.53,0.95,s, va='top', ha='left', )
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2,2,4)
    plt.plot(np.real(trace),np.imag(trace),'.')
    plt.plot(np.real(res_fit.resonator_f_to_S(freqs, *fit_params[0])), np.imag(res_fit.resonator_f_to_S(freqs, *fit_params[0])))
    plt.axvline(0,c='gray')
    plt.axhline(0, c='gray')
    plt.ylabel("I [arb.]")
    plt.xlabel("Q [Arb]")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=200)
    return fit_result


def plot_resonator_location(freqs, trace, n_res=8, filename=None, peak_fitting=[1, 10, 80], show_plot=True):
    angle_diff = np.abs(np.diff(np.unwrap(np.angle(trace))))

    plt.plot(freqs[1:], angle_diff)

    peaks = np.array(sp.signal.find_peaks_cwt(angle_diff, np.array(peak_fitting)))

    peaks = peaks[np.argsort(angle_diff[peaks])[::-1]][:n_res]
    for p in peaks:
        plt.axvline(freqs[1:][p], ls='--', c='red')
    plt.title("Resonator Locations")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Phase Difference")
    r_freqs = sorted(freqs[1:][peaks])
    plt.legend(['Trace', '{}'.format('\n'.join(map(str, np.around(r_freqs, 5))))])
    if filename is not None:
        plt.savefig(filename, dpi=200)
    if show_plot:
        plt.show()
    return r_freqs
