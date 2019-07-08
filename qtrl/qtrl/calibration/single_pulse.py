# Copyright (c) 2018-2019, UC Regents

from ..sequencer import Sequence
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from qtrl import *


def rough_pulse_tuning_sequence(config, qubits, pulse_type='180', n_pulses=[1, 2, 3, 5, 11], min_amplitude=0.01,
                                max_amplitude=0.2, prepulse=None, postpulse=None):
    n_pulses = np.array(n_pulses)
    if pulse_type == '90':
        n_pulses = 2 * np.array(n_pulses)

    pulse = 'X{}'.format(pulse_type)
    amp_sweep = np.linspace(min_amplitude, max_amplitude, 20)

    seq = Sequence(n_elements=len(n_pulses) * len(amp_sweep))

    readout_refs = []
    herald_refs = []

    for k, amp in enumerate(amp_sweep):
        r = []
        h = []
        for i, n in enumerate(n_pulses):
            element = i * len(amp_sweep) + k
            for qubit in qubits:
                # loop this n times making sure end of each pulse is at the end of the previous
                e = "Start"
                if prepulse is not None:
                    s, e = seq.append(prepulse, element=element)
                # print(pulse)
                for nq in range(n):
                    s, e = seq.append(config.pulses[f'Q{qubit}/{pulse}'],
                                      element=element,
                                      start=e,
                                      end_delay=0 * ns,
                                      kwargs={'global_amp': amp},
                                      start_delay=10 * ns)
                if postpulse is not None:
                    _, e = seq.append(postpulse, element=element, start=e, start_delay=10 * ns)
            r.append(e)
            h.append(s)
        herald_refs.append(h)
        readout_refs.append(r)

    readout_refs = np.array(readout_refs, ).T.reshape(-1)
    herald_refs = np.array(herald_refs, ).T.reshape(-1)

    # assuming that the add_readout function has been added to the config object
    config.add_readout(config=config, seq=seq, readout_refs=readout_refs, herald_refs=herald_refs)
    seq._name = 'rough_pulse_tuning'  # this sets what subdirectory to save the data into after the acquisition
    seq.compile()
    return seq


def rough_pulse_tuning(config, qubits, pulse_type='180', n_pulses=[1, 2, 3, 5, 11], min_amplitude=0.01,
                       max_amplitude=0.2, prepulse=None, postpulse=None, plot_result=True):
    amp_sweep = np.linspace(min_amplitude, max_amplitude, 20)
    seq = rough_pulse_tuning_sequence(config, qubits, pulse_type=pulse_type, n_pulses=n_pulses,
                                      min_amplitude=min_amplitude, max_amplitude=max_amplitude,
                                      prepulse=prepulse, postpulse=postpulse)
    config.DAC.write_sequence(seq)
    meas = config.acquire(seq, n_reps=1000, n_batches=1, blocking=True)#['measurement']
    amps = {}
    for qubit in qubits:
        # Normalize the data and remove the mean
        fit_data = meas[f'R{qubit}']['Classified'].mean(0).reshape(-1, len(amp_sweep)).T
        fit_data -= np.mean(fit_data)
        fit_data /= np.max(np.abs(fit_data))

        if plot_result:
            plt.figure(figsize=(9, 4))
            plt.subplot(1, 2, 1)
            plt.pcolormesh(fit_data)
            plt.ylabel('Amplitude')
            plt.xlabel('N Pulses')
            plt.colorbar()
            plt.xticks(np.arange(len(n_pulses)) + 0.5, n_pulses)
            plt.yticks(np.arange(len(amp_sweep)) + 0.5, np.around(amp_sweep, 3));

        fit_data = fit_data.reshape(-1, order='F')

        def f(x, amp):
            n, amp_in = x
            return np.abs(np.fft.rfft(np.cos(np.pi * amp_in / amp * n)))

        spacing = np.linspace(min_amplitude, 2 * max_amplitude, 10000)

        error = [np.linalg.norm(f(np.array(list(product(n_pulses, amp_sweep))).T, x) - np.abs(np.fft.rfft(fit_data)))
                 for x in spacing]

        amp = np.around(spacing[np.argmin(error)], 5)
        if pulse_type == '90':
            spacing /= 2.
            amp /= 2.

        if plot_result:
            plt.subplot(1, 2, 2)
            plt.xlabel("Amplitude")
            plt.ylabel("Fit Error")
            plt.plot(spacing, error, label='Qubit {} {}'.format(qubit, pulse_type))
            plt.axvline(spacing[np.argmin(error)], ls='--', c='red', label=amp, alpha=0.2)
            plt.legend()
            assert 'save_path' in meas.keys(), "no save path known!"
            save_path = meas['save_path']['filename']
            plt.savefig(save_path, dpi=200)
        amps[qubit] = amp
    return amps, seq


def drag_calibration_sequence(cfg, qubit, alpha_range=None):
    """Calibrate DRAG alpha parameter using a 2 pulse combination with a sweep of the DRAG
    alpha parameter, Only works for 1 qubit at a time"""

    first_pulses = [['X90'], ['Y90', ]]

    second_pulses = [['Y180'], ['X180']]

    if alpha_range is None:
        alpha_range = np.linspace(-2, 2, 50)

    seq = Sequence(n_elements=2 * len(alpha_range))

    readout_refs = []

    for i, alpha in enumerate(alpha_range):
        e_ref = 'Start'
        for pulse in first_pulses[0]:
            s_ref, e_ref = seq.append(cfg.pulses['Q{}/{}'.format(qubit, pulse)],
                                      start=e_ref,
                                      element=2 * i,
                                      kwargs={'alpha': float(alpha)})

        for pulse in second_pulses[0]:
            s_ref, e_ref = seq.append(cfg.pulses['Q{}/{}'.format(qubit, pulse)],
                                      start=e_ref,
                                      element=2 * i,
                                      kwargs={'alpha': float(alpha)})
        readout_refs.append(e_ref)
        e_ref = 'Start'
        for pulse in first_pulses[1]:
            s_ref, e_ref = seq.append(cfg.pulses['Q{}/{}'.format(qubit, pulse)],
                                      start=e_ref,
                                      element=2 * i + 1,
                                      kwargs={'alpha': float(alpha)})

        for pulse in second_pulses[1]:
            s_ref, e_ref = seq.append(cfg.pulses['Q{}/{}'.format(qubit, pulse)],
                                      start=e_ref,
                                      element=2 * i + 1,
                                      kwargs={'alpha': float(alpha)})
        readout_refs.append(e_ref)

    cfg.add_readout(cfg,
                    seq=seq,
                    herald_refs=seq.n_elements * ['Start'],
                    readout_refs=readout_refs)
    seq._name = 'drag_calibration'
    seq.compile()
    return seq


def drag_calibration(cfg, qubit, alpha_range=None, input_name='Classified', plot=False):
    """Calibrate DRAG alpha parameter using a 2 pulse combination with a sweep of the DRAG
    alpha parameter, Only works for 1 qubit at a time"""

    first_pulses = [['X90'], ['Y90', ]]

    second_pulses = [['Y180'], ['X180']]

    if alpha_range is None:
        alpha_range = np.linspace(-2, 2, 50)

    seq = Sequence(n_elements=2 * len(alpha_range))

    readout_refs = []

    for i, alpha in enumerate(alpha_range):
        e_ref = 'Start'
        for pulse in first_pulses[0]:
            s_ref, e_ref = seq.append(cfg.pulses['Q{}/{}'.format(qubit, pulse)],
                                      start=e_ref,
                                      element=2 * i,
                                      kwargs={'alpha': float(alpha)})

        for pulse in second_pulses[0]:
            s_ref, e_ref = seq.append(cfg.pulses['Q{}/{}'.format(qubit, pulse)],
                                      start=e_ref,
                                      element=2 * i,
                                      kwargs={'alpha': float(alpha)})
        readout_refs.append(e_ref)
        e_ref = 'Start'
        for pulse in first_pulses[1]:
            s_ref, e_ref = seq.append(cfg.pulses['Q{}/{}'.format(qubit, pulse)],
                                      start=e_ref,
                                      element=2 * i + 1,
                                      kwargs={'alpha': float(alpha)})

        for pulse in second_pulses[1]:
            s_ref, e_ref = seq.append(cfg.pulses['Q{}/{}'.format(qubit, pulse)],
                                      start=e_ref,
                                      element=2 * i + 1,
                                      kwargs={'alpha': float(alpha)})
        readout_refs.append(e_ref)

    cfg.add_readout(cfg,
                    seq=seq,
                    herald_refs=seq.n_elements * ['Start'],
                    readout_refs=readout_refs)
    seq._name = 'drag_calibration'
    seq.compile()
    cfg.DAC.write_sequence(seq)
    meas = cfg.acquire(seq, n_reps=1000, n_batches=1, blocking=True)
    counts = meas[f'R{qubit}'][input_name].mean(0).reshape(-1, 2)

    m_1, b_1 = np.polyfit(alpha_range, counts[:, 0], 1)
    m_2, b_2 = np.polyfit(alpha_range, counts[:, 1], 1)

    x_int = (b_2 - b_1) / (m_1 - m_2)
    if plot:
        plt.figure()
        plt.plot(alpha_range, counts[:, 0], label=f'{first_pulses[0][0]},{second_pulses[0][0]}')
        plt.plot(alpha_range, counts[:, 1], label=f'{first_pulses[1][0]},{second_pulses[1][0]}')
        plt.plot(alpha_range, np.poly1d([m_1, b_1])(alpha_range), c='Black')
        plt.plot(alpha_range, np.poly1d([m_2, b_2])(alpha_range), c='Black')
        plt.axvline(x_int, c='black', ls='--', label=f'alpha={np.around(x_int, 3)}')
        plt.xlabel('DRAG Coefficient (MHz)')
        plt.legend()

    return x_int
