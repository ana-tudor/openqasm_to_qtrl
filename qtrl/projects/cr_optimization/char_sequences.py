from .readout import add_readout
from qtrl.sequencer import Sequence
from qtrl import *
import numpy as np
import matplotlib.pyplot as plt

from qtrl.calibration.readout import find_rotation, find_GMM
from qtrl.processing.fitting_time_domain import FitSinExpAll
from qtrl.processing.plotting import plot_heatmap

def rabi(cfg, qubits, n_elements=100, step_size=100*ns, pulse='rabi', trig_delay=500*ns, prepulse=None, postpulse=None, **kwargs):
    cfg.pulses.load()

    seq = Sequence(n_elements=n_elements,
                   x_axis=step_size * 1e6)

    if isinstance(qubits, int):
        qubits = [qubits]
    # assert len(qubits) > 0, 'Specify a qubit or qubits'

    readout_refs = []
    herald_refs = []
    e_ref = 'Start'
    s_ref = 'Start'
    for element in range(seq.n_elements):
        for q in qubits:
            if prepulse is not None:
                s_ref, e_ref = seq.append(prepulse, element=element)
            _, e_ref = seq.append(cfg.pulses[f'Q{q}/{pulse}'],
                                  start_delay=10*ns,
                                  start=s_ref,
                                  element=element,
                                  width=step_size * element,
                                  **kwargs)

            if postpulse is not None:
                _, e_ref = seq.append(postpulse, element=element, start=e_ref)
        # herald_refs.append(s_ref)
        herald_refs.append("Start")
        readout_refs.append(e_ref)

    add_readout(config=cfg,
                seq=seq,
                readout_refs=readout_refs,
                herald_refs=herald_refs,
                trig_delay=trig_delay)
    seq._name = 'rabi'
    seq.compile()
    return seq


def calibrate_readout(cfg, qubits,show_plot=False):
    cfg.load()

    # disable auto-rotation of constellation plots
    cfg.ADC['processing/00_simple_readout/02_rotate_IQ/enable'] = False
    # disable heralding, time-domain fitting, and joint readout
    # since we don't have classification running yet
    cfg.ADC['processing/05_advanced_readout/06_heralding/enable'] = False
    cfg.ADC['processing/05_advanced_readout/06_heralding_heterodyne/enable'] = False
    cfg.ADC['processing/10_fitting/enable'] = False
    cfg.ADC['processing/15_joint_readout/enable'] = False
    cfg.load()
    cfg.devices.reset_devices()

    # make the sequence
    seq = rabi(cfg, qubits=qubits, step_size=5 * ns, n_elements=50)
    seq._name = 'readout_calibration'
    cfg.write_sequence(seq)
    cfg.DAC.trigger_interval(cfg.DAC['clock_delay'])
    # do the measurement
    meas = cfg.acquire(seq, n_reps=100, n_batches=10)

    # find and apply the angle of rotation
    cfg.ADC['processing/00_simple_readout/02_rotate_IQ/kwargs/angles'] = [float(np.around(x, 3)) for x in
                                                                          find_rotation(meas, display=False)]
    cfg.ADC['processing/00_simple_readout/02_rotate_IQ/enable'] = True
    cfg.load()
    cfg.ADC._post(seq)

    # do the classification and re-enable features that require classification
    results = find_GMM(meas, n_gaussians=2)
    results['result_name'] = 'GMM'
    cfg.ADC['processing/00_simple_readout/055_GMM/kwargs'] = results
    cfg.ADC['processing/00_simple_readout/02_rotate_IQ/enable'] = False
    cfg.ADC['processing/10_fitting/enable'] = True
    cfg.load()

    cfg.ADC._post(seq)

    plot_heatmap(meas, log=True, config=cfg,show_plot=show_plot)

    cfg.ADC['processing/00_simple_readout/02_rotate_IQ/enable'] = True
    cfg.ADC['processing/05_advanced_readout/06_heralding/enable'] = True
    cfg.ADC['processing/05_advanced_readout/06_heralding_heterodyne/enable'] = True
    cfg.ADC['processing/05_advanced_readout/06_heralding/enable'] = False
    cfg.ADC['processing/05_advanced_readout/06_heralding_heterodyne/enable'] = False
    cfg.ADC['processing/15_joint_readout/enable'] = True
    # cfg.ADC._processing['20_record_keeping']['save_config'][0].recent_measurements.append(meas['save_path'])
    cfg.load()
    return meas
def t1(cfg, qubits, n_elements=20, step_size=1*us):
    cfg.pulses.load()

    seq = Sequence(n_elements=n_elements,
                   x_axis=step_size * 1e6)

    if isinstance(qubits, int):
        qubits = [qubits]
    # assert len(qubits) > 0, 'Specify a qubit or qubits'

    readout_refs = []
    herald_refs = []
    for element in range(seq.n_elements):
        e_ref = 'Start'
        for q in qubits:
            s_ref, e_ref = seq.append(cfg.pulses[f'Q{q}/X180'],
                                      start_delay=10*ns,
                                      element=element,
                                      end_delay=step_size * element)

        readout_refs.append(e_ref)
        herald_refs.append("Start")

    add_readout(config=cfg,
                seq=seq,
                readout_refs=readout_refs,
                herald_refs=herald_refs)
    seq._name = 'T1'  # this sets what subdirectory to save the data into after the acquisition
    seq.compile()
    return seq


def ramsey(cfg, qubits, n_elements=20, step_size=25*ns, artificial_detune=0*MHz, start_time=0*us, trig_delay=500*ns, prepulse=None, postpulse=None):
    cfg.pulses.load()

    seq = Sequence(n_elements=n_elements,
                   x_axis=np.arange(start_time,start_time + n_elements*step_size, step_size)*1e6)

    if isinstance(qubits, int):
        qubits = [qubits]
    assert len(qubits) > 0, 'Specify a qubit or qubits'

    readout_refs = []
    herald_refs = []
    for element in range(seq.n_elements):
        phase = (start_time + element*step_size)*artificial_detune * 360
        for q in qubits:
            e_ref = "Start"
            if prepulse is not None:
                _, e_ref = seq.append(prepulse, element=element)
            s_ref, e_ref = seq.append(cfg.pulses[f'Q{q}/X90'],
                                      start=e_ref,
                                      start_delay=10*ns,
                                      element=element,
                                      end_delay=start_time + step_size * element)

            s_ref, e_ref = seq.append(cfg.pulses[f'Q{q}/Z{phase}'],
                                      start=e_ref,
                                      element=element)
            s_ref, e_ref = seq.append(cfg.pulses[f'Q{q}/X90'],
                                      start=e_ref,
                                      element=element)
            if postpulse is not None:
                _, e_ref = seq.append(postpulse, element=element, start=e_ref)
        readout_refs.append(e_ref)
        herald_refs.append("Start")

    add_readout(config=cfg,
                seq=seq,
                readout_refs=readout_refs,
                herald_refs=herald_refs,
                trig_delay=trig_delay)
    seq._name = 'ramsey'  # this sets what subdirectory to save the data into after the acquisition
    seq.compile()
    return seq


def echo(cfg, qubits, n_elements=20, step_size=25*ns, artificial_detune=0*MHz):
    cfg.pulses.load()

    seq = Sequence(n_elements=n_elements,
                   x_axis=step_size * 2e6)

    if isinstance(qubits, int):
        qubits = [qubits]
    assert len(qubits) > 0, 'Specify a qubit or qubits'

    herald_refs = []
    readout_refs = []
    for element in range(seq.n_elements):
        phase = element*step_size*artificial_detune * 360 * 2
        for q in qubits:
            s_ref, e_ref = seq.append(cfg.pulses[f'Q{q}/X90'],
                                      start_delay=10*ns,
                                      element=element,
                                      end_delay=step_size * element)

            s_ref, e_ref = seq.append(cfg.pulses[f'Q{q}/Y180'],
                                      start=e_ref,
                                      element=element,
                                      end_delay=step_size * element)

            s_ref, e_ref = seq.append(cfg.pulses[f'Q{q}/Z{phase}'],
                                      start=e_ref,
                                      element=element)

            s_ref, e_ref = seq.append(cfg.pulses[f'Q{q}/X90'],
                                      start=e_ref,
                                      element=element)

        readout_refs.append(e_ref)
        herald_refs.append("Start")

    add_readout(config=cfg,
                seq=seq,
                readout_refs=readout_refs,
                herald_refs=herald_refs)
    seq._name = 'echo'  # this sets what subdirectory to save the data into after the acquisition
    seq.compile()
    return seq

def cw_two_tone(cfg, qubits):
    cfg.pulses.load()

    seq = Sequence(n_elements=2)
    readout_refs = []
    s_ref = 'Start'
    e_ref = 'Start'
    for q in qubits:
        cw_drive = cfg.pulses[f'Q{q}/rabi']
        res_drive = cfg.pulses[f'Readout/R{q}']
        trig_pulse = cfg.pulses['Readout/Trigger']

        clock_delay = cfg.DAC['clock_delay']
        _, e_ref = seq.append(cw_drive,
                   env_func='square',
                   width=clock_delay*ns,
                   start=s_ref)
        seq.append(res_drive,
                   env_func='square',
                   width=clock_delay * ns,
                   start=s_ref,
                   freq=res_drive[0].envelope.kwargs['frequency'])
        # seq.append(trig_pulse,
        #            start=s_ref,
        #            start_delay=130*ns,
        #            )
        seq.append(trig_pulse,
                   start=s_ref,
                   start_delay=clock_delay/2.0*ns,
                   )
        seq.compile()
        seq._name = 'cw_two_tone'
    return seq

def pi_no_pi(cfg, qubits):
    """Assuming X180 pulses are well tuned for the qubits specified,
    This generates a 2 element sequence, element 1 is X180 on all qubit
    element 0 is nothing on all qubits"""
    seq = Sequence(n_elements=2)

    e_ref = 'Start'
    for qubit in qubits:
        s_ref, e_ref = seq.append([cfg.pulses[f'Q{qubit}/X180']], element=1, end_delay=10e-9)

    cfg.add_readout(cfg, seq=seq, herald_refs=['Start', 'Start'], readout_refs=['Start', e_ref])
    seq.compile()
    seq._name = 'pi_no_pi'
    return seq

def pi_no_pi_qutrit(cfg, qubits):
    """Assuming X180 pulses are well tuned for the qubits specified,
    This generates a 2 element sequence, element 1 is X180 on all qubit
    element 0 is nothing on all qubits"""
    seq = Sequence(n_elements=3)

    e_ref = 'Start'
    r_refs = []
    for qubit in qubits:
        s_ref, e_ref0 = seq.append([cfg.pulses[f'Q{qubit}/X180']], element=1, end_delay=10e-9)
        s_ref, e_ref1 = seq.append([cfg.pulses[f'Q{qubit}/X180']], element=2, end_delay=10e-9)
        s_ref, e_ref2 = seq.append([cfg.pulses[f'Q{qubit}/X180_ef']], element=2, start=e_ref1,
                                  end_delay=10e-9)


    cfg.add_readout(cfg, seq=seq, herald_refs=['Start']*3, readout_refs=['Start', e_ref0, e_ref2])
    seq.compile()
    seq._name = 'pi_no_pi_qutrit'
    return seq

def all_xy(cfg, qubit, prepulse=None):
    """AllXY sequence for the GE levels of a qubit"""
    first_pulses = ['I', 'X180', 'Y180', 'X180', 'Y180',  # end in |0> state
                    'X90', 'Y90', 'X90', 'Y90', 'X90', 'Y90',  # end in |0>+|1> state
                    'X180', 'Y180', 'X90', 'X180', 'Y90', 'Y180',  # end in |0>+|1> state
                    'X180', 'Y180', 'X90', 'Y90']  # end in |1> state

    second_pulses = ['I', 'X180', 'Y180', 'Y180', 'X180', 'I', 'I',
                     'Y90', 'X90', 'Y180', 'X180', 'Y90',
                     'X90', 'X180', 'X90', 'Y180', 'Y90',
                     'I', 'I', 'X90', 'Y90']

    seq = Sequence(n_elements=len(first_pulses),x_axis = list(zip(first_pulses,second_pulses)))
    seq._name = 'allxy'
    readout_refs = []
    if prepulse is not None:
        _, og_e_ref = seq.append(prepulse)
    else:
        og_e_ref = "Start"
        
    for i, (p1, p2) in enumerate(zip(first_pulses, second_pulses)):
        _, e_ref = seq.append(cfg.pulses[f'Q{qubit}/{p1}'],
                              start=og_e_ref,
                              element=i)

        _, e_ref = seq.append(cfg.pulses[f'Q{qubit}/{p2}'],
                              start=e_ref,
                              element=i)
        readout_refs.append(e_ref)

    cfg.add_readout(cfg, seq, readout_refs, seq.n_elements*['Start'])
    seq._name = 'all_xy'  # this sets what subdirectory to save the data into after the acquisition
    seq.compile()

    return seq

def chevron(cfg, qubits, span=20*1e6, step=50.0*1e4,name = 'chevron',plot=True):
    cfg.load()
    chev_data_all = {}
    for q in qubits:
        chev_data = []
        center_freq = cfg.variables[f'Q{q}/mod_freq']
        # span = 10 * 1e6
        # chev_step = 5.0 * 1e5  # kHz steps
        freq_sweep = np.arange(center_freq - span / 2.0, center_freq + span / 2.0 + step, step)
        for freq in freq_sweep:
            print(q, (freq - center_freq)/1e6)
            seq = rabi(cfg, qubits=q, step_size=5 * ns, n_elements=50, freq=freq)
            seq._name = name
            cfg.write_sequence(seq)
            meas = cfg.acquire(seq, n_reps=500, n_batches=1)
            chev_data.append(meas)
            FitSinExpAll.plot(meas)
            # plt.show()

        chev_data_all[q] = chev_data

        if plot:
            chevron_plot(cfg, chev_data, q, span=span, step=step)
            assert 'save_path' in chev_data[-1].keys(), "no save path known!"
            save_path = chev_data[-1]['save_path']['filename']
            plt.savefig(save_path + '_ChevronPlot', dpi=200)
    return chev_data_all

def chevron_plot(cfg, chevron_data, qubit, span=20*1e6, step=50.0*1e4):
    plt.figure()
    x_axis = chevron_data[0][f'R{qubit}']['FitExpSinAll']['x']
    rabi_data = np.array([ch[f'R{qubit}']['FitExpSinAll']['y_original'] for ch in chevron_data])
    center_freq = cfg.variables[f'Q{qubit}/mod_freq']
    freq_sweep = np.arange(center_freq - span / 2.0, center_freq + span / 2.0 + step, step)
    plt.pcolormesh(x_axis, freq_sweep/1e6, rabi_data, cmap='RdBu_r')

    plt.axhline(center_freq / 1e6, c='black', ls='dashed', label='Current Frequency')
    plt.title(f'Q{qubit}')
    plt.legend()
    plt.xlabel('t [us]')
    plt.ylabel('Mod. Frequency [MHz]')
    plt.colorbar(label='P(|1>)')
    plt.tight_layout()

def rabi_crosstalk(cfg, qubits, drive_frequency, drive_line, pulse='rabi', n_elements=20, step_size=25*ns, prepulse=None, postpulse=None, **kwargs):
    """

    :param cfg:                 qtrl Config with method pulses
    :param qubits:              list of qubits
    :param drive_frequency:     either: a string specifying the drive frequency from a config or a numerical modulation frequency
    :param drive_line:          integer specifying which qubit line is being driven
    :param pulse:               string specifying the pulse to be
    :param n_elements:
    :param step_size:
    :param prepulse:
    :param postpulse:
    :param kwargs:
    :return:
    """
    cfg.pulses.load()

    seq = Sequence(n_elements=n_elements,
                   x_axis=step_size * 1e6)

    if isinstance(qubits, int):
        qubits = [qubits]
    assert len(qubits) > 0, 'Specify a qubit or qubits'

    if isinstance(drive_frequency, str):
        drive_frequency_value = cfg.variables[drive_frequency]
    else:
        drive_frequency_value = drive_frequency

    if not kwargs:
        kwargs = {}
    kwargs['freq'] = drive_frequency_value
    drive_pulse = cfg.pulses[f'Q{drive_line}/{pulse}']

    readout_refs = []
    herald_refs = []
    phase = 0
    for element in range(seq.n_elements):
        for q in qubits:
            e_ref = "Start"
            if prepulse is not None:
                _, e_ref = seq.append(prepulse, element=element)

            _, e_ref = seq.append(drive_pulse,
                                  start_delay=10 * ns,
                                  start=e_ref,
                                  element=element,
                                  width=step_size * element,
                                  **kwargs)

            if postpulse is not None:
                _, e_ref = seq.append(postpulse, element=element, start=e_ref)
        readout_refs.append(e_ref)
        herald_refs.append("Start")

    add_readout(config=cfg,
                seq=seq,
                readout_refs=readout_refs,
                herald_refs=herald_refs)
    seq._name = 'rabi_crosstalk'  # this sets what subdirectory to save the data into after the acquisition
    seq.compile()
    return seq

def crosstalk_phase(cfg, qubit, drive_line, width_scaling = 1.0, phases = np.linspace(0,2*np.pi,20), prepulse=None, postpulse=None, **kwargs):
    """

    :param cfg:
    :param qubit:
    :param drive_line:
    :param pulse:
    :param n_elements:
    :param step_size:
    :param prepulse:
    :param postpulse:
    :param kwargs:
    :return:
    """
    cfg.pulses.load()
    assert len(phases) > 1, "phases must be len > 1!"

    seq = Sequence(n_elements=len(phases),
                   x_axis=phases)

    pulse = 'X90'
    pulse_width = cfg.pulses[f'Q{qubit}/{pulse}'][0].envelope.width
    pulse_freq = cfg.pulses[f'Q{qubit}/{pulse}'][0].freq

    readout_refs = []
    herald_refs = []
    for i, element in enumerate(range(seq.n_elements)):
        e_ref = "Start"
        # drive_pulse_cross = cfg.pulses[f'Q{drive_line}/{pulse}']
        if prepulse is not None:
            _, e_ref = seq.append(prepulse, element=element)

        _, e_ref = seq.append(cfg.pulses[f'Q{qubit}/{pulse}'],
                              start_delay=10 * ns,
                              start=e_ref,
                              element=element,
                              freq=pulse_freq,
                              )

        _, e_ref = seq.append(cfg.pulses[f'Q{drive_line}/{pulse}'],
                              start=e_ref,
                              element=element,
                              width=width_scaling*pulse_width,
                              freq=pulse_freq,
                              kwargs={'global_phase': phases[i]}
                              )

        if postpulse is not None:
            _, e_ref = seq.append(postpulse, element=element, start=e_ref)
        readout_refs.append(e_ref)
        herald_refs.append("Start")

    add_readout(config=cfg,
                seq=seq,
                readout_refs=readout_refs,
                herald_refs=herald_refs)
    seq._name = 'crosstalk_phase'  # this sets what subdirectory to save the data into after the acquisition
    seq.compile()
    return seq

def stark_shift_measurement(cfg, qubits, drive_frequency, drive_lines, pulse='rabi', n_elements=20, step_size=25*ns, artificial_detune=0*MHz, prepulse=None, postpulse=None, **kwargs):
    """

    :param cfg:                 qtrl Config with method pulses
    :param qubits:              list of qubits
    :param drive_frequency:     either: a string specifying the drive frequency from a config or a numerical modulation frequency
    :param drive_line:          integer specifying which qubit line is being driven
    :param pulse:               string specifying the pulse to be
    :param n_elements:
    :param step_size:
    :param artificial_detune:
    :param prepulse:
    :param postpulse:
    :param kwargs:
    :return:
    """
    cfg.pulses.load()

    seq = Sequence(n_elements=n_elements,
                   x_axis=step_size * 1e6)

    if isinstance(qubits, int):
        qubits = [qubits]
    assert len(qubits) > 0, 'Specify a qubit or qubits'

    # channels = [cfg.variables[f'Q{drive_line}/chan_I'], cfg.variables[f'Q{drive_line}/chan_Q']]

    if isinstance(drive_frequency, str):
        drive_qubit = int(drive_frequency.split('/')[0][1]) # pull drive qubit from the frequency
        drive_frequency_value = cfg.variables[drive_frequency]
    else:
        drive_frequency_value = drive_frequency

    if not kwargs:
        kwargs = {}
    kwargs['freq'] = drive_frequency_value
    if isinstance(drive_lines, int):
        drive_lines = [drive_lines]

    drive_pulses = [cfg.pulses[f'Q{drive_line}/{pulse}'] for drive_line in drive_lines]
    if 'env_func' in kwargs:
        env_func = kwargs['env_func']
    else:
        env_func = drive_pulses[0][0].envelope.env_func
    readout_refs = []
    herald_refs = []
    phase = 0
    for element in range(seq.n_elements):
        phase += step_size * artificial_detune * 360
        for q in qubits:
            e_ref = "Start"
            if prepulse is not None:
                _, e_ref = seq.append(prepulse, element=element)
            s_ref, e_ref = seq.append(cfg.pulses[f'Q{q}/X90'],
                                      start=e_ref,
                                      start_delay=10*ns,
                                      element=element,
                                      )

            s_ref, e_ref = seq.append(cfg.pulses[f'Q{q}/Z{phase}'],
                                      start=e_ref,
                                      element=element)
            pulse_start_ref = e_ref
            for k,drive_pulse in enumerate(drive_pulses):
                if 'crosstalk_matrix' in kwargs:
                    rel_amp, rel_phase = kwargs['crosstalk_matrix'][f'F{drive_qubit}'][f'T{drive_qubit}'][f'L{drive_lines[k]}']

                    # if drive_qubit == drive_lines[k]:
                    #     phase_offset = 0.0
                    # else:
                    phase_offset = np.pi - rel_phase
                    kwargs['global_phase'] = phase_offset
                    kwargs['global_amp'] = 1.0/rel_amp

                _, e_ref = seq.append(drive_pulse,
                                      start_delay=10 * ns,
                                      start=pulse_start_ref ,
                                      element=element,
                                      width=step_size * element,
                                      env_func = env_func,
                                      freq=drive_frequency_value,
                                      kwargs=kwargs
                                      )

            s_ref, e_ref = seq.append(cfg.pulses[f'Q{q}/X90'],
                                      start=e_ref,
                                      element=element)
            if postpulse is not None:
                _, e_ref = seq.append(postpulse, element=element, start=e_ref)
        readout_refs.append(e_ref)
        herald_refs.append("Start")

    add_readout(config=cfg,
                seq=seq,
                readout_refs=readout_refs,
                herald_refs=herald_refs)
    seq._name = 'stark_shift_measurement'  # this sets what subdirectory to save the data into after the acquisition
    seq.compile()
    return seq