import numpy as np
from qtrl.numerics.tunable_qubits import dressed_frequencies_vs_flux
from qtrl.frequency.two_tone import two_tone
from qtrl.analysis.sweep_plotting import plot_two_tone_vs_flux


def mutual_inductance_matrix(qubits, coils, chip):
    assert len(qubits) == len(coils), "mutual inductance matrix ill-conditioned!"
    M = np.zeros([len(qubits), len(coils)])
    for q, qubit in enumerate(qubits):
        for c, coil in enumerate(coils):
            M[q, c] = 1.0 / chip[qubit][f'Coil_{coil}']['phi0']
    return M


def set_qubit_fluxes(qubit_flux_dict, coils, chip):
    """qubit_flux_dict: {"Q5": float, "Q4": float}
       coils:           array-like specifying which coils are used
       chip:            chip yaml with phi_off and phi0 keys
    """
    qubits = list(qubit_flux_dict.keys())
    M = mutual_inductance_matrix(qubits, coils, chip)
    flux_offsets = np.array(
        [np.mean(np.array([chip[q][f'Coil_{c}']['phi_off'] / chip[q][f'Coil_{c}']['phi0'] for c in coils])) for q in
         qubits])
    currents = np.linalg.inv(M).dot(np.array([qubit_flux_dict[qubit] for qubit in qubit_flux_dict] + flux_offsets))

    currents = {f'coil_{coils[i]}': np.around(currents[i], 7) for i in range(len(coils))}

    return currents  # ,flux_offsets,M


def ramp_down_coils(config, coils=None, step=1e-3, delay=0.1, c_range=0.1):
    if coils is None:
        coils = [1, 2, 3, 4]
    for coil in coils:
        print(f'ramping down coil {coil}...')
        dc_source = config.devices.connections[f'coil_{coil}']
        #         print(dc_source._get_source_mode())
        if dc_source._get_source_mode() == 'VOLT':
            dc_source.ramp_voltage(0.0, step=step * 30.0, delay=delay)
            dc_source.off()
            dc_source._set_source_mode('CURR')

        dc_source.ramp_current(0.0, step=step, delay=delay)
        dc_source.on()
        print(f'coil {coil} current: {dc_source._get_set_output("CURR")}')


def turn_on_coils(config, coils=None, step=1e-3, delay=0.1, c_range=0.1):
    if coils is None:
        coils = [1, 2, 3, 4]
    for coil in coils:
        print(f'turning on {coil}...')
        dc_source = config.devices.connections[f'coil_{coil}']

        print(dc_source.ask('SOUR:FUNC?'))
        if dc_source.ask('SOUR:FUNC?') == 'VOLT':
            dc_source._set_auto_range(True)
            dc_source.ramp_voltage(0.0, step=step * 30.0, delay=delay)
            dc_source.off()
            dc_source._set_source_mode('CURR')
            print(dc_source.ask('SOUR:FUNC?'))

        dc_source.ramp_current(0.0, step=step, delay=delay)
        dc_source.range(c_range)
        dc_source.on()
        print(f'coil {coil} current: {dc_source._get_set_output("CURR")}')


def two_tone_flux_sweep(config, qubits, flux_sweep, qubit_frequencies='predict', tuned_qubit=None, exp_config=None):
    try:
        coil_list = config.two_tone['two_tone_flux_sweep']['coils']
        switch_config = config.two_tone['switch_config']
        currents_to_apply = np.array([set_qubit_fluxes(fl, coil_list, config.chip) for fl in flux_sweep])
        two_tone_vs_flux = {f'Q{qubit}': [] for qubit in qubits}

        for c, currents in enumerate(currents_to_apply):
            # set coils
            for coil in currents:
                config.devices.connections[coil].on()
                config.devices.connections[coil].ramp_current(currents[coil], step=1e-3, delay=0.1)
                set_current = config.devices.connections[coil]._get_set_output("CURR")
                print(f'{coil} current: {set_current*1000} mA')

            for qubit in qubits:
                # flip switch to qubit control line
                config.devices.connections['switch'].channels.switch(switch_config[f'Q{qubit}'])
                flux = flux_sweep[c][f'Q{qubit}']
                if tuned_qubit is None:
                    tuned_qubit = qubit
                tuned_qubit_flux = np.around(flux_sweep[c][f"Q{tuned_qubit}"], 5)
                if qubit_frequencies == 'predict':
                    predict_freq = np.around(
                        dressed_frequencies_vs_flux(flux, qubit, chip=config.chip)['dressed_qubit_frequency'], 3)
                else:
                    predict_freq = qubit_frequencies[f'Q{qubit}']

                sweep_span = config.two_tone['two_tone_flux_sweep']['two_tone_span'] / 1e9
                sweep_points = config.two_tone['two_tone_flux_sweep']['two_tone_points']
                two_tone_freqs = 1e9 * np.linspace(predict_freq - sweep_span / 2.0, predict_freq + sweep_span / 2.0,
                                                   sweep_points)
                if exp_config is None:
                    exp_config = {}
                for q in qubits:
                    exp_config[f'flux_Q{q}'] = flux_sweep[c][f'Q{q}']
                exp_config[f'predict_freq_Q{qubit}'] = predict_freq
                two_tone_data = two_tone(config, qubit, two_tone_freqs,
                                         exp_config=exp_config,
                                         filename=f'flux_index_{str(c).zfill(3)}',
                                         plot_title=f'Two Tone Q{qubit}: $\mathrm{{\Phi}}_{{Q{tuned_qubit}}} = {tuned_qubit_flux} \mathrm{{\Phi}}_0$')
                two_tone_vs_flux[f'Q{qubit}'].append(two_tone_data)

            # to reduce heating, ramp down after each point... slow :(
            ramp_down_coils(config, coil_list)
        tuned_fluxes = np.array([[f[f'Q{tuned_qubit}']] for f in flux_sweep])
        for qubit in qubits:
            if tuned_qubit is None:
                tuned_qubit = qubit

            tuned_fluxes = np.array([[f[f'Q{tuned_qubit}']] for f in flux_sweep])
            plot_two_tone_vs_flux(two_tone_vs_flux[f'Q{qubit}'], title=f'Qubit Flux Sweep Q{qubit}',
                                  filename=rf'{config.datamanager.save_directory}\flux_sweep_Q{qubit}_ID{config.datamanager.exp_ID}',
                                  fluxes=tuned_fluxes)
    except Exception as e:
        print(f'Error: {e}')
    finally:
        ramp_down_coils(config)
        return two_tone_vs_flux