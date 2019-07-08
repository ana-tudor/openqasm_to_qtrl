from collections import namedtuple
from .pulse_manipulation import change_channel
from .char_sequences import rabi
from qtrl.sequencer.yaml_loader import pulse_to_dict


crosstalk_key = namedtuple('crosstalk_key', ['pulse', 'qubit_freq', 'drive_line', 'condition_pulse', 'amp_frac', 'phase_offset'])
#labels for crosstalk measurements use the crosstalk_key named tuple for better clarity in record keeping


def crosstalk_measurement(cfg, qubit_freq, drive_lines, condition_pulse, pulse_name='rabi',
                          amp_fracs=[1], phase_offsets=[0.0], crosstalk_meas={}):
    """This generates composite pulses which go down multiple control lines.
        you pick the qubit frequency of interest, provide
            drive_lines - a list of control lines
            amp_fracs - a list of amplitude scalings
            phase_offsets - a list of phase offsets
            pulse_name - the name of the pulse which will have it's control channel, amplitude and phase altered
            condition_pulse - is the name of a pulse which will be applied at the beginning of the sequence

        results from this are stored in the dictinoary crosstalk_meas under a fully identifying Key.
    """

    # each of the drive_lines, phase_offsets and amp_fracs have to be the same length
    assert len(drive_lines) == len(phase_offsets)
    assert len(phase_offsets) == len(amp_fracs)

    # now we generate a full set of pulses by changing the channel from the provided pulse_name
    pulse = []
    for i in range(len(drive_lines)):
        pulse.extend(change_channel(cfg.pulses[f'Q{qubit_freq}/{pulse_name}'],
                                    drive_line_qubit=drive_lines[i],
                                    phase_offset=phase_offsets[i],
                                    amp_fraction=amp_fracs[i]))

    # we need to store this new pulse object in the pulse YAML temporarily so that it can be sent
    # to the server for acquisition.
    # cfg.pulses[f'Q{qubit_freq}/Temp'] = pulse_to_dict(pulse)

    # run the damn thing
    seq = rabi(cfg,
               qubits=qubit_freq,
               n_elements=50,
               step_size=10 * ns,
               pulse='Temp',
               prepulse=condition_pulse)

    meas = cfg.acquire(seq)

    # Generate a dictionary key describing the measurement which was done
    key = crosstalk_key(f'Q{qubit_freq}/{pulse_name}',
                        qubit_freq,
                        tuple(drive_lines),
                        condition_pulse,
                        tuple(amp_fracs),
                        tuple(phase_offsets))

    return meas