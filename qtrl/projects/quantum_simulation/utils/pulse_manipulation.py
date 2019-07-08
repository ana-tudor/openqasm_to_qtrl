import copy


def change_channel(pulse, drive_line_qubit=0, phase_offset=0.0, amp_fraction=1.0):

    # Mapping qubit number to channel pair
    drive_mapping = {0: [1, 0],
                     5: [6, 7],
                     6: [5, 4],
                     7: [3, 2]}
    # qubits 0, 5, 6, 7

    chans = drive_mapping[drive_line_qubit]
    cur_pulses = copy.deepcopy(pulse)
    assert len(cur_pulses) == 2, "Only works for 2 channel IQ pulses currently"
    cur_pulses[0] = cur_pulses[0]._replace(channel=chans[0])
    cur_pulses[1] = cur_pulses[1]._replace(channel=chans[1])

    for c_pulse in cur_pulses:
        kwargs = c_pulse.envelope.kwargs
        cur_phase = kwargs.get('phase', 0)
        cur_phase = [cur_phase] if not isinstance(cur_phase, list) else cur_phase
        cur_phase.append(phase_offset)
        kwargs['phase'] = cur_phase

        cur_amp = kwargs.get('amplitude', 0)
        cur_amp = [cur_amp] if not isinstance(cur_amp, list) else cur_amp
        cur_amp.append(amp_fraction)
        kwargs['amplitude'] =  cur_amp

    return cur_pulses