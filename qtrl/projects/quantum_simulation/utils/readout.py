from qtrl import *
from qtrl.sequencer import SequenceWrapper, Sequence


def add_readout(config, seq, readout_refs, herald_refs=None, trig_delay=130e-9, herald_delay=10*us):
    if isinstance(seq, Sequence):
        add_sequence_readout(config, seq, readout_refs, herald_refs, trig_delay, herald_delay)
    elif isinstance(seq, SequenceWrapper):
        add_wrapper_readout(config, seq, readout_refs, herald_refs, trig_delay, herald_delay)
    else:
        raise Exception("Readout addition Failed!")


def add_sequence_readout(config, seq, readout_refs, herald_refs=None, trig_delay=130e-9, herald_delay=5*us):
    """Add readout pulses to the given sequence, only pulses listed in config['readout/readout_qubits]
    will be added to the sequence"""

    n_elements = seq.n_elements
    assert n_elements == len(readout_refs), "Number of readout references (%d) doesn't match n_elements (%d)" % (len(readout_refs), n_elements)

    try:
        qubits = config.variables.get('readout/readout_qubits')
    except KeyError:
        return

    if isinstance(qubits, int):
        qubits = [qubits]

    assert len(qubits) != 0, 'You are not reading out any qubits, really?'

    trig_pulse = config.pulses['Readout/Trigger']
    for elem, r_ref in enumerate(readout_refs):
        seq.append(trig_pulse,
                   start=r_ref,
                   start_delay=trig_delay,
                   element=elem)

        for i, qubit in enumerate(qubits):
            seq.append(config.pulses[f'Readout/R{qubit}'],
                       start=r_ref,
                       element=elem,
                       end_delay=10e-9)

    herald = config.variables['readout/heralding']
    if herald:
        for elem, r_ref in enumerate(herald_refs):
            for i, qubit in enumerate(qubits):
                s_ref, _ = seq.append(config.pulses[f'Readout/R{qubit}'],
                                      start_delay=10*ns,
                                      end=r_ref,
                                      end_delay=herald_delay,
                                      element=elem)
            seq.append(trig_pulse,
                       start=s_ref,
                       start_delay=trig_delay+10*ns,
                       element=elem)


def add_wrapper_readout(config, seq, readout_refs, herald_refs=None, trig_delay=130e-9, herald_delay=10*us):
    """Add readout pulses to the given sequence, only pulses listed in config['readout/readout_qubits]
    will be added to the sequence"""

    n_elements = seq.n_elements
    assert n_elements == len(readout_refs), "Number of readout reference doesn't match n_elements"

    qubits = config.variables.get('readout/readout_qubits')
    if isinstance(qubits, int):
        qubits = [qubits]

    assert len(qubits) != 0, 'You are not reading out any qubits, really?'

    trig_pulse = 'Readout/Trigger'
    for elem, r_ref in enumerate(readout_refs):
        seq.append(trig_pulse,
                   start=r_ref,
                   start_delay=trig_delay,
                   element=elem)

        for i, qubit in enumerate(qubits):
            seq.append(f'Readout/R{qubit}',
                       start=r_ref,
                       element=elem,
                       end_delay=10e-9)

    herald = config.variables['readout/heralding']
    if herald:
        for elem, r_ref in enumerate(herald_refs):
            for i, qubit in enumerate(qubits):
                s_ref, _ = seq.append(f'Readout/R{qubit}',
                                      start_delay=10*ns,
                                      end=r_ref,
                                      end_delay=herald_delay,
                                      element=elem)
                if i == 0:
                    seq.append(trig_pulse,
                               start=s_ref,
                               start_delay=trig_delay+10*ns,
                               element=elem)
