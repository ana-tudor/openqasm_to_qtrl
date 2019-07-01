import os, sys
import numpy as np
sys.path.append("/home/wlav/quantum/Gang/qtrl/")
import pkgutil
import qtrl
package = qtrl
for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
    print("Found submodule {} (is a package: {})".format(modname, ispkg))

#import qtrl.sequencer
#import importlib.util
#spec = importlib.util.spec_from_file_location("qtrl.sequencer", "/home/wlav/quantum/Gang/qtrl/qtrl/sequencer/sequencer.py")
#sequencer = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(sequencer)


def verify_output(name):
    with open(os.path.join('/home/wlav/quantum/Gang/qtrl/tests/ref_outputs','{}_output.np'.format(name)), 'rb') as f:
        ref_array = np.load(f)
        print(ref_array)
        print(ref_array.shape)
    #assert np.sum(ref_array != sequence.array) == 0

def test_arb_seq(file_name):
    ns = 1e-9

    import logging
    #caplog.set_level(logging.DEBUG, logger='qtrl')

    from qtrl.sequencer import Sequence

    qubits = [0]

    seq = Sequence(n_elements=2, name=file_name[:len(file_name)-5])
    l = [line.rstrip('\n') for line in open(file_name)]


    e_ref = 'Start'
    for line in l:
        s_ref, e_ref = seq.append([self.cfg.pulses[line]], element=0, end_delay=10*ns)
    
    #self.cfg.add_readout(self.cfg, seq=seq, herald_refs=['Start', 'Start'], readout_refs=['Start', e_ref])
    seq.compile()

    assert seq.array.shape == (12, 2, 122110, 1)
    
    active_channels = 0
    for idx, val in enumerate(seq.array):
        if val.any():
            active_channels += 1

    assert active_channels == 5 #2 for qubit, 3 for readout



list_names = ['interleaved_coherence',\
              'echo',\
              't1',\
              'pi_no_pi',\
              'all_xy',\
              'ramsey',\
              'rough_pulse_tuning_sequence',\
              'rabi']

#for name in list_names:
#verify_output(list_names[3])
test_arb_seq(sys.argv[1])
