import os, sys
import numpy as np
sys.path.append("/home/wlav/quantum/Gang/qtrl/")
import pkgutil
import qtrl
package = qtrl
for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
    print("Found submodule {} (is a package: {})".format(modname, ispkg))



# some units for readability
GHz = 1e9
MHz = 1e6
KHz = 1e3
ms = 1e-3
us = 1e-6
ns = 1e-9

class TestREFERENCE:
    def setup_class(cls):
        common.setup_managers('online')

        from qtrl.managers import VariableManager, MetaManager, PulseManager
        from utils.readout import add_readout

        EXAMPLE_DIR = os.path.join(MYPROJECT_DIR, 'ref_config')
        assert os.path.exists(EXAMPLE_DIR)

        var = VariableManager(os.path.join(EXAMPLE_DIR, 'Variables.yaml'))
        cls.cfg = MetaManager({'variables': var,
            'pulses': PulseManager(os.path.join(EXAMPLE_DIR,
'Pulses.yaml'), var)
        })

        cls.cfg.add_readout = add_readout
        cls.cfg.load()

        cls.output_dir = os.path.join(rf'{os.getcwd()}', 'sequence_outputs')
        if not os.path.exists(cls.output_dir):
            os.makedirs(cls.output_dir)

    def verify_output(self, name, sequence):
        with open(os.path.join('/home/wlav/quantum/Gang/qtrl/tests/ref_outputs','{}_output.np'.format(name)), 'rb') as f:
            ref_array = np.load(f)
            print(ref_array)
            print(ref_array.shape)
        #assert np.sum(ref_array != sequence.array) == 0

    def test_arb_seq(self, file_name):

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
test_ref = TestREFERENCE()
test_ref.setup_class()
test_ref.test_arb_seq(sys.argv[1])
