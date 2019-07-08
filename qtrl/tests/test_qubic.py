# Copyright (c) 2018-2019, UC Regents

import py, os, sys
import common
from pytest import raises

MYPROJECT_DIR = os.path.curdir

# for readout and char sequences modules
READOUT_UTILS_DIR = os.path.join(os.path.pardir, 'projects', 'quantum_simulation')
assert os.path.exists(READOUT_UTILS_DIR)
sys.path.append(READOUT_UTILS_DIR)

# some units for readability
GHz = 1e9
MHz = 1e6
KHz = 1e3
ms = 1e-3
us = 1e-6
ns = 1e-9


class TestQUBIC:
    def setup_class(cls):
        common.setup_managers('qubic')

    def _managers_setup(self):
        if  not hasattr(self, 'cfg'):
            from qtrl.settings import Settings
            from qtrl.managers import MetaManager

            self.cfg = MetaManager(Settings)

            from utils.readout import add_readout
            self.cfg.add_readout = add_readout

    def test00_meta_manager(self):
        """Initializing the meta manager from settings"""

        self._managers_setup()

    def test00_X90(self, caplog):
        """Create and verify a single X90 gate"""

        import logging
        caplog.set_level(logging.DEBUG, logger='qtrl')

        from qtrl.sequencer import Sequence

        self._managers_setup()

        qubits = [7]     # qubit 0 not in qubitcfg.json

        seq = Sequence(n_elements=2, name='single_X90')

        e_ref = 'Start'
        for qubit in qubits:
            s_ref, e_ref = seq.append([self.cfg.pulses[f'Q{qubit}/X90']], element=0, end_delay=10*ns)

        #self.cfg.add_readout(self.cfg, seq=seq, herald_refs=['Start', 'Start'], readout_refs=['Start', e_ref])
        seq.compile()
        seq._name = 'single_X90'

        #assert seq.array.shape == (12, 2, 12110, 1)

        active_channels = 0
        for idx, val in enumerate(seq.array):
            if val.any():
                active_channels += 1

        #assert active_channels == 5     # 2 for qubit, 3 for readout

    def test01_rabi(self):
        """Create and verify a Rabi sequence"""

        from utils.char_sequences import rabi

        self._managers_setup()

        qubits = [0]

        kwargs = {'cfg': self.cfg,
                  'qubits': qubits,
                  'step_size': 20*ns,
                  'n_elements': 5
                  }

        sequence = rabi(**kwargs)

        """
        from hfbridge import c_hfbridge
        hf = c_hfbridge()
        class c_seqs:
            def __init__(self, seq):
                self.seqlist = [seq]

            def allocate(self):
                pass

            def calcperiod(self):
                pass

        hf.cmdgen0(seqs=c_seqs(sequence))
        """
