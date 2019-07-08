# Copyright (c) 2018-2019, UC Regents

import py, os, sys
import common
import numpy as np
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


class TestREFERENCE:
    def setup_class(cls):
        common.setup_managers('online')

        from qtrl.managers import VariableManager, MetaManager, PulseManager
        from utils.readout import add_readout

        EXAMPLE_DIR = os.path.join(MYPROJECT_DIR, 'ref_config')
        assert os.path.exists(EXAMPLE_DIR)

        var = VariableManager(os.path.join(EXAMPLE_DIR, 'Variables.yaml'))
        cls.cfg = MetaManager({'variables': var,
            'pulses': PulseManager(os.path.join(EXAMPLE_DIR, 'Pulses.yaml'), var)
        })

        cls.cfg.add_readout = add_readout
        cls.cfg.load()

        cls.output_dir = os.path.join(rf'{os.getcwd()}', 'sequence_outputs')
        if not os.path.exists(cls.output_dir):
            os.makedirs(cls.output_dir)

    def verify_output(self, name, sequence):
        with open(os.path.join('ref_outputs', rf'{name}_output.np'), 'rb') as f:
            ref_array = np.load(f)

        assert np.sum(ref_array != sequence.array) == 0

    def test00_X90(self, caplog):
        """Create a single X90 gate + readout sequence"""

        import logging
        caplog.set_level(logging.DEBUG, logger='qtrl')

        from qtrl.sequencer import Sequence

        qubits = [0]

        seq = Sequence(n_elements=2, name='single_X90')

        e_ref = 'Start'
        for qubit in qubits:
            s_ref, e_ref = seq.append([self.cfg.pulses[f'Q{qubit}/X90']], element=0, end_delay=10*ns)

        self.cfg.add_readout(self.cfg, seq=seq, herald_refs=['Start', 'Start'], readout_refs=['Start', e_ref])
        seq.compile()

        assert seq.array.shape == (12, 2, 12110, 1)

        active_channels = 0
        for idx, val in enumerate(seq.array):
            if val.any():
                active_channels += 1

        assert active_channels == 5     # 2 for qubit, 3 for readout

    def test01_rabi(self):
        """Create and verify a Rabi sequence"""

        from utils.char_sequences import rabi

        qubits = [0]

        kwargs = {'cfg': self.cfg,
                  'qubits': qubits,
                  'step_size': 20*ns,
                  'n_elements': 5
                  }

        sequence = rabi(**kwargs)
        self.verify_output('rabi', sequence)

    def test02_rough_pulse_tuning_sequence(self):
        """Create and verify a rough pulse tuning sequence"""

        from qtrl.calibration.single_pulse import rough_pulse_tuning_sequence

        qubits = [0]

        kwargs = {'config': self.cfg,
                  'qubits': qubits,
                  'pulse_type': '90',
                  'min_amplitude': 0.3,
                  'max_amplitude': 0.5,
                 }

        sequence = rough_pulse_tuning_sequence(**kwargs)
        self.verify_output('rough_pulse_tuning_sequence', sequence)

    def test03_ramsey(self):
        """Create and verify a Ramsey sequence"""

        from utils.char_sequences import ramsey

        qubits = [0]

        kwargs = {'cfg': self.cfg,
                  'qubits': qubits,
                  'n_elements': 50,
                  'step_size': 1000*ns,
                  'artificial_detune': 0.1*MHz,
                 }

        sequence = ramsey(**kwargs)
        self.verify_output('ramsey', sequence)

    def test04_drag_calibration_sequence(self):
        """Create and verify a drag calibration sequence"""

        from qtrl.calibration.single_pulse import drag_calibration_sequence

        qubits = [0]

        kwargs = {'cfg': self.cfg,
                  'qubit': qubits[0],
                  'alpha_range': np.linspace(0,3,50)
                 }

        sequence = drag_calibration_sequence(**kwargs)
        self.verify_output('drag_calibration_sequence', sequence)

    def test05_all_xy(self):
        """Create and verify an all xy sequence"""

        from utils.char_sequences import all_xy

        qubits = [0]

        kwargs = {'cfg': self.cfg,
                  'qubit': qubits[0]
                 }

        sequence = all_xy(**kwargs)
        self.verify_output('all_xy', sequence)

    def test06_pi_no_pi(self):
        """Create and verify a pi_no_pi sequence"""

        from utils.char_sequences import pi_no_pi

        qubits = [0]

        kwargs = {'cfg': self.cfg,
                  'qubits': qubits
                 }

        sequence = pi_no_pi(**kwargs)
        self.verify_output('pi_no_pi', sequence)

    def test07_t1(self):
        """Create and verify a T1 sequence"""

        from utils.char_sequences import t1

        qubits = [0]

        kwargs = {'cfg': self.cfg,
                  'qubits': qubits,
                  'n_elements': 25,
                  'step_size': 4000*ns
                 }

        sequence = t1(**kwargs)
        self.verify_output('t1', sequence)

    def test08_echo(self):
        """Create and verify an echo sequence"""

        from utils.char_sequences import echo

        qubits = [0]

        kwargs = {'cfg': self.cfg,
                  'qubits': qubits,
                  'n_elements': 25,
                  'step_size': 4000*ns,
                  'artificial_detune': 0.05*MHz
                 }

        sequence = echo(**kwargs)
        self.verify_output('echo', sequence)

    def test09_interleaved_coherence(self):
        """Create and verify an interleaved coherence sequence"""

        from utils.char_sequences import interleaved_coherence

        qubits = [0]

        kwargs = {'cfg': self.cfg,
                  'qubits': qubits,
                  'n_elements': 10,
                  'step_size': 2500*ns,
                  'artificial_detune': 0.06*MHz,
                  'start_time': 0*us
                 }

        sequence = interleaved_coherence(**kwargs)
        self.verify_output('interleaved_coherence', sequence)
