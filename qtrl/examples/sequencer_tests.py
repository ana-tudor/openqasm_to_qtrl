import sys
import os
import numpy as np
sys.path.append('../')

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
from qtrl.managers import *
from qtrl.utils import Config

#sequences used
from example_sequences import rabi
from example_sequences import ramsey
from example_sequences import all_xy
from example_sequences import pi_no_pi
from example_sequences import t1
from example_sequences import echo
from example_sequences import interleaved_coherence
from utils.readout import add_readout
from qtrl.calibration.single_pulse import rough_pulse_tuning_sequence
from qtrl.calibration.single_pulse import drag_calibration_sequence

# some units for readability
GHz = 1e9
MHz = 1e6
KHz = 1e3
ms = 1e-3
us = 1e-6
ns = 1e-9


# In[2]:


var = VariableManager('Config_Example/Variables.yaml')

cfg = MetaManager({'variables': var,
                   'pulses': PulseManager('Config_Example/Pulses.yaml', var)
                  })
cfg.add_readout = add_readout
cfg.load()


# In[6]:


qubit =[0]
cfg.load()
seq_methods = [rabi,
               rough_pulse_tuning_sequence,
               ramsey,
               drag_calibration_sequence,
               all_xy,
               pi_no_pi,
               t1,
               echo,
               interleaved_coherence
              ]
seq_arg_inputs = [{'cfg': cfg,
                   'qubits': qubit,
                   'step_size': 20*ns,
                   'n_elements': 5
                  },
                  {'config': cfg,
                  'qubits': qubit,
                  'pulse_type': '90',
                  'min_amplitude': 0.3,
                  'max_amplitude': 0.5,
                  },
                  {'cfg': cfg,
                  'qubits': qubit,
                  'n_elements': 50,
                  'step_size': 1000*ns,
                  'artificial_detune': 0.1*MHz,
                  },
                  {'cfg': cfg,
                   'qubit': qubit[0],
                   'alpha_range': np.linspace(0,3,50)
                  },
                  {'cfg': cfg,
                   'qubit': qubit[0]},
                  {'cfg': cfg,
                   'qubits': qubit
                  },
                  {'cfg': cfg, 
                   'qubits': qubit, 
                   'n_elements': 25, 
                   'step_size': 4000*ns
                  },
                  {'cfg': cfg, 
                   'qubits': qubit, 
                   'n_elements': 25, 
                   'step_size': 4000*ns,
                   'artificial_detune': 0.05*MHz
                  },
                  {'cfg': cfg,
                   'qubits': qubit,
                   'n_elements': 10,
                   'step_size': 2500*ns,
                   'artificial_detune': 0.06*MHz,
                   'start_time': 0*us
                  }
                 ]


# In[8]:


for func, args in zip(seq_methods, seq_arg_inputs):
    seq = func(**args)
    DAC_array = seq.array
    save_dir = os.path.join(rf'{os.getcwd()}', 'sequence_outputs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(rf'{save_dir}', rf'{func.__name__}_output.np'),'wb') as f:
        print(rf'saving {save_dir}/{func.__name__}_output')
        np.save(f, DAC_array)


# In[5]:


qubit =[0]
cfg.load()
seq = rabi(cfg, qubits=qubit, step_size=20*1e-9, n_elements=5)
DAC_array = seq.array


# In[6]:


#(channel, element, sequence, 1)
DAC_array.shape

