# Author Ravi Naik: rnaik24@gmail.com
import numpy as np
from numpy import linalg as la

def transmon_solver(E_J, E_C, N_c=15, n_levels=5):
    """
    Given an E_J and E_C, diagonalizes the transmon Hamiltonian in the charge basis. N_c specifies the maximum charge
    difference (a.k.a. "gate charge") allowed on the Copper pair box (needs to be sufficiently large for convergence).
    Returns a dictionary of properties of the transmon: its spectrum, anharmonicity, and charge dispersion.
    """

    eigen_freqs = {}
    for N_g in [0.0, 0.5]:  # diagonalize at the gate charge = 0 and 0.5 points to find charge dispersion
        # Construct the n-hat operator
        n_hat = np.diag(np.arange(-float(N_c), float(N_c + 1)))
        # Construct the charging Hamiltonian H_c = 4E_c(n_hat)^2
        H_C = 4.0 * E_C * np.dot(n_hat - N_g * np.identity(2 * N_c + 1), n_hat - N_g * np.identity(2 * N_c + 1))
        # Construct the Josephson Hamiltonian H_J = -E_J/2*(|n><n+1|+|n+1><n|)
        H_J = -E_J / 2.0 * np.diag(np.ones(2 * N_c), 1) - E_J / 2.0 * np.diag(np.ones(2 * N_c), -1)

        E, ev = la.eigh(H_C + H_J)

        eigen_freqs[N_g] = E

    properties = {}
    properties['transition_freqs'] = [eigen_freqs[0.0][n + 1] - eigen_freqs[0.0][n] for n in range(n_levels)]
    properties['absolute_freqs'] = [eigen_freqs[0.0][n] - eigen_freqs[0.0][0] for n in range(n_levels)]
    properties['anharmonicity'] = properties['transition_freqs'][1] - properties['transition_freqs'][0]
    properties['charge_dispersion'] = (eigen_freqs[0.0][1] - eigen_freqs[0.5][1]) - (
            eigen_freqs[0.0][0] - eigen_freqs[0.5][0])

    return properties


def two_body_JC_solver(res1_freqs, res2_freqs, g):
    """
    Takes lists of resonant frequencies (first entry = 0 GHz) for 2 oscillators with vacuum Rabi coupling g and
    diagonalizes the Jaynes-Cummings Hamiltonian.
    Outputs the full spectrum of eigenfrequencies, their respective eigenvectors, and the shape of the Hibert space
    in a dictionary.
    """

    n_levels_1 = len(res1_freqs)
    n_levels_2 = len(res2_freqs)
    id_1 = np.identity(n_levels_1)
    id_2 = np.identity(n_levels_2)

    H_1 = np.kron(np.diag(res1_freqs), id_2)
    H_2 = np.kron(id_1, np.diag(res2_freqs))

    a_1 = np.kron(np.diag(np.sqrt(np.arange(n_levels_1 - 1) + 1), 1), id_2)
    a_2 = np.kron(id_1, np.diag(np.sqrt(np.arange(n_levels_2 - 1) + 1), 1))

    H = H_1 + H_2 + g * (np.dot(a_1.T, a_2) + np.dot(a_1, a_2.T))

    E, ev = la.eigh(H)
    E_1, ev_1 = la.eigh(H_1)
    E_2, ev_2 = la.eigh(H_2)

    properties = {}

    properties['full_spectrum'] = E
    properties['eigenvectors'] = ev
    properties['Hilbert_space'] = (len(res1_freqs), len(res2_freqs))

    return properties


def qubit_res_dressed_frequencies(solver_prop_dict):
    """
    Takes output of two_body_JC_solver and returns the properties of the dressed eigenstates.
    Finds the closest eigenstates to the states in the product state basis and gives the eigenfrequencies associated with that
    state. Can be used to find the"dressed" qubit and resonator frequencies.
    """

    evals = solver_prop_dict['full_spectrum']
    evecs = solver_prop_dict['eigenvectors']
    H_space = solver_prop_dict['Hilbert_space']
    basis = {}
    for i in range(H_space[0]):
        base_i = np.zeros(H_space[0])
        base_i[i] = 1.0
        for j in range(H_space[1]):
            base_j = np.zeros(H_space[1])
            base_j[j] = 1.0
            basis[(i, j)] = np.kron(base_i, base_j)
    dressed_dict = {}
    for ev, evec in zip(evals, evecs.T):
        overlap = {}
        for b, bvec in basis.items():
            overlap[b] = np.abs(np.dot(evec, bvec))
        dressed_dict[max(overlap, key=overlap.get)] = ev

    properties = {}
    properties['dressed_state_dictionary'] = dressed_dict
    properties['dressed_qubit_frequency'] = dressed_dict[(1, 0)]
    properties['dressed_resonator_frequency'] = dressed_dict[(0, 1)]
    properties['dressed_anharmonicity'] = dressed_dict[(2, 0)] - 2 * dressed_dict[(1, 0)]
    properties['chi'] = 0.5 * (dressed_dict[(1, 1)] - dressed_dict[(1, 0)] - dressed_dict[(0, 1)])
    return properties
