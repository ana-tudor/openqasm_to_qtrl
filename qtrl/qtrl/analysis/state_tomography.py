# Copyright (c) 2018-2019, UC Regents

import numpy as np
import itertools
from functools import reduce
from .gates import full_set
from . import tensor, meas_decomp, trace, sigmaz, I
from itertools import product
try:
    import picos
    picos_available = True
except ImportError:
    picos_available = False

try:
    import cvxopt as cvx
    cvx_available = True
except ImportError:
    cvx_available = False


def generate_tomo_pulse_list(n_qubits=1, tomography_list=None):
    """Pass a list of names, like ['I', 'X_2', 'Y_2'] and
       returns all possible unique combinations"""
    if tomography_list is None:
        tomography_list = ['I', 'X90', 'Y90']
    return list(product(tomography_list, repeat=n_qubits))


def binary_arrays_to_bins(x, system_levels=2):
    """Assuming that data is big endian binary, with the form [bit_depth, ..., ...]
       where ... is any other additional set of dimensions, and bit depth
       index corresponds to 2^bit_depth, the calculates the value along the
       bit depth dimension and returns an array of shape [..., ...]

       Example: [0, 1] corresponds to the binary number 0b10, and
                binary_arrays_to_bins([0, 1])
                returns 2

       Example: [[1, 1, 0, 0, 1], [1, 0, 1, 0, 1]] corresponds to
                [0b11, 0b01, 0b10, 0b00, 0b11]
                [3, 1, 2, 0, 3]"""

    x = np.array(x)
    return np.sum(np.einsum('i..., i->i...', x, system_levels ** np.arange(x.shape[0])), 0)


def construct_meas_from_list(meas_list, projectors, mapping=full_set):
    measurements = []
    meas_names = []
    for c in projectors:
        for x in meas_list:
            rot = np.conj(tensor(*[mapping[y] for y in x])).T
            measurements.append(rot*c*rot.H)
            meas_names.append("{}  {}".format(np.argmax(np.diag(c)), x))
    return measurements, meas_names


def create_projectors(meas_operators):
    """Taking a list of measurements that can be done, this creates the projectors onto all the possible subspaces
       defined by the different possible combinations of the measurements.
       assumes that the measurement operators commute"""
    return [reduce(np.dot, x) for x in list(itertools.product(*[meas_decomp(x)[0] for x in meas_operators]))[::-1]]


def standard_tomography(measurement_avgs, n_qubits=1, meas_operator=sigmaz, tomography_list=None):
    """Tomography taking a number of measurement operators and their results,
    returning an estimate of the density matrix"""

    meas_ops = []
    for q in range(n_qubits):
        ops = [I() for _ in range(q)]
        ops.append(meas_operator)
        ops.extend([I() for _ in range(n_qubits-q-1)])
        meas_ops.append(tensor(*ops))

    projectors = create_projectors(meas_ops)

    pre_rotation_list = generate_tomo_pulse_list(n_qubits=n_qubits,
                                                 tomography_list=tomography_list)

    meas_matrices, names = construct_meas_from_list(pre_rotation_list, projectors)

    N = len(meas_matrices)  # Number of measurement operators
    d = meas_matrices[0].shape[0]  # Dimension of the state space

    # Build measurement matrix from given operators
    A = np.zeros((N, d ** 2), dtype=complex)
    for i, v in enumerate(meas_matrices):
        A[i, :] = np.reshape(np.conj(np.array(v)), (1, -1))

    # solve for our density matrix
    rho_uncons = np.linalg.lstsq(A, np.array(measurement_avgs), rcond=-1)[0]
    rho_uncons = np.reshape(rho_uncons, (d, d))
    
    return rho_uncons


def compressed_sensing_tomography(measurement_ops, measurement_avgs, epsilon=.5):
    """Compressed Sensing based tomography, this should reproduce the results of standard_tomography
    in the limit of more measurements."""

    if not picos_available:
        raise Exception("picos package not installed, please ensure picos and cvxopt are installed.")
    if not cvx_available:
        raise Exception("cvxopt package not installed, please ensure picos and cvxopt are installed.")

    def e(dim=2, initial_state=0):
        """Return the state vector of a system in a pure state"""
        assert dim > 0
        state = np.matrix(np.zeros(dim, dtype='complex')).T
        state[initial_state % dim] = 1
        return state

    n = np.shape(measurement_ops)[1]  # dimension of space

    cvx_pops = cvx.matrix(measurement_avgs)
    reshaped_matrix = np.array(measurement_ops).reshape(-1, n)

    F = picos.Problem()
    Z = F.add_variable('Z', (n, n), 'hermitian')

    # This does the trace of the measurement matricies * Z,
    # which should result in the populations measured
    meas_opt = picos.sum([cvx.matrix(reshaped_matrix[i::n, :]) * Z * cvx.matrix(e(n, i)) for i in range(n)], 'i')

    F.set_objective("min", pic.trace(Z))
    F.add_constraint(picos.norm(meas_opt - cvx_pops) < epsilon)
    F.add_constraint(Z >> 0)
    F.solve(verbose=0)

    return np.matrix(Z.value) / trace(Z.value)


def project_and_normalize_density_matrix(rho_uncons):
    """Take a density matrix that is possibly not positive semi-definite, and also not trace one, and 
    return the closest positive semi-definite density matrix with trace-1 using the algorithm in
    PhysRevLett.108.070502
    """

    rho_uncons = rho_uncons / trace(rho_uncons)

    d = rho_uncons.shape[0]  # the dimension of the Hilbert space
    [eigvals_un, eigvecs_un] = np.linalg.eigh(rho_uncons)
    # print eigvals_un
    # If matrix is already trace one PSD, we are done
    if np.min(eigvals_un) >= 0:
        # print 'Already PSD'
        return rho_uncons
    # Otherwise, continue finding closest trace one, PSD matrix through eigenvalue modification
    eigvals_un = list(eigvals_un)
    eigvals_un.reverse()
    eigvals_new = [0.0] * len(eigvals_un)
    i = d
    a = 0.0  # Accumulator
    while eigvals_un[i - 1] + a / float(i) < 0:
        a += eigvals_un[i - 1]
        i -= 1
    for j in range(i):
        eigvals_new[j] = eigvals_un[j] + a / float(i)
    eigvals_new.reverse()

    # Reconstruct the matrix
    rho_cons = np.dot(eigvecs_un, np.dot(np.diag(eigvals_new), np.conj(eigvecs_un.T)))

    return rho_cons

