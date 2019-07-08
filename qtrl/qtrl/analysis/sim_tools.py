# Copyright (c) 2018-2019, UC Regents

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib.font_manager import FontProperties
import itertools
from itertools import product
# import inspect
from collections import OrderedDict
from functools import reduce

sigmax = np.matrix([[0, 1], [1, 0]], dtype='complex')
sigmay = np.matrix([[0, -1j], [1j, 0]], dtype='complex')
sigmaz = np.matrix([[1, 0], [0, -1]], dtype='complex')
x90 = la.expm(-1j*sigmax*np.pi/4)
y90 = la.expm(-1j*sigmay*np.pi/4)
z90 = la.expm(-1j*sigmaz*np.pi/4)
hadamard = np.matrix([[1,1],[1,-1]])/np.sqrt(2)

cnot = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype='complex')
swap = np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype='complex')


gell_mann = {'I': np.eye(3, dtype=complex),
             1: np.matrix([[0,1,0],
                           [1,0,0],
                           [0,0,0]],
                          dtype=complex),
             2: np.matrix([[0 ,-1j, 0],
                           [1j,  0, 0],
                           [0 ,  0, 0]],
                          dtype=complex),
             3: np.matrix([[1, 0, 0],
                           [0,-1, 0],
                           [0, 0, 0]],
                          dtype=complex),
             4: np.matrix([[0,0,1],
                           [0,0,0],
                           [1,0,0]],
                          dtype=complex),
             5: np.matrix([[0, 0,-1j],
                           [0, 0, 0],
                           [1j,0, 0]],
                          dtype=complex),
             6: np.matrix([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0]],
                          dtype=complex),
             7: np.matrix([[0, 0,  0],
                           [0, 0,-1j],
                           [0,1j,  0]],
                          dtype=complex),
             8: np.matrix([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, -2]],
                          dtype=complex)/np.sqrt(3)
             }

qt_csum_p1 = np.matrix([[1,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0],
                        [0,0,1,0,0,0,0,0,0],
                        [0,0,0,0,0,1,0,0,0],
                        [0,0,0,1,0,0,0,0,0],
                        [0,0,0,0,1,0,0,0,0],
                        [0,0,0,0,0,0,0,1,0],
                        [0,0,0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,1,0,0]])
qt_csum_m1 = np.matrix([[1,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,1,0],
                        [0,0,0,0,0,1,0,0,0],
                        [0,0,0,1,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,1,0,0],
                        [0,0,0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0,0,0]])
qt_CPiX_p1 = np.matrix([[1,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0],
                        [0,0,1,0,0,0,0,0,0],
                        [0,0,0,0,-1j,0,0,0,0],
                        [0,0,0,-1j,0,0,0,0,0],
                        [0,0,0,0,0,1,0,0,0],
                        [0,0,0,0,0,0,1,0,0],
                        [0,0,0,0,0,0,0,1,0],
                        [0,0,0,0,0,0,0,0,1]])
qt_CPiY_p1 = np.matrix([[1,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0],
                        [0,0,1,0,0,0,0,0,0],
                        [0,0,0,0,-1,0,0,0,0],
                        [0,0,0,1,0,0,0,0,0],
                        [0,0,0,0,0,1,0,0,0],
                        [0,0,0,0,0,0,1,0,0],
                        [0,0,0,0,0,0,0,1,0],
                        [0,0,0,0,0,0,0,0,1]])
qt_scrambling = np.matrix([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                            [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
                            [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
                            [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
                            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
                            [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
                            [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
qt_X = np.array([[0,1,0],[0,0,1],[1,0,0]], dtype = 'complex64')
qt_Xdag = np.conj(qt_X).T

paulis = OrderedDict([('I', np.matrix(np.eye(2, dtype=complex))),
                      ('X', sigmax),
                      ('Y', sigmay),
                      ('Z', sigmaz)])

def numberToBase(n, b, depth=None):
    if n == 0:
        if depth is None or depth==1:
            return '0'
        else:
            return ''.join(depth*['0'])
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b

    if depth is not None:
        assert depth >= len(digits)

        while len(digits) < depth:
            digits.append(0)
    return ''.join([str(x) for x in digits[::-1]])

def enumerate_levels(n_qudits, qudit_level=2):
    return [numberToBase(x, qudit_level, n_qudits) for x in range(qudit_level**n_qudits)]

def generalized_gell_mann(dim=3):
    """Generates a generalization of the gell-mann matricies (orthogonal, traceless)"""
    # This uses the algorithm as defined on the wolfram website
    # it is broken into 3 parts
    
    gell_manns = []
    
    # Generate the symmetric ones first
    for k in range(dim):
        for i in range(k):
            cur_mat = np.zeros([dim, dim], dtype='complex')
            cur_mat[i, k] = 1
            cur_mat[k, i] = 1
            gell_manns.append(cur_mat)
    
    # Generate the anti-symmetric ones
    for k in range(dim):
        for i in range(k):
            cur_mat = np.zeros([dim, dim], dtype='complex')
            cur_mat[i, k] = -1j
            cur_mat[k, i] = 1j
            gell_manns.append(cur_mat)

    # Diagonals:
    for l in range(1, dim):
        cur_mat = np.zeros([dim, dim], dtype='complex')
        for j in range(1, l+1):
            cur_mat[j-1, j-1] = np.sqrt(2./(l*(l+1)))
            cur_mat[j, j] -= np.sqrt(2./(l*(l+1)))*l
        gell_manns.append(cur_mat)

    return gell_manns


def generate_paulis(n_dimensions=2):
    X = np.roll(np.eye(n_dimensions), 1, 0)
    Z = np.diag([np.exp(-1j*np.pi/n_dimensions*j*2) for j in np.arange(n_dimensions)])

    group = []
    for r in range(n_dimensions):
        for s in range(n_dimensions):
            group.append(np.dot(np.linalg.matrix_power(X, r), np.linalg.matrix_power(Z, s)))
    return group


def ctrl_gate(U, qudit=2):
    u = np.eye(qudit**2, dtype='complex')
    u[qudit:, qudit:] = U
    return np.matrix(u)


def H(theta):
    c = np.cos(2*theta)
    s = np.sin(2*theta)
    return np.matrix([[c, s], [s, -c]])


def rot_x(theta):
    """Rotation matrix around the x axis"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.matrix([[c, -1j * s], [-1j * s, c]])


def rot_y(theta):
    """Rotation matrix around the y axis"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.matrix([[c, -s], [s, c]])


def rot_z(theta):
    """Rotation matrix around the z axis"""
    exp_z = np.exp(1j * theta / 2.)
    exp_z_inv = np.exp(-1j * theta / 2.)
    return np.matrix([[exp_z_inv, 0], [0, exp_z]])

def trit_z(theta, level=-1):
    '''
    we used this for defining the qutrit Z gates we implement in the lab
    '''

    m = np.eye(3, dtype='complex')
    if level == -1:
        # here we're attaching the phase to the ground state, so
        m[0,0] = np.exp(-1j*theta)
        jac = np.diag([-1j*np.exp(-1j*theta),0,0])
    elif level == 1:
        m[2,2] = np.exp(1j*theta)
        jac = np.diag([0,0,1j*np.exp(1j*theta)])
    else:
        m[1,1] = np.exp(1j*theta)
        jac = np.diag([0,1j*np.exp(1j*theta),0])
    return [m, jac]

def trit_H_z(theta, level=-1):
    '''
    we used this to return the hamiltonian of the Z instead of the unitary
    '''

    m = np.zeros((3,3), dtype='complex')
    if level == -1:
        # here we're attaching the phase to the ground state, so
        m[0,0] = theta
    elif level == 1:
        m[2,2] = -1*theta
    else:
        m[1,1] = -1*theta
    return [m, None]
def rot_n(theta, n):
    """Rotation matrix about an arbitrary 3d-vector n by angle theta
    n should be specified [nx, ny, nz]
    but need not be normalized"""

    assert len(n) == 3

    # normalize vector n
    n_normalized = np.array(n)/np.linalg.norm(n, ord = 2)

    # create rotation matrix
    return np.cos(theta/2.)*np.eye(2) - 1j*np.sin(theta/2.)*(n[0]*sigmax + n[1]*sigmay + n[2]*sigmaz)

def qt_arb_rot(Theta_1, Theta_2, Theta_3, Phi_1, Phi_2, Phi_3, Phi_4, Phi_5):
    """Using the parameterization found in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.38.1994,
        this method constructs an arbitrary single_qutrit unitary operation.

        Arguments:
        qutrit_params: a list of eight parameters, in the following order
            Theta_1, Theta_2, Theta_3, Phi_1, Phi_2, Phi_3, Phi_4, Phi_5
        The formula for the matrix is:
            u11 = cos[Theta_1]*cos[Theta_2]*exp[i*Phi_1]
            u12 = sin[Theta_1]*exp[i*Phi_3]
            u13 = cos[Theta_1]*sin[Theta_2]*exp[i*Phi_4]
            u21 = sin[Theta_2]*sin[Theta_3]*exp[-i*Phi_4 - i*Phi_5] -
                    sin[Theta_1]*cos[Theta_2]*cos[Theta_3]*exp[i*Phi_1+i*Phi_2-i*Phi_3]
            u22 = cos[Theta_1]*cos[Theta_3]*exp[i*Phi_2]
            u23 = -cos[Theta_2]*sin[Theta_3]*exp[-i*Phi_1 - i*Phi_5] -
                    sin[Theta_1]*sin[Theta_2]*cos[Theta_3]*exp[i*Phi_2 - i*Phi_3 + i*Phi_4]
            u31 = -sin[Theta_1]*cos[Theta_2]*sin[Theta_3]*exp[i*Phi_1 - i*Phi_3 + i*Phi_5]
                    - sin[Theta_2]*cos[Theta_3]*exp[-i*Phi_2-i*Phi_4]
            u32 = cos[Theta_1]*sin[Theta_3]*exp[i*Phi_5]
            u33 = cos[Theta_2]*cos[Theta_3]*exp[-i*Phi_1 - i*Phi_2] -
                    sin[Theta_1]*sin[Theta_2]*sin[Theta_3]*exp[-i*Phi_3 + i*Phi_4 + i*Phi_5]


    """

    # construct unitary, element by element
    u11 = np.cos(Theta_1)*np.cos(Theta_2)*np.exp(1j*Phi_1)
    u12 = np.sin(Theta_1)*np.exp(1j*Phi_3)
    u13 = np.cos(Theta_1)*np.sin(Theta_2)*np.exp(1j*Phi_4)
    u21 = np.sin(Theta_2)*np.sin(Theta_3)*np.exp(-1j*Phi_4 - 1j*Phi_5) - np.sin(Theta_1)*np.cos(Theta_2)*np.cos(Theta_3)*np.exp(1j*Phi_1+1j*Phi_2-1j*Phi_3)
    u22 = np.cos(Theta_1)*np.cos(Theta_3)*np.exp(1j*Phi_2)
    u23 = -1*np.cos(Theta_2)*np.sin(Theta_3)*np.exp(-1j*Phi_1 - 1j*Phi_5) - np.sin(Theta_1)*np.sin(Theta_2)*np.cos(Theta_3)*np.exp(1j*Phi_2 - 1j*Phi_3 + 1j*Phi_4)
    u31 = -1*np.sin(Theta_1)*np.cos(Theta_2)*np.sin(Theta_3)*np.exp(1j*Phi_1 - 1j*Phi_3 + 1j*Phi_5) - np.sin(Theta_2)*np.cos(Theta_3)*np.exp(-1j*Phi_2-1j*Phi_4)
    u32 = np.cos(Theta_1)*np.sin(Theta_3)*np.exp(1j*Phi_5)
    u33 = np.cos(Theta_2)*np.cos(Theta_3)*np.exp(-1j*Phi_1 - 1j*Phi_2) - np.sin(Theta_1)*np.sin(Theta_2)*np.sin(Theta_3)*np.exp(-1j*Phi_3 + 1j*Phi_4 + 1j*Phi_5)

    evaluated_unitary = np.matrix([[u11, u12, u13], [u21, u22, u23], [u31, u32, u33]])

    return evaluated_unitary

def Q1_unitary(x):
    """Create an arbitrary single qubit unitary using the z(x[0])* X90 * z(x[1])* X90 z(x[2]) decomposition"""
    return reduce(np.dot, [rot_z(x[0]), rot_x(np.pi/2), rot_z(np.pi + x[1]), rot_x(np.pi/2), rot_z(x[2] - np.pi)])

def upscale(U, q_num=0, n_new_qudits=1, dims=2):
    return np.array(tensor(*[np.eye(dims**(q_num)), U, np.eye(dims**(n_new_qudits - q_num))]))

def bit_to_trit(m, ind=0):
    out = np.eye(3, dtype='complex')
    out[ind:ind+2, ind:ind+2] = m
    return out

def tensor(*t_list):
    """accepts a list of matricies, and tensor products them in order.
       No size checks are done, this can result in very large matricies."""
    result = np.matrix([1], dtype='complex')
    for t in t_list:
        result = np.kron(result, t)
    return result


def basis(dim=2, initial_state=0):
    """Return the density matrix of a system in a pure state"""
    assert initial_state < dim
    assert initial_state >= 0
    dm = np.matrix(np.zeros([dim, dim], dtype='complex'))
    dm[initial_state, initial_state] = 1.

    return dm

def vec_to_dm(vec):
    """Perform the outer product on a vector and return a density matrix"""
    if 1 in np.shape(vec) or len(np.shape(vec))==1:
        if type(vec) == list:
            dm = np.dot(np.array(vec).reshape(-1, 1), np.conjugate(np.array(vec).reshape(1, -1)))
            return dm    
        else:
            dm = np.dot(vec.reshape(-1, 1), np.conjugate(vec.reshape(1, -1)))
            return dm
    return vec


def trace_dist(A, B):
    '''Trace distance as defined by wikipedia'''
    diff = np.matrix(A-B)
    return trace(sp.linalg.sqrtm(diff.H*diff)) * 0.5


def I(n=2):
    return np.eye(n, dtype='complex')


def trace(x, real=True, rounding=6):
    result = np.round(np.trace(x), rounding)
    if real:
        result = np.float(np.real(result))
    return result


def random_S_u_rotation(u):
    """Creates a random unitary matrix of shape, u x u
        -accepts:
            u - integer > 0
        -returns:
            complex unitary matrix """
    # Taken from:
    # http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
    # the construction of a random unitary matrix under eq 35 in paper
    M = np.random.rand(u, u) + 1j * np.random.rand(u, u)
    Q, R = np.linalg.qr(M, 'complete')
    r = np.diag(R)
    L = np.diag(r / np.abs(r))

    # matrix cast so that we can do easy multiplication in numpy
    return np.matrix(np.dot(Q, L))


def random_hamiltonian(eigenvalues):
    """Returns a complex matrix with eigenvalues specified"""
    e_vecs = random_S_u_rotation(len(eigenvalues))

    return np.array(np.dot(np.dot(e_vecs, np.diag(eigenvalues)), e_vecs.T.conj())), e_vecs


def is_hermitian(A):
    n_close = np.sum(np.isclose(A, np.matrix(A).H))
    return n_close == np.product(np.shape(A))


def purity(dm):
    dm = np.matrix(vec_to_dm(dm))
    return trace(dm*dm)


def concurrence(dm):
    
    dm = np.matrix(vec_to_dm(dm))
    dm = dm / np.linalg.norm(dm)

    rho_hat = dm * tensor(sigmay, sigmay) * dm.conjugate() * tensor(sigmay, sigmay)
    
    e_vals = np.abs(sorted(np.real(np.linalg.eigvals(rho_hat))))
    e_vals = list(map(np.sqrt, e_vals))[::-1]
    print(e_vals[0] - np.sum(e_vals[1:]))
    return max(0, e_vals[0] - np.sum(e_vals[1:]))


def entropy(dm):
    """Calculate the von neuumann entropy"""
    dm = np.matrix(vec_to_dm(dm), dtype='complex')

    e_vals = np.linalg.eigvals(dm)
    e_vals = e_vals[np.logical_not(np.isclose(e_vals, 0))]

    return float(np.real(np.sum(-e_vals * np.log2(e_vals))))


def fidelity(M, U):
    """Calculate the fidelity between two density matricies as defined in
    https://journals.aps.org/pra/pdf/10.1103/PhysRevA.71.062310"""
    sqrt_m = np.matrix(sp.linalg.sqrtm(M))
    return trace(sp.linalg.sqrtm(np.matmul(sqrt_m, np.matmul(U, sqrt_m))))**2


def measure(dm, operator, **kwargs):
    '''Perform a projective measurement on a density matrix dm with operator.'''
    ops, e_vals = meas_decomp(operator)

    if not np.allclose(np.sum([a for a in ops], 0), I(len(ops[0]))) and 'force' not in kwargs:
        raise Exception("Projectors do not sum to the identity matrix.")

    probabilities = []
    for op in ops:
        probabilities.append(trace(op * vec_to_dm(dm)))

    draw = np.random.rand()
    bins = np.cumsum(probabilities)

    measured = np.digitize(draw, bins)  # identify which measurement has happened

    final_dm = ops[measured] * vec_to_dm(dm) * ops[measured].H
    final_dm /= probabilities[measured]
    e_vals = np.real(e_vals)
    return final_dm, e_vals[measured], probabilities, measured


def meas_decomp(meas):
    """Decompose a measurement into its eigenvalues and projectors for the eigenvalues"""
    vals, vecs = np.linalg.eig(meas)

    projectors = []
    e_vals = []
    for val in np.unique(vals):
        out = np.zeros_like(meas)
        out[:, vals == val] = vecs[:, vals == val]
        projectors.append(out*np.conjugate(out.T))
        e_vals.append(val)

    return projectors, e_vals


def plot_dm(dm, display='Full', show_colorbar=False, show_ticks=True, x_labels=None, y_labels=None, axes=None, title=None,scale=None, text_color='Grey'):
    dm = vec_to_dm(dm)

    x_dim = np.shape(dm)[1]
    y_dim = np.shape(dm)[0]

    # check if the dimensions are power of three and if so, update labels accordingly
    try:
        n_qutrits = [3, 9, 81, 243, 729, 2187, 6561].index(x_dim) + 1
        if n_qutrits > 2:
            print("FIX LABELS IN SimTools.plot_dm")
        #TODO these labels are wrong for 3 qutrits.  If this ever comes up, fix
        x_labels = [str(i/3)+str(i%3) for i in range(3**n_qutrits)]
        y_labels = [str(i/3)+str(i%3) for i in range(3**n_qutrits)]
    except:
        pass

    if scale is None:
        scale = np.max([x_dim/1.-1, 1.3])

    if axes is None:
        fig = plt.figure(figsize=(scale, scale))

    display = display.lower()

    if display == 'real':
        dm_show = np.real(dm)
        display = "Real Component"
    elif display == 'full':
        dm_show = dm
        display = "Full"
    elif display == 'imag' or display == 'complex':
        dm_show = np.imag(dm)
        display = "Imaginary Component"
    else:
        display = 'Magnitude'
        dm_show = np.abs(dm)
    if axes:
        im = axes.imshow(np.real(dm_show), interpolation='none', cmap='RdBu', vmin=-1, vmax=1)
    else:
        im = plt.imshow(np.real(dm_show), interpolation='none', cmap='RdBu', vmin=-1, vmax=1)
    small_font = FontProperties()#size='small')

    if text_color is None:
        text_color = plt.get_cmap("RdBu")(0.5)

    for x in range(dm.shape[1]):
        for y in range(dm.shape[0]):
            if np.abs(np.round(dm_show[y, x], 2)) > 0:
                val = np.round(dm_show[y, x], 2)
                if not np.isclose(np.imag(val), 0):
                    val_str = "{}\n{}".format(np.real(val), np.imag(val))
                else:
                    val_str = "{}".format(np.real(val))
                if axes:
                    axes.text(x, y, val_str, ha='center', va='center', color=text_color, fontproperties=small_font )
                else:
                    plt.text(x, y, val_str, ha='center', va='center', color=text_color, fontproperties=small_font )

    if show_ticks:
        if x_labels is None:
            fmt_str = "{0:0" + str(int(np.log2(dm.shape[1]))) + "b}"
            plt.xticks(np.arange(dm.shape[1]),
                       [fmt_str.format(x) for x in np.arange(dm.shape[1])],
                       rotation='vertical',
                       fontproperties=small_font)
        else:
            plt.xticks(np.arange(dm.shape[1]),
                       x_labels,
                       fontproperties=small_font)

        if y_labels is None:
            fmt_str = "{0:0" + str(int(np.log2(dm.shape[0]))) + "b}"
            plt.yticks(np.arange(dm.shape[0]),
                       [fmt_str.format(x) for x in np.arange(dm.shape[0])],
                       fontproperties=small_font)
        else:
            plt.yticks(np.arange(dm.shape[0]),
                       y_labels,
                       fontproperties=small_font)
    if title:
        plt.title(title)
    else:
        plt.xticks([],[])
        plt.yticks([],[])
    if show_colorbar:
        plt.colorbar(im, fraction=0.03, pad=0.04)

    if not axes:
        return fig
    else:
        return axes

def partial_trace(dm, k=1, dim=None):
    """This was largely taken from https://github.com/gsagnol/picos with minor changes"""
    sz = dm.shape
    if dim is None:
        if sz[0] == sz[1] and (sz[0] ** 0.5) == int(sz[0] ** 0.5) and (sz[1] ** 0.5) == int(sz[1] ** 0.5):
            dim = (int(sz[0] ** 0.5), int(sz[1] ** 0.5))
        else:
            raise ValueError('The default parameter dim=None assumes X is a n**2 x n**2 matrix')

    # checks if dim is a list (or tuple) of lists (or tuples) of two integers each
    T = [list, tuple]
    if type(dim) in T and all([type(d) in T and len(d) == 2 for d in dim]) and all(
            [type(n) is int for d in dim for n in d]):
        dim = [d for d in zip(*dim)]
        pdim = np.product(dim[0]), np.product(dim[1])

    # if dim is a single list of integers we assume that no subsystem is rectangular
    elif type(dim) in [list, tuple] and all([type(n) is int for n in dim]):
        pdim = np.product(dim), np.product(dim)
        dim = (dim, dim)
    else:
        raise ValueError('Wrong dim variable')

    if len(dim[0]) != len(dim[1]):
        raise ValueError('Inconsistent number of subsystems, fix dim variable')

    if pdim[0] != sz[0] or pdim[1] != sz[1]:
        print(pdim, sz)
        raise ValueError('The product of the sub-dimensions does not match the size of X')

    if k > len(dim[0]) - 1:
        raise Exception('There is no k-th subsystem, fix k or dim variable')

    if dim[0][k] != dim[1][k]:
        raise ValueError('The dimensions of the subsystem to trace over don\'t match')

    dim_reduced = [list(d) for d in dim]
    del dim_reduced[0][k]
    del dim_reduced[1][k]
    dim_reduced = tuple(tuple(d) for d in dim_reduced)
    pdimred = tuple([np.product(d) for d in dim_reduced])

    fact = np.zeros((np.product(pdimred), np.product(pdim)), dtype='complex')

    for iii in itertools.product(*[range(i) for i in dim_reduced[0]]):
        for jjj in itertools.product(*[range(j) for j in dim_reduced[1]]):
            # element iii,jjj of the partial trace

            row = int(sum([iii[j] * np.product(dim_reduced[0][j + 1:]) for j in range(len(dim_reduced[0]))]))
            col = int(sum([jjj[j] * np.product(dim_reduced[1][j + 1:]) for j in range(len(dim_reduced[1]))]))
            # this corresponds to the element row,col in the matrix basis
            rowij = col * pdimred[0] + row
            # this corresponds to the elem rowij in vectorized form

            # computes the partial trace for iii,jjj
            for l in range(dim[0][k]):
                iili = list(iii)
                iili.insert(k, l)
                iili = tuple(iili)

                jjlj = list(jjj)
                jjlj.insert(k, l)
                jjlj = tuple(jjlj)

                row_l = int(sum([iili[j] * np.product(dim[0][j + 1:]) for j in range(len(dim[0]))]))
                col_l = int(sum([jjlj[j] * np.product(dim[1][j + 1:]) for j in range(len(dim[1]))]))

                colij_l = col_l * pdim[0] + row_l
                fact[int(rowij), int(colij_l)] = 1

    return np.dot(dm.reshape(-1), fact.T).reshape(pdimred[0], pdimred[1])


def remove_phase(array):
    phase = np.angle(array[..., 0, 0]+0)
    where_zero = np.where(np.around(array[..., 0, 0], 5) == 0)

    if len(array.shape) > 2:
        phase[where_zero] = np.array(np.angle(array[..., 1, 0]+0))[where_zero]
    elif len(where_zero[0]) == 1:
        phase = np.angle(array[1, 0]+0)

    array *= np.exp(-1j*phase)[..., np.newaxis, np.newaxis]
    return array


def is_ident(array):
    n_dims = np.shape(array)[-1]
    remove_phase(array)
    return np.isclose(np.linalg.norm(array - np.eye(n_dims), axis=(-2, -1)), 0)


def decompose_to_paulis(target):
    """Decomposes a matrix into pauli matricies, returns a dictionary of the paulis and amplitudes.
    Additionally returns a list of the paulis not present."""

    n_qubits = int(target.shape[0]**0.5)
    flattened_paulis = []


    pauli_name_combinations = list(product(paulis, repeat=n_qubits))
    for combo in pauli_name_combinations:
        flattened_paulis.append(np.array(reduce(np.kron, [paulis[x] for x in combo])).reshape(-1).astype(complex))

    flattened_paulis = np.transpose(flattened_paulis)
    target_parts = np.linalg.solve(flattened_paulis, np.array(target).reshape(-1))
    target_parts = np.around(target_parts, 14) + 0

    error  = np.dot(flattened_paulis, target_parts).reshape(2**n_qubits, -1) - target

    if not np.isclose(np.linalg.norm(error), 0, atol=1e-3):
        print("FIT FAILED!")
        print(error)

    found_paulis = [(combo, t) for combo, t in zip(pauli_name_combinations, target_parts) if not np.isclose(t, 0)]
    missing_paulis = [combo for combo, t in zip(pauli_name_combinations, target_parts) if np.isclose(t, 0)]

    return OrderedDict(found_paulis), missing_paulis


def trace_out_qudit(x, qudit=0, n_levels=3):
    """Takes an array of shape M x N, assuming that the M index represents
    binned data such as 000, 001, 002, 010, 011, 012, 020, 021, 022, ... etc
    You can then pass either a single qudit or a list of qudits and it will
    partial trace over the array along the M axis only, summing over the data.
    So in the index example above, when qudit=1, n_levels=3, the data in the bins
    A0B, A1B, A2B will be added together into a new array which is M/3 x N in size
    
    More explicitly:
    000, 010, 020, get added
    001, 011, 021, get added
    002, 012, 022, get added etc.
    Accepts:
        x - array of data of shape M x ...
        qudit - int or list of ints which correspond to the qubit to be traced over
        n_levels - 2 for qubit, 3 for qutrit, etc.
        
    Returns:
        result - np.array of shape M/(n_levels)**len(qudit) x ...
    """
    
    # Recursive check, if qudit is an array of qubits preform the operation
    # recursively, 1 qubit at a time.
    if isinstance(qudit, list) or isinstance(qudit, np.ndarray):
        qudit = np.array(sorted(qudit))
        x = trace_out_qudit(x, qudit[0], n_levels)
        if len(qudit) == 1:
            return x
        else:
            qudit -= 1
            return trace_out_qudit(x, qudit[1:], n_levels)

    x = np.array(x)
    
    n_qudits = int(np.log(x.shape[0]) / np.log(n_levels))
    
    # Prepare an empty array which we will add the data to
    result = np.zeros([n_levels**(n_qudits-1), x.shape[1]])
    
    # product will enumerate over all of all combinations IE: 000, 001, 002...
    for old_index, combo in enumerate(product(range(n_levels), repeat=n_qudits)):
        # Throw away the qubit in the correct index, IE:
        # if combo=(0,2,1), qudit=1, we want reduce_combo=(0,1), having thrown out the 2
        reduce_combo = list(combo[:qudit])
        reduce_combo.extend(list(combo[qudit+1:]))
        
        # we then compute this value in the n_levels number base, this is the new index
        new_index = sum([p*n_levels**c for c, p in enumerate(reduce_combo[::-1])])
        
        result[new_index] += x[old_index]
        
    return result


def remove_I(U):
    """Removes the I, I, I etc pauli combination from a gate, effectively a phase removal tool"""
    U = np.array(U)
    n_qubits = int(np.log2(U.shape[0]))

    U_paulis, _ = decompose_to_paulis(sp.linalg.logm(U))

    # if the U_paulis matrix was empty, sp.linalg.logm must have returned a zero matrix
    # which means the input matrix was the identity
    if len(U_paulis) == 0:
        return np.eye(U.shape[0])

    if tuple(n_qubits*['I']) in U_paulis:
        del U_paulis[tuple(n_qubits*['I'])]

    new_U = gate_from_paulis(U_paulis)

    return np.around(sp.linalg.expm(new_U), 14) + 0


def gate_from_paulis(pauli_dict):
    """Take a dictionary as provided by decompose into paulis and reconstruct the original gate"""
    n_qubits = len(list(pauli_dict.keys())[0])

    # Reconstruct the damn thing
    pauli_name_combinations = list(product(paulis, repeat=n_qubits))

    new_U = np.zeros([2**n_qubits, 2**n_qubits], dtype=complex)
    for combo in pauli_dict:
        new_U += reduce(np.kron, [paulis[x] for x in combo]) * pauli_dict[combo]

    return np.around(new_U, 14) + 0
