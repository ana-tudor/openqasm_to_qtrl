# Copyright (c) 2018-2019, UC Regents

import numpy as np
from collections import OrderedDict

__all__ = ['sigmax', 'sigmay', 'sigmaz', 'paulis', 'I']

sigmax = np.matrix([[0, 1], [1, 0]], dtype='complex')
sigmay = np.matrix([[0, -1j], [1j, 0]], dtype='complex')
sigmaz = np.matrix([[1, 0], [0, -1]], dtype='complex')

paulis = OrderedDict([('I', np.matrix(np.eye(2, dtype=complex))),
                      ('X', sigmax),
                      ('Y', sigmay),
                      ('Z', sigmaz)])


def I(n=2):
    """Identity matrix of size n"""
    return np.eye(n, dtype='complex')
