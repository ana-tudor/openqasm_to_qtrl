# Copyright (c) 2018-2019, UC Regents

from .base import *
import numpy as np
from scipy.linalg import expm

__all__ = ['cnot', 'swap', 'rot_x', 'rot_y', 'rot_z']

cnot = np.matrix([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]],
                 dtype='complex')

CZ = np.matrix([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, -1]],
                 dtype='complex')

sqrt_not = np.matrix([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, np.sqrt(2)/2, -np.sqrt(2)/2],
                  [0, 0, np.sqrt(2)/2, np.sqrt(2)/2]],
                 dtype='complex')

swap = np.matrix([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]],
                 dtype='complex')


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


X90 = rot_x(np.pi/2)
Y90 = rot_y(np.pi/2)
Z90 = rot_z(np.pi/2)
X180 = rot_x(np.pi)
Y180 = rot_y(np.pi)
Z180 = rot_z(np.pi)
X270 = rot_x(3*np.pi/2)
Y270 = rot_y(3*np.pi/2)
Z270 = rot_z(3*np.pi/2)

computational_set = {'I': I(),
                     'X90': X90,
                     'Z90': Z90,
                     'Z180': Z180,
                     'Z270': Z270,}


minimum_set = {'I': I(),
               'X90': X90,
               'Z90': Z90,}


pauli_set = {'I': I(),
             'X180': X180,
             'Y180': Y180,
             'Z180': Z180,}


full_set = {'I': I(),
            'X180': X180,
            'Y180': Y180,
            'Z180': Z180,
            'X90': X90,
            'Y90': Y90,
            'Z90': Z90,
            'X270': X270,
            'Y270': Y270,
            'Z270': Z270,
            }


full_set_no_Z = {'I': I(),
                 'X180': X180,
                 'Y180': Y180,
                 'X90': X90,
                 'Y90': Y90,
                 'X270': X270,
                 'Y270': Y270,}

# cliffords but only using a single X90/180/270 or single Y90/180/270 and 1 z gate
cliffords = {('I', 'I'): I(),
             ('I', 'X90'): X90,
             ('I', 'Y90'): Y90,
             ('I', 'X180'): X180,
             ('I', 'Y180'): Y180,
             ('I', 'X270'): X270,
             ('I', 'Y270'): Y270,
             ('I', 'Z90'): Z90,
             ('X90', 'Z90'): X90*Z90,
             ('Y90', 'Z90'): Y90*Z90,
             ('X180', 'Z90'): X180*Z90,
             ('Y180', 'Z90'): Y180*Z90,
             ('X270', 'Z90'): X270*Z90,
             ('Y270', 'Z90'): Y270*Z90,
             ('I', 'Z180'): Z180,
             ('X90', 'Z180'): X90*Z180,
             ('Y90', 'Z180'): Y90*Z180,
             ('X270', 'Z180'): X270*Z180,
             ('Y270', 'Z180'): Y270*Z180,
             ('I', 'Z270'): Z270,
             ('X90', 'Z270'): X90*Z270,
             ('Y90', 'Z270'): Y90*Z270,
             ('X270', 'Z270'): X270*Z270,
             ('Y270', 'Z270'): Y270*Z270,
             }

# Cliffords but decomposed using 2 X90 gates always with 0-3 z gates
cliffords_X90 = {('I', 'I', 'X90', 'X90', 'Z180'): X90*X90*Z180,
                 ('I', 'I', 'I', 'X90', 'X90'): X90*X90,
                 ('Z270', 'X90', 'Z90', 'X90', 'Z270'): Z270*X90*Z90*X90*Z270,
                 ('I', 'X90', 'Z180', 'X90', 'Z90'): X90*Z180*X90*Z90,
                 ('I', 'X90', 'Z270', 'X90', 'Z180'): X90*Z270*X90*Z180,
                 ('I', 'Z90', 'X90', 'Z270', 'X90'): Z90*X90*Z270*X90,
                 ('I', 'I', 'X90', 'Z270', 'X90'): X90*Z270*X90,
                 ('I', 'X90', 'Z270', 'X90', 'Z90'): X90*Z270*X90*Z90,
                 ('I', 'X90', 'Z90', 'X90', 'Z180'): X90*Z90*X90*Z180,
                 ('I', 'Z270', 'X90', 'Z270', 'X90'): Z270*X90*Z270*X90,
                 ('I', 'X90', 'Z90', 'X90', 'Z90'): X90*Z90*X90*Z90,
                 ('Z270', 'X90', 'Z270', 'X90', 'Z270'): Z270*X90*Z270*X90*Z270,
                 ('Z270', 'X90', 'Z270', 'X90', 'Z90'): Z270*X90*Z270*X90*Z90,
                 ('I', 'X90', 'Z90', 'X90', 'Z270'): X90*Z90*X90*Z270,
                 ('I', 'I', 'X90', 'Z180', 'X90'): X90*Z180*X90,
                 ('Z270', 'X90', 'Z90', 'X90', 'Z90'): Z270*X90*Z90*X90*Z90,
                 ('I', 'X90', 'Z180', 'X90', 'Z180'): X90*Z180*X90*Z180,
                 ('I', 'I', 'X90', 'X90', 'Z90'): X90*X90*Z90,
                 ('I', 'Z270', 'X90', 'Z90', 'X90'): Z270*X90*Z90*X90,
                 ('I', 'I', 'X90', 'X90', 'Z270'): X90*X90*Z270,
                 ('I', 'Z90', 'X90', 'Z90', 'X90'): Z90*X90*Z90*X90,
                 ('I', 'X90', 'Z180', 'X90', 'Z270'): X90*Z180*X90*Z270,
                 ('I', 'X90', 'Z270', 'X90', 'Z270'): X90*Z270*X90*Z270,
                 ('I', 'I', 'X90', 'Z90', 'X90'): X90*Z90*X90,
                 }

phase_gate = np.matrix([[1, 0], [0, 1j]])

hadamard = cliffords[('Y90', 'Z180')] * 1j

ZX_gate = expm(-np.kron(sigmaz, sigmax)*1j*np.pi/4)*np.exp(1j*np.pi/2)
