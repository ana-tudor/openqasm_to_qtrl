# Copyright (c) 2018-2019, UC Regents

import numpy as np
from scipy.special import erf
from .base import ADCProcess, find_resonator_names
from sklearn import mixture
from ..analysis.state_tomography import binary_arrays_to_bins
"""This is for the classification and post selection of data,
things like gaussian mixture model, and heralding should go here"""


class IQRotation(ADCProcess):
    def __init__(self, angles, result_name='Heterodyne', input_name='Heterodyne'):
        """This assumes data is of the shape [2, ...]"""
        self._angles = angles
        self._result_name = result_name
        self._input_name = input_name

    def post(self, measurement, seq=None):
        res_names = find_resonator_names(measurement)
        assert len(res_names) == len(self._angles), ('The number of resonators {} doesnt match the number of provided '
                                                     'angles {}'.format(len(res_names), len(self._angles)))

        for i, res in enumerate(res_names):
            c = np.cos(self._angles[i])
            s = np.sin(self._angles[i])
            rot_matrix = [[c, -s], [s, c]]
            measurement[res][self._result_name] = np.einsum('ij,j...->i...', rot_matrix, measurement[res][self._input_name])


class GMM(ADCProcess):
    def __init__(self, means, covariances, result_name='GMM', input_name='Heterodyne'):
        """Means should be a matrix of size Nx(2*n_gaussians)
        covariances should be a vector of Nx(n_gaussians)"""
        means = np.array(means)
        covariances = np.array(covariances)
        self._mixes = []
        self._result_name = result_name
        self._input_name = input_name
        n_gaussians = len(covariances[0])
        for i in range(len(covariances)):
            mix = mixture.GaussianMixture(n_components=len(covariances[0]),
                                          covariance_type='spherical')
            mix.means_ = means[i].reshape(n_gaussians, 2)
            mix.covariances_ = covariances[i]
            mix.precisions_cholesky_ = 0.01
            mix.weights_ = np.ones(n_gaussians) / n_gaussians
            self._mixes.append(mix)

    def post(self, measurement, seq=None):
        res_names = find_resonator_names(measurement)

        for i, res in enumerate(res_names):
            tmp_dat = measurement[res][self._input_name]
            original_shape = tmp_dat.shape[1:]

            measurement[res][self._result_name] = self._mixes[i].predict(tmp_dat.reshape(2, -1).T).reshape(original_shape)

            # add separation fidelity
            means = np.reshape(self._mixes[i].means_,
                               [-1, 2]).T # [mean_0, mean_1]

            variances = np.reshape(self._mixes[i].covariances_,
                              [-1, 2]).T
            separation = np.sqrt(np.sum((np.diff(means)) ** 2.0) / np.sqrt(np.product(variances)))
            measurement[res]['GMM_separation'] = {'Separation': separation,
                                                  'Separation_Fidelity': 0.5 + erf(np.sqrt(separation/8))/2.0,
                                                  'Separation_Absolute': np.sqrt(np.sum((np.diff(means)) ** 2.0))}

class Heralding(ADCProcess):
    def __init__(self, herald_classified_name, input_name, result_name, target_state=0, rep_index=1, elem_index=0):
        """This selects data which passes the heralding test.
        Accepts:
            herald_classified_name - This is the name of the data which is used to deterrmine if
                                     a measurement has passed a herald test
            input_name - this is the data which will be down selected using a mask generated
                         from the herald_classified_name
            result_name - this is the name of where the data will be stored
            target_state - this is what state we are after in the herald test
        """
        self._herald_classified_name = herald_classified_name
        self._input_name = input_name
        self._result_name = result_name
        self._target_state = target_state
        self.rep_index = rep_index
        self.elem_index = elem_index
    def post(self, measurement, seq=None):

        res_names = find_resonator_names(measurement)
        herald_mask = np.zeros_like(measurement[res_names[0]][self._herald_classified_name])

        for res in res_names:
            herald_mask = herald_mask | measurement[res][self._herald_classified_name]

        n_pass_herald = np.min(np.sum(herald_mask == self._target_state, 0))
        if n_pass_herald == 0:
            print("No measurements passed herald!")

        # since not all repetitions will pass the same herald, IE: qubit 7 might be 0 but 6 1,
        # we need to not only make sure we create a mask which only has all 00000 states as well
        # as make sure that the total number of heralds in each sequence element is the same length
        for i in range(herald_mask.shape[1]):
            j = 0
            while np.sum(herald_mask[:, i] == self._target_state) > n_pass_herald:
                n_short = np.sum(herald_mask[:, i] == self._target_state) - n_pass_herald
                herald_mask[j:j + n_short, i] = -1
                j += n_short

        # Create our final mask
        herald_mask = herald_mask == self._target_state

        # down select using the mask.

        input_shape = np.shape(measurement[res][self._input_name])
        new_dims = np.arange(len(input_shape))

        rep_index = self.rep_index
        elem_index = self.elem_index
        # puts input array into (reps, elements, ...) and keeps it in this form if it already is
        new_dims[1], new_dims[rep_index] = new_dims[rep_index], new_dims[1]
        new_dims[0], new_dims[elem_index] = new_dims[elem_index], new_dims[0]

        if rep_index == 1 and elem_index == 0:
            new_dims = new_dims[::-1] #normal transpose
        for res in res_names:
            new_shape = list(measurement[res][self._input_name].transpose(*new_dims).shape)
            new_shape[1] = n_pass_herald
            measurement[res][self._result_name] = measurement[res][self._input_name].transpose(*new_dims)[herald_mask.T].reshape(*new_shape).transpose(*new_dims)

class CorrelatedBins(ADCProcess):
    def __init__(self, input_name, result_name, resonators=None):
        """Bin the correlated measurements into bins:
            Example qubit 5 reads 0, 0, 1
                    qubit 6 reads 0, 1, 1
            this calculates the bins counts 00, 01, 10, 11,
            then returns [1, 1, 0, 1], the counts in each bin.
            Automatically scales as the number of qubits increase.

            Accepts input_name - name of the classified data as input,
                    result_name - name of the binned results, put in the 'joint' key of the measurement
                    resonators - (optional) which resonators will be used in the binning, if not provided
                            will bin all the resonator data. IE: ['R5', 'R6', 'R7']
        """

        self._input_name = input_name
        self._result_name = result_name
        self._resonators = resonators

    def post(self, measurement, seq=None):
        if self._resonators is None:
            self._resonators = find_resonator_names(measurement)

        n_qubits = len(self._resonators)

        unbinned_data = binary_arrays_to_bins([measurement[name][self._input_name] for name in self._resonators[::-1]])

        binned_data = np.array([np.mean(unbinned_data == n, 0) for n in range(2**n_qubits)])

        measurement['joint'][self._result_name] = binned_data

        # here is a way of selecting certain qubit states
        # mask = [int(f"{n:0{n_qubits}b}"[0]) == 1 for n in range(2**n_qubits)]

