# Copyright (c) 2018-2019, UC Regents

import numpy as np
from .base import ADCProcess, find_resonator_names
from ..analysis.sim_tools import tensor


class IndividualReadoutCorrection(ADCProcess):

    def __init__(self, corr_matrix, input_name, result_name):

        n_levels = np.sqrt(len(corr_matrix[0]))
        n_res = len(corr_matrix)

        assert n_res != 0, 'No correction matrix specified'
        assert n_levels == int(n_levels), "Correction matrix is not square, fix it."
        n_levels = int(n_levels)

        self._correction_tensor = np.reshape(corr_matrix, [n_res, n_levels, n_levels])
        self._input_name = input_name
        self._result_name = result_name
        self._n_levels = n_levels

    def post(self, measurement, seq=None):

        res_names = find_resonator_names(measurement)

        for i, res in enumerate(res_names):
            res_dat = measurement[res][self._input_name]
            classified_dat = np.mean([res_dat == x for x in range(self._n_levels)], 1)

            corrected = np.linalg.solve(self._correction_tensor[i], classified_dat)

            measurement[res][self._result_name] = corrected


class JointReadoutCorrection(ADCProcess):

    def __init__(self, corr_matrix, resonators, input_name, result_name):
        n_levels = np.sqrt(len(corr_matrix[0]))
        n_res = len(corr_matrix)

        assert n_res != 0, 'No correction matrix specified'
        assert n_levels == int(n_levels), "Correction matrix is not square, fix it."
        n_levels = int(n_levels)

        self._correction_tensor = np.reshape(corr_matrix, [n_res, n_levels, n_levels])
        self._input_name = input_name
        self._resonators = resonators
        self._result_name = result_name
        self._n_levels = n_levels

    def post(self, measurement, seq=None):
        res_names = find_resonator_names(measurement)

        res_dat = measurement['joint'][self._input_name]

        res_indices = [np.argwhere(i == np.array(res_names))[0, 0] for i in self._resonators][::-1]

        local_cor_matrix = np.real(tensor(*self._correction_tensor[res_indices]))

        corrected = np.linalg.solve(local_cor_matrix, res_dat)

        measurement['joint'][self._result_name] = corrected
