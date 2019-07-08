# Copyright (c) 2018-2019, UC Regents

from .base import ADCProcess, find_resonator_names
import numpy as np
"""This is intended for data organization, not processing
Things like saving data, or relabeling things without actually
changing the data itself"""


class AxisTriggerSelector(ADCProcess):
    """This allows you to select a specific axis of IQ data, and/or a specific trigger
    from the data."""
    def __init__(self, axis, trigger, result_name='AxisTrigger', input_name='Heterodyne'):
        if axis is None:
            axis = slice(0, None)
        if trigger is None:
            trigger = slice(0, None)

        self._result_name = result_name
        self._input_name = input_name
        self._axis = axis
        self._trigger = trigger

    def post(self, measurement, seq=None):
        res_names = find_resonator_names(measurement)

        for res in res_names:
            measurement[res][self._result_name] = measurement[res][self._input_name][self._axis, ..., self._trigger]


class AverageAxis(ADCProcess):
    """Average the data over a specific axis"""
    def __init__(self, axis, result_name, input_name):
        """Specify which axis to average over, along with input and output"""
        self._axis = axis
        self._result_name = result_name
        self._input_name = input_name

    def post(self, meas, seq=None):
        res_names = find_resonator_names(meas)

        for res in res_names:
            meas[res][self._result_name] = np.mean(meas[res][self._input_name], self._axis)
