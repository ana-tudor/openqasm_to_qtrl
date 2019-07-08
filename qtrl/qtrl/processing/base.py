# Copyright (c) 2018-2019, UC Regents

# This import is needed for the eval process, do no remove
# the eval process needs access to the qtrl._DAC and qtrl._ADC objects
import qtrl


class ADCProcess:

    def prep(self):
        pass

    def batch_start(self):
        pass

    def batch_end(self, measurement):
        pass

    def post(self, measurement, seq=None):
        pass


class Eval(ADCProcess):
    def __init__(self, prep='None', batch_start='None', batch_end='None', post='None'):
        self._prep = prep
        self._batch_start = batch_start
        self._batch_end = batch_end
        self._post = post

    def prep(self):
        eval(self._prep, globals())

    def batch_start(self):
        eval(self._batch_start, globals())

    def batch_end(self, measurement):
        eval(self._batch_end, globals())

    def post(self, measurement, seq=None):
        eval(self._post, globals())


def find_resonator_names(measurement):
    """estimate the number of resonators in the measurement"""
    res_names = []
    for k in measurement.keys():
        if 'R' in k:
            res_names.append(k)
    return res_names
