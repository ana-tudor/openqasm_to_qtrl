# Copyright (c) 2018-2019, UC Regents

from .base import ADCProcess
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# TODO: does not import, but it's only needed for set_num_threads()?
try:
    import mkl
    mkl_get_num_threads = mkl.get_num_threads
    mkl_set_num_threads = mkl.set_num_threads
except ImportError:
    import ctypes
    try:
        mkl_rt = ctypes.CDLL('libmkl_rt.so')
    except (OSError, IOError):
        # TODO: may want to use egg_info/RECORD instead as this assumes
        # a normal 'site-packages' installation
        import pkg_resources, os
        dist_path = pkg_resources.get_distribution('mkl').module_path
        fp = os.path.join(dist_path, os.pardir, os.pardir, 'libmkl_rt.so')
        mkl_rt = ctypes.CDLL(fp)
        mkl_get_max_threads = mkl_rt.mkl_get_max_threads
        def mkl_set_num_threads(cores):
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))


def demod(raw_data, demod_sin, demod_cos):
    """Demodulate a frequency multiplexed signal
    # TODO: Clean up this code, it has been re-written many times by many people and needs some love
    """
    demod_data_type = np.float32
    demod_order = 'C'

    # set the number of threads.
    # 4 threads gives the shortest demod time based on tests. demod time slowly increases.
    # TODO: see above, mkl does not import // TODO: is 4 machine dependent?
    mkl_set_num_threads(4)
    if mkl_get_max_threads() == 1:
        'Warning: Intel Math Kernel Library still running in single threaded mode'

    # Raw acquisition has a shape of [channels, repetitions, triggers, elements, acquisition length]
    # demod_exp_ have the shape [frequencies, acquisition length]
    # a, b have the shape [frequencies, channels, repetitions, triggers, elements]

    original_shape = np.shape(raw_data)
    integration_width = original_shape[-1]
    n_demod_freq = np.shape(demod_sin)[1]

    final_shape = list(original_shape)
    final_shape.insert(0, n_demod_freq)
    final_shape = final_shape[:-1]

    # prepare the raw data for the demod. raw data has the shape:
    # [channels, repetitions, triggers,  elements, acquisition length]
    raw_data_ch1 = raw_data[0].reshape([-1, integration_width]).astype(demod_data_type)
    raw_data_ch2 = raw_data[1].reshape([-1, integration_width]).astype(demod_data_type)

    raw_data_ch1 = raw_data_ch1 - np.mean(raw_data_ch1)
    raw_data_ch2 = raw_data_ch2 - np.mean(raw_data_ch2)

    # give a warning if the raw data is not C_CONTIGUOUS because the dot product will be much slower.
    if not raw_data_ch1.flags['C_CONTIGUOUS'] or not raw_data_ch2.flags['C_CONTIGUOUS']:
        print('Warning: raw_data is not C_CONTIGUOUS. np.dot() may be slower')

    # make an empty array for processed data
    # processed_data has the shape [frequencies, channels, repetitions, triggers,  elements]
    processed_data = np.zeros(final_shape[:], dtype=demod_data_type)

    # loop over demod frequency
    for demod_freq_index in range(n_demod_freq):
        # t0 = timeit.default_timer()

        # Demod math #
        # IQ = exp(1j*2*np.pi*freq*time)*(ch1+1j*ch2)
        # I = np.real(IQ), Q = np.imag(IQ)
        # shift to reals to use BLAS:
        # I = cos(2*np.pi*freq*time)*ch1 - sin(2*np.pi*freq*time)*ch2
        # Q = sin(2*np.pi*freq*time)*ch1 + cos(2*np.pi*freq*time)*ch2

        # prepare the demod arrays
        demod_sin_array = np.array(demod_sin[:, demod_freq_index],
                                   dtype=demod_data_type,
                                   order=demod_order)
        demod_cos_array = np.array(demod_cos[:, demod_freq_index],
                                   dtype=demod_data_type,
                                   order=demod_order)

        # demod I
        processed_data_I = np.dot(raw_data_ch1,
                                  demod_cos_array).reshape(final_shape[2:])
        processed_data_I -= np.dot(raw_data_ch2,
                                   demod_sin_array).reshape(final_shape[2:])

        # demod Q
        processed_data_Q = np.dot(raw_data_ch1,
                                  demod_sin_array).reshape(final_shape[2:])
        processed_data_Q += np.dot(raw_data_ch2,
                                   demod_cos_array).reshape(final_shape[2:])

        # save the data for this demod frequency
        processed_data[demod_freq_index, 0] = processed_data_I
        processed_data[demod_freq_index, 1] = processed_data_Q

    # return data normalized to how long each time trace is.
    return processed_data / integration_width


class Heterodyne(ADCProcess):

    def __init__(self, freq_labels, frequencies, sample_rate):
        """This manages heterodyne processing of the signals, it uses the demod
        function defined above along with threadPoolExecutor to manage multi-processing"""
        if freq_labels is None:
            freq_labels = np.arange(len(frequencies))

        assert len(freq_labels) == len(frequencies), "There has to be the same number of labels as frequencies"

        frequencies = np.array(frequencies)
        self._freqs_ind = np.array(frequencies/sample_rate)

        self._freq_labels = freq_labels
        self._demod_cos = None
        self._demod_sin = None
        self.result = None

        self._results = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Heterodyne_Threads")

    def prep(self):
        self._results = []

    def batch_end(self, measurement):
        raw_traces = measurement['raw_trace']

        n_samples = raw_traces.shape[-1]
        time_frequency_product = np.einsum('i,j->ij', np.arange(0, n_samples), self._freqs_ind)

        self._demod_sin = np.sin(2*np.pi*time_frequency_product)
        self._demod_cos = np.cos(2*np.pi*time_frequency_product)

        fut = self._executor.submit(demod, raw_traces, self._demod_sin, self._demod_cos)
        measurement['raw_trace'] = None
        self._results.append(fut)

    def post(self, measurement, seq=None):
        self.result = []
        for r in self._results:
            self.result.append(r.result())

        if len(self.result) == 0:
            return self.result

        self.result = np.concatenate(self.result, 2)

        for label in self._freq_labels:
            if 'R{}'.format(label) not in measurement:
                measurement['R{}'.format(label)] = {}

        for i, label in enumerate(self._freq_labels):
            measurement['R{}'.format(label)]['Heterodyne'] = self.result[i]
        return self.result


class PartialHeterodyne(ADCProcess):

    def __init__(self, freq_labels, frequencies, sample_rate, decimation=None):
        """This manages heterodyne processing of the signals, it uses the demod
        function defined above along with threadPoolExecutor to manage multi-processing"""
        if freq_labels is None:
            freq_labels = np.arange(len(frequencies))

        assert len(freq_labels) == len(frequencies), "There has to be the same number of labels as frequencies"

        frequencies = np.array(frequencies)
        self._freqs_ind = np.array(frequencies/sample_rate)

        self._freq_labels = freq_labels
        self._demod_exp = None
        self._decimation = decimation
        self.result = None

        self._results = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Heterodyne_Threads")

    @staticmethod
    def _partial_demod(demod_tensor, data_tensor, decimation=None):
        # set the number of threads.
        # 4 threads gives the shortest demod time based on tests. demod time slowly increases.
        mkl.set_num_threads(4)
        if mkl.get_max_threads() == 1:
            'Warning: Intel Math Kernel Library still running in single threaded mode'

        # data_tensor has a shape of [IQ, repetitions, triggers, elements, acquisition length]
        # demod_tensor have the shape [frequencies, acquisition length] and is complex in value

        # data_tensor now has a shape of [repetitions, triggers, elements, acquisition length]
        data_tensor = (data_tensor[0] - 1j*data_tensor[1]).astype(np.complex64)

        # Remove the DC offset
        data_tensor = data_tensor.astype(np.complex64) - np.mean(data_tensor).astype(np.complex64)

        # give a warning if the raw data is not C_CONTIGUOUS because the dot product will be much slower.
        if not data_tensor.flags['C_CONTIGUOUS']:
            print('Warning: raw_data is not C_CONTIGUOUS. np.dot() may be slower')

        if decimation is None:
            decimation = 1

        # Need to calculate the shape the tensor has to be before the deecimation
        data_shape = list(np.shape(data_tensor))
        demod_shape = list(np.shape(demod_tensor))

        print(data_shape, demod_shape)

        assert demod_shape[-1] == data_shape[-1], "Demod Tensor and Data do not have the same length"
        assert data_shape[-1] % decimation == 0, "Decimation must be a multiple of acquisition length"

        l_dim = int(data_shape[-1] / decimation)

        # append the new shape on the end, we are adding an index here
        demod_shape[-1] = decimation
        demod_shape.append(l_dim)
        data_shape[-1] = decimation
        data_shape.append(l_dim)

        data_tensor = data_tensor.reshape(data_shape)
        demod_tensor = demod_tensor.reshape(demod_shape)

        print(data_shape, demod_shape)

        # dot product however leave the time series,
        # this should be the same as a kronecker product
        data_tensor = np.einsum('fdl,rtedl->frtel', demod_tensor, data_tensor)

        # now we have to go from the complex valued to real valued by re-adding the index for IQ
        final_shape = list(np.shape(data_tensor))
        final_shape.insert(1, 2)

        final_data = np.zeros(final_shape, dtype=np.float32)
        final_data[:, 0] = data_tensor.real
        final_data[:, 1] = data_tensor.imag

        return final_data

    def prep(self):
        self._results = []

    def batch_end(self, measurement):
        raw_traces = measurement['raw_trace']
        measurement['raw_trace'] = None

        n_samples = raw_traces.shape[-1]
        time_frequency_product = np.einsum('i,j->ij', np.arange(0, n_samples), self._freqs_ind)

        self._demod_exp = np.exp(1j*2*np.pi*time_frequency_product).T

        fut = self._executor.submit(self._partial_demod, self._demod_exp, raw_traces, self._decimation)
        self._results.append(fut)

    def post(self, measurement, seq=None):
        self.result = []
        for r in self._results:
            self.result.append(r.result())

        if len(self.result) == 0:
            return self.result

        self.result = np.concatenate(self.result, 2)

        for label in self._freq_labels:
            if 'R{}'.format(label) not in measurement:
                measurement['R{}'.format(label)] = {}

        for i, label in enumerate(self._freq_labels):
            measurement['R{}'.format(label)]['Heterodyne'] = self.result[i]
        return self.result


def mock_trace(n_elements, n_points, n_repeats, sample_rate=500e6, freqs=[0.e6], noise=0.25):
    """Generates raw trace of a rabi oscillation, which can be used for testing"""

    n_elements = int(n_elements)
    n_points = int(n_points)

    # make an empty array to be filled
    trace = np.zeros([2, n_elements, n_repeats, n_points], dtype=float)

    # for each freq calculate the variation
    for i, freq in enumerate(freqs):
        q_trace = np.exp(1j * np.pi * 2 * freq / sample_rate * np.arange(0, n_points))

        # make a mask of which single shots will have a bit flip
        mask = np.random.rand(n_repeats, n_elements) < np.cos(np.linspace(0, np.pi * (i + 2), n_elements)) ** 2
        phase_offset = np.exp(1j * np.pi / 2. * mask + 1j * np.random.randn(n_repeats, n_elements) * noise)
        phase_offset *= np.random.randn(n_repeats, n_elements) * noise + 1.
        q_trace = np.einsum('i,jk->kji', q_trace, phase_offset)

        trace[0] += np.imag(q_trace)
        trace[1] += np.real(q_trace)

    return trace

