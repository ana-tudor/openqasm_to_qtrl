# Copyright (c) 2018-2019, UC Regents

from .base import ADCProcess, find_resonator_names
from ..fitting import fit, common
import numpy as np
import matplotlib.pyplot as plt
from .plotting import plot_shapes

class FitExpAll(ADCProcess):
    def __init__(self, plot=False, input_name='AxisTrigger', result_name='FitExpAll'):
        self._plotting = plot
        self._result_name = result_name
        self._input_name = input_name

    @staticmethod
    def plot(measurement, result_name='FitExpAll', save_plot=True):
        res_names = find_resonator_names(measurement)

        plt.figure(figsize=np.array(plot_shapes[len(res_names)])*3)
        for i, res in enumerate(res_names):
            plt.subplot(plot_shapes[len(res_names)][1],
                        plot_shapes[len(res_names)][0],
                        i + 1)
            fit_res = measurement[f'{res}'][result_name]

            if 'y_fit' not in fit_res:
                plt.text(0.2, 0.2, 'Fit Failed', ha='center', va='center', transform=plt.gca().transAxes)
            else:
                plt.plot(fit_res['x'],
                         fit_res['y_fit'],
                         label='Decay: {0:0.2f} us'.format(fit_res['params']['tau']),
                         c='black',
                         ms=2)
            plt.plot(fit_res['x'],
                     fit_res['y_original'],'.-')
            plt.text(0.1, 0.9, res, ha='center', va='center', transform=plt.gca().transAxes)
            plt.legend(loc=1)
            if (np.max(fit_res['y_original']) - np.min(fit_res['y_original'])) < 2:
                plt.ylim(0, 1)
        plt.tight_layout()
        if save_plot:
            assert 'save_path' in measurement.keys(), "no save path known!"
            save_path = measurement['save_path']['filename']
            plt.savefig(save_path, dpi=200)

    def post(self, measurement, seq=None):
        res_names = find_resonator_names(measurement)

        # for each resonator make a plot
        for res in res_names:
            if seq is not None:
                x_axis = seq.x_axis
            else:
                x_axis = np.arange(measurement[res][self._input_name].shape[2])
            y_axis = measurement[res][self._input_name]

            mean = np.mean(y_axis)
            amp_guess = np.max(y_axis) - np.min(y_axis)

            result_dict = dict()
            result_dict['x'] = x_axis
            result_dict['y_original'] = y_axis

            if len(x_axis) < 3:
                return

            try:
                fit_result = fit.fit1d(x_axis,
                                       y_axis,
                                       common.fit_exp_decay_with_offset,
                                       mean,
                                       amp_guess,
                                       np.mean(x_axis),
                                       ret=True)
            except Exception as e:
                print(e)
                measurement[res][self._result_name] = result_dict
                return

            if not isinstance(fit_result, int) and fit_result['success'] == 1:
                result_dict['params'] = fit_result['params_dict']
                result_dict['fit_func_str'] = fit_result['fitfunc_str']
                result_dict['y_fit'] = fit_result['fitfunc'](fit_result['x'])
                # print(fit_result.keys())
                del fit_result['fitfunc']  # this is unpickleable
                result_dict['fit_result'] = fit_result

            measurement[res][self._result_name] = result_dict
        if self._plotting:
            self.plot(measurement, self._result_name)


class FitSinExpAll(ADCProcess):
    """Fits a decaying sine wave to the measurement, optionally plots the fit results"""
    def __init__(self, plot=False, input_name='AxisTrigger', result_name='FitExpSinAll'):
        self._plotting = plot
        self._result_name = result_name
        self._input_name = input_name

    @staticmethod
    def plot(measurement, result_name='FitExpSinAll',save_plot=True):
        # estimate the number of resonators in the measurement
        res_names = find_resonator_names(measurement)

        plt.figure(figsize=np.array(plot_shapes[len(res_names)])*3)
        for i, res in enumerate(res_names):
            plt.subplot(plot_shapes[len(res_names)][1],
                        plot_shapes[len(res_names)][0],
                        i + 1)

            fit_res = measurement[f'{res}'][result_name]
            plt.plot(fit_res['x'],
                     fit_res['y_original'],'.-')
            if 'y_fit' not in fit_res:
                plt.text(0.2, 0.2, 'Fit Failed', ha='center', va='center', transform=plt.gca().transAxes)
            else:
                decay = np.format_float_scientific(fit_res['params']['t'],precision=3)
                decay_error = np.format_float_scientific(fit_res['fit_result']['error_dict']['t'],precision=3)
                frequency = np.format_float_scientific(fit_res['params']['f'], precision=3)
                frequency_error = np.format_float_scientific(fit_res['fit_result']['error_dict']['f'], precision=3)
                plt.plot(fit_res['x'],
                         fit_res['y_fit'],
                         label='Decay: {0:}({1:}) us\nFreq: {2:}({3:}) MHz'.format(
                             decay,
                             decay_error,
                             frequency,
                             frequency_error),
                         c='black',
                         ms=2)
            plt.legend(loc=1)
            plt.text(0.1, 0.9, res, ha='center', va='center', transform=plt.gca().transAxes)
            if (np.max(fit_res['y_original']) - np.min(fit_res['y_original'])) < 2:
                plt.ylim(0, 1)
        plt.tight_layout()
        if save_plot:
            assert 'save_path' in measurement.keys(), "no save path known!"
            save_path = measurement['save_path']['filename']
            plt.savefig(save_path, dpi=200)


    def post(self, measurement, seq=None):
        # estimate the number of resonators in the measurement
        res_names = find_resonator_names(measurement)

        # for each resonator make a plot
        for i, res in enumerate(res_names):
            if seq is not None:
                x_axis = seq.x_axis
            else:
                x_axis = np.arange(measurement[res][self._input_name].shape[1])
            y_axis = measurement[res][self._input_name]

            mean = np.mean(y_axis)
            amp_guess = (np.max(y_axis) - np.min(y_axis))/2.

            # estimate the freq
            trace = y_axis - np.mean(y_axis)
            # trace = np.concatenate([np.zeros(1000), trace, np.zeros(1000)])
            # trace = np.fft.ifftshift(trace)

            result_dict = dict()
            result_dict['x'] = x_axis
            result_dict['y_original'] = y_axis

            if len(x_axis) < 5:
                return

            try:
                peak_ind = np.argmax(np.abs(np.fft.rfft(trace)))
                freq_est = np.fft.rfftfreq(len(trace), np.diff(x_axis)[0])[peak_ind]
                # phase_est = np.angle(np.fft.rfft(trace)[peak_ind])/np.pi*180
                fit_result = fit.fit1d(x_axis,
                                       y_axis,
                                       common.fit_decaying_cos,
                                       freq_est,
                                       mean,
                                       amp_guess,
                                       180.0,
                                       np.mean(x_axis),
                                       ret=True)
            except Exception as e:
                print(e)
                measurement[res][self._result_name] = result_dict
                return

            self._last_fit = fit_result

            if not isinstance(fit_result, int) and fit_result['success'] == 1:
                result_dict['params'] = fit_result['params_dict']
                result_dict['fit_func_str'] = fit_result['fitfunc_str']
                result_dict['y_fit'] = fit_result['fitfunc'](fit_result['x'])
                # print(fit_result.keys())
                del fit_result['fitfunc'] #this is unpickleable
                result_dict['fit_result'] = fit_result
            measurement[res][self._result_name] = result_dict

        if self._plotting:
            self.plot(measurement, self._result_name)

class FitCosAll(ADCProcess):
    """Fits a decaying sine wave to the measurement, optionally plots the fit results"""
    def __init__(self, plot=False, input_name='AxisTrigger', result_name='FitCosAll'):
        self._plotting = plot
        self._result_name = result_name
        self._input_name = input_name

    @staticmethod
    def plot(measurement, result_name='FitCosAll', save_plot=True):
        # estimate the number of resonators in the measurement
        res_names = find_resonator_names(measurement)

        plt.figure(figsize=np.array(plot_shapes[len(res_names)])*3)
        for i, res in enumerate(res_names):
            plt.subplot(plot_shapes[len(res_names)][1],
                        plot_shapes[len(res_names)][0],
                        i + 1)
            # print(measurement[f'{res}'].keys())
            fit_res = measurement[f'{res}'][result_name]
            plt.plot(fit_res['x'],
                     fit_res['y_original'],'.-')
            if 'y_fit' not in fit_res:
                plt.text(0.2, 0.2, 'Fit Failed', ha='center', va='center', transform=plt.gca().transAxes)
            else:
                # print(fit_res['params'])
                plt.plot(fit_res['x'],
                         fit_res['y_fit'],
                         label='x0: {0:0.2f} \nFreq: {1:0.3f} MHz'.format(
                             fit_res['params']['x0'],
                             fit_res['params']['f']),
                         c='black',
                         ms=2)
            plt.legend(loc=1)
            plt.text(0.1, 0.9, res, ha='center', va='center', transform=plt.gca().transAxes)
            if (np.max(fit_res['y_original']) - np.min(fit_res['y_original'])) < 2:
                plt.ylim(0, 1)
        plt.tight_layout()
        if save_plot:
            assert 'save_path' in measurement.keys(), "no save path known!"
            save_path = measurement['save_path']['filename']
            plt.savefig(save_path, dpi=200)

    def post(self, measurement, seq=None):
        # estimate the number of resonators in the measurement
        res_names = find_resonator_names(measurement)
        # print(self._result_name, res_names)
        # for each resonator make a plot
        for i, res in enumerate(res_names):
            # print(self._result_name, res)
            if seq is not None:
                x_axis = seq.x_axis
            else:
                x_axis = np.arange(measurement[res][self._input_name].shape[1])
            y_axis = measurement[res][self._input_name]

            mean = np.mean(y_axis)
            amp_guess = (np.max(y_axis) - np.min(y_axis))/2.

            # estimate the freq
            trace = y_axis - np.mean(y_axis)
            # trace = np.concatenate([np.zeros(1000), trace, np.zeros(1000)])
            # trace = np.fft.ifftshift(trace)

            result_dict = dict()
            result_dict['x'] = x_axis
            result_dict['y_original'] = y_axis

            if len(x_axis) < 5:
                return

            try:
                peak_ind = np.argmax(np.abs(np.fft.rfft(trace)))
                freq_est = np.fft.rfftfreq(len(trace), np.diff(x_axis)[0])[peak_ind]
                # print(freq_est)
                x0_est= x_axis[np.argmax(y_axis)]
                fit_result = fit.fit1d(x_axis,
                                       y_axis,
                                       common.fit_cos,
                                       freq_est,
                                       mean,
                                       amp_guess,
                                       x0_est,
                                       ret=True)
            except Exception as e:
                print(e)
                # result_dict['y_guess'] = common.fit_cos(freq_est, mean, amp_guess, 180.0)[1](x_axis)
                # result_dict['params'] = 'fit_failed'
                measurement[res][self._result_name] = result_dict
                return

            self._last_fit = fit_result

            if not isinstance(fit_result, int) and fit_result['success'] == 1:
                result_dict['params'] = fit_result['params_dict']
                # print(result_dict['params'])
                result_dict['fit_func_str'] = fit_result['fitfunc_str']
                result_dict['y_fit'] = fit_result['fitfunc'](fit_result['x'])
                # print(fit_result.keys())
                del fit_result['fitfunc']  # this is unpickleable
                result_dict['fit_result'] = fit_result
            measurement[res][self._result_name] = result_dict

        if self._plotting:
            self.plot(measurement, self._result_name)
