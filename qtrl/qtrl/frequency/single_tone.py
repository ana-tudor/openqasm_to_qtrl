import numpy as np
from .res_plotting import plot_resonator, plot_resonator_location
import matplotlib.pyplot as plt


def find_resonators(config, filename=None, show_plot=True):
    """
    Find resonances using the VNA within the frequency range defined in config.

    Arguments:
    config:     A Metamanager object with attribute devices, with devices having a VNA object
                Config also must have a single_tone attribute, with 'n_resonators' and 'peak_widths'
                settings under 'find_resonators'

    filename:   string, A filepath to save the image, if desired
    show_plot:  boolean, whether or not to show the plot in the output

    Returns:
    res_freqs:  list of resonator frequencies [GHz], with number of resonators specified in
                the config single_tone
    """

    # set the settings specified in the config
    config.devices.reset_devices()

    # get the VNA from devices and  set according to find_resonators config keys
    vna = config.devices.connections['vna']
    for setting in config.single_tone['find_resonators']['vna']:
        getattr(vna, setting)(config.single_tone['find_resonators']['vna'][setting])

    logmag, phase = vna.get_trace()  # assumes output format is in logmag, phase
    mag = np.array(10.0 ** (np.array(logmag) / 20))
    s_param = mag * np.exp(-1.j * np.array(phase))
    f = np.linspace(vna.start_frequency(), vna.stop_frequency(), vna.num_points()) / 1e9

    res_freqs = plot_resonator_location(f, s_param, n_res=config.single_tone['find_resonators']['n_resonators'],
                                        peak_fitting=config.single_tone['find_resonators']['peak_widths'],
                                        filename=filename, show_plot=show_plot)
    return res_freqs


def characterize_resonator(config, center_freq, filename=None, save_dir='characterize_resonator', exp_config={}, fit_errors=True, show_plot=False):
    """
    Characterize resonator using the VNA within the frequency range defined in config.

    Arguments:
    config:     A Metamanager object with attribute devices, with devices having a VNA object
                Config also must have a single_tone attribute, with 'n_resonators' and 'peak_widths'
                settings under 'find_resonators'.

                the single_tone attribute must also have a 'res_fitting' key, with 'vna' key containing
                keys that are all attributes of the VNA device within the config

    center_freq: float, frequency in GHz around which to center the VNA trace

    filename:   string, A filepath to save the image output by plot_resonator(), if desired

    fit_errors: bool, whether to include fit uncertainties or not (default=True)

    show_plot:  boolean, whether or not to show the plot in the output

    Returns:    a dict containing the frequencies, the vna trace, and the fit results of
                the measurement
    """

    # set the settings specified in the config
    # config.devices.reset_devices()
    config.load()
    # get the VNA from devices
    vna = config.devices.connections['vna']

    # set auxiliary settings of the VNA
    settings = config.single_tone['characterize_resonator']['vna']
    for func in settings:
        getattr(vna, func)(settings[func])

    vna.center_frequency(center_freq * 1e9)

    # take the data
    logmag, phase = vna.get_trace()
    mag = np.array(10.0 ** (np.array(logmag) / 20))
    trace = mag * np.exp(-1.j * np.array(phase))
    f = np.linspace(vna.start_frequency(), vna.stop_frequency(), vna.num_points()) / 1e9
    results = plot_resonator(f, trace, fit_errors=fit_errors)
    data_dict = {'frequencies': f,
                 'logmag': logmag,
                 'phase': phase,
                 'fit_results': results}

    # saving
    instrument_metadata = config.devices.get_instrument_metadata()
    metadata = {**instrument_metadata, **exp_config}
    config.datamanager.make_save_dir(rf'{save_dir}')
    config.datamanager.save_data(rf'{save_dir}', filename=filename,
                                 data_dict=data_dict, config_dict=metadata,
                                 extension=config.datamanager.data_format)

    plt.savefig(config.datamanager.last_filename, dpi=200)
    if show_plot:
        plt.show()  # plotting
    return data_dict


def characterize_resonators_sweep(config, res_freqs, sweep_dict, instrument=None, save_files=True, fit_errors=True, show_plot=True):
    """
    Characterize resonators while sweeping over an instruments values

    Arguments:
    config:     A Metamanager object with attribute devices, with devices having a VNA object
                Config also must have a single_tone attribute, with 'n_resonators' and 'peak_widths'
                settings under 'find_resonators'.

                the single_tone attribute must also have a 'res_fitting' key, with 'vna' key containing
                keys that are all attributes of the VNA device within the config

    res_freqs:  list, frequencies in GHz around which to characterize each resonator

    save_files: bool, whether to save the images output by plot_resonator() for each sweep iteration

    fit_errors: bool, whether to include fit uncertainties or not (default=True)

    show_plot:  boolean, whether or not to show the plot in the output

    Returns:    a dict with one argument containing the swept parameter values, and keys
                for each resonator fit, with a list value of the results of the resonator characterization
                for each value of the swept instrument e.g. {'VNA_power': [...], 'R1': [{}, {},..], 'R2': [{},{},...]}
    """

    sweep_data = {f"{sweep_dict['instrument']}_{sweep_dict['setting']}": sweep_dict["values"]}
    if instrument is None:
        instrument = config.devices.connections[sweep_dict['instrument']]

    for i, r in enumerate(res_freqs):
        if f'R{i}' not in sweep_data:
            sweep_data[f'R{i}'] = []
        for j, v in enumerate(sweep_dict['values']):
            getattr(instrument, sweep_dict['setting'])(v)

            # add in metadata to be saved
            exp_metadata = {f"{sweep_dict['instrument']}_{sweep_dict['setting']}": v}
            # config.datamanager.exp_config = exp_metadata
            if save_files:
                fname = rf"R{i}_fit_{sweep_dict['instrument']}_{sweep_dict['setting']}_index{str(j).zfill(3)}"
            else:
                fname = None
            sweep_data[f'R{i}'].append(characterize_resonator(config, r, fit_errors=fit_errors,
                                                              show_plot=show_plot, filename=fname,
                                                              exp_config=exp_metadata))
    return sweep_data
