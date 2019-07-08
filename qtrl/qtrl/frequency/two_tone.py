import numpy as np
from qtrl.numerics.tunable_qubits import resonator_max, resonator_min
from qtrl.frequency.single_tone import characterize_resonator

from IPython.display import clear_output

from cycler import cycler
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', **{'size': 16, 'family': 'Arial'})
plt.rc('axes', **{'linewidth': 1, 'edgecolor': 'black'})
plt.rc('grid', **{'linewidth': 2, 'color': '0.8'})
color_set = sns.color_palette("Set1", n_colors=8, desat=.8)
plt.rcParams['axes.prop_cycle'] = cycler('color', color_set)


def two_tone(config, qubit, two_tone_freqs, exp_config=None, filename=None, save_dir='two_tone', plot_title=None):
    """
    Performs a two-tone continuous spectroscopy measurement using a VNA and an additional RF source. Prior to the two-
    tone frequency sweep, a single tone sweep to determine the frequency of the readout resonator

    :param config:          Metamanager object with the following attributes:
                                chip:   a config object specifying sample/chip parameters:
                                        Qubit:
                                            EJ_Sum or EJ:
                                            f_r_bare:
                                devices: InstrumentManager object
                                    connections:
                                        vna:
                                two_tone:
                                    two_tone:
                                        two_tone_generator:
                                            <generator settings here>
                                datamanager: DataManager object


    :param qubit:           int specifying which qubit is being driven

    :param two_tone_freqs:  array-like specifying the frequencies through which the two_tone_generator will sweep

    :param exp_config:      dict additional experiment-specific metadata to include in the metadata for the experiment

    :param filename:        str filename to save the data with (need only be a partial name)

    :param save_dir:        str directory name where the data will be saved. Data is saved
                            something like <config.datamanager.base_directory>/<save_dir>/<filename>

    :param plot_title:      str title to pass for plotting the two-tone data

    :return: data_dict      dict containing the frequencies, phases, and resonator-characterization data from the
                            measurement

    """
    try:

        if exp_config is None:
            exp_config = {}
        exp_config['qubit'] = f'Q{qubit}'
        exp_config['tag'] = 'two_tone'
        if 'EJ_Sum' in config.chip[f'Q{qubit}']:  # if this is a tunable qubit
            res_freq_est = np.mean([resonator_max(qubit, config.chip), resonator_min(qubit, config.chip)])
        elif 'EJ' in config.chip[f'Q{qubit}']:
            res_freq_est = config.chip[f'Q{qubit}']['f_r_bare']

        # get resonator frequency
        if filename is not None:
            res_fname = rf'{filename}_R{qubit}_fit'
        else:
            res_fname = rf'R{qubit}_fit'
        res_char = characterize_resonator(config, res_freq_est, filename=res_fname, save_dir=rf'{save_dir}_Q{qubit}',
                                          exp_config={'resonator': qubit, 'tag': 'characterize_resonator'})

        # set the vna to resonance and settings for two-tone
        # set up two-tone sweep based on resonance frequency estimate
        res_freq = res_char['fit_results']['f0']
        # set the settings specified in the config
        config.load()
        # get the VNA from devices
        vna = config.devices.connections['vna']

        # set auxiliary settings of the VNA
        settings = config.two_tone['two_tone']['vna']
        for func in settings:
            getattr(vna, func)(settings[func])
        vna.center_frequency(res_freq * 1e9)

        # get second tone and turn on drive
        two_tone_generator = config.devices.connections['two_tone_generator']
        # set two-tone settings of the two_tone_generator
        settings = config.two_tone['two_tone']['two_tone_generator']
        for func in settings:
            getattr(two_tone_generator, func)(settings[func])

        if filename is not None:
            two_tone_fname = rf'{filename}_Q{qubit}'
        else:
            two_tone_fname = rf'Q{qubit}'

        # do two-tone
        two_tone_generator.rf_output('on')
        phases = []
        for i, freq in enumerate(two_tone_freqs):  # do two-tone
            two_tone_generator.frequency(freq)

            phases.append(np.mean(vna.get_trace()[1]))
            clear_output(wait=True)

            plt.plot(two_tone_freqs[:i + 1] / 1e9, np.unwrap(phases))
            plt.xlim(np.min(two_tone_freqs) / 1e9, np.max(two_tone_freqs) / 1e9)
            plt.xlabel('Freq [GHz]')
            plt.ylabel('Arg[S21]')
            if plot_title is None:
                plt.title(f'Two Tone {two_tone_fname.replace("_", " ")}: $f_{{res}} = {np.around(res_freq,4)}$ GHz')
            else:
                plt.title(plot_title)
            plt.show()
        # turn off generator
        two_tone_generator.rf_output('off')

        data_dict = {'phases': np.array(phases),
                     'frequencies': np.array(two_tone_freqs),
                     'res_characterization': res_char}

        # saving
        instrument_metadata = config.devices.get_instrument_metadata()
        metadata = {**instrument_metadata, **exp_config}

        config.datamanager.save_data(rf'{save_dir}_Q{qubit}', filename=two_tone_fname,
                                     data_dict=data_dict, config_dict=metadata,
                                     extension=config.datamanager.data_format)

        plt.figure()
        plt.plot(two_tone_freqs / 1e9, np.unwrap(phases))
        plt.xlabel('Freq [GHz]')
        plt.ylabel('Arg[S21]')
        if plot_title is None:
            plt.title(f'Two Tone {two_tone_fname.replace("_", " ")}: $f_{{res}} = {np.around(res_freq,4)}$ GHz')
        else:
            plt.title(plot_title)
        plt.tight_layout()
        print('saving plot...')
        plt.savefig(config.datamanager.last_filename, dpi=200)
    except Exception as e:
        print(e)

    return data_dict
