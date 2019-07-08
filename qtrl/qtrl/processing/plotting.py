# Copyright (c) 2018-2019, UC Regents

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .base import ADCProcess, find_resonator_names
import numpy as np
import matplotlib as mpl
# Define the grid shapes for plots which will vary as the qubit count increases
# IE: 1 qubit should be plotted on a 1x1 grid, 2 on a 2x1 etc
plot_shapes = {1: [1, 1],
               2: [2, 1],
               3: [3, 1],
               4: [4, 1],
               5: [5, 1],
               6: [3, 2],
               7: [4, 2],
               8: [4, 2]}


class PlotIQHeatmap(ADCProcess):
    """Plot the blobs of an IQ measurement"""
    def __init__(self, log_plot=True, input_name='Heterodyne'):
        """Accepts log_plot, which generates the plot in log scale"""
        self._input_name = input_name
        self._log = log_plot

    def post(self, measurement, seq=None):
        plot_heatmap(measurement, self._input_name, self._log)


def plot_heatmap(measurement, input_name='Heterodyne', trigger=-1, log=True, save_plot=True,config=None,show_plot=False):
    """Plot a heatmap of the measurement"""
    res_names = find_resonator_names(measurement)

    plt.figure(figsize=np.array(plot_shapes[len(res_names)]) * 3)

    # for each resonator make a plot
    for i, res in enumerate(res_names):
        plt.subplot(plot_shapes[len(res_names)][1],
                    plot_shapes[len(res_names)][0],
                    i + 1)
        r = np.max(np.abs(measurement[res][input_name][..., -1])) * 1.1

        if config is not None:

            # plot the decision boundary and print separation
            if config.ADC._processing['00_simple_readout']['055_GMM'][1]:
                nx, ny = 200, 100
                x_min, x_max = [-r, r]
                y_min, y_max = [-r, r]
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                                     np.linspace(y_min, y_max, ny))
                Z = config.ADC._processing['00_simple_readout']['055_GMM'][0]._mixes[i].predict(
                    np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                cmap = plt.get_cmap('Accent')
                colors_to_use = [cmap.colors[z] for z in np.unique(Z)]
                cm = mpl.colors.ListedColormap(colors_to_use,name='state_map', N=len(colors_to_use))

                plt.pcolormesh(xx, yy, Z, cmap=cm, alpha=1.0, shading='gouraud',)
                plt.colorbar(label='State',ticks=np.unique(Z), fraction=0.046, pad=0.04)
                plt.contour(xx, yy, Z, [0.5], linewidths=2., alpha=0.5, colors='k')

                sep_fidelity = measurement[res]['GMM_separation']['Separation']
                plt.text(0.25, 0.8, f'Sep.: {np.around(sep_fidelity,1)}', ha='center', va='center', transform=plt.gca().transAxes)

        plt.axhline(0, alpha=0.1, color='black')
        plt.axvline(0, alpha=0.1, color='black')
        plt.gca().set_aspect('equal')
        if log:
            plt.hexbin(*measurement[res][input_name][..., trigger].reshape(2, -1),
                       cmap='Greys',
                       norm=LogNorm(),
                       extent=[-r, r, -r, r], alpha=0.4)
        else:
            plt.hexbin(*measurement[res][input_name][..., trigger].reshape(2, -1),
                       cmap='Greys',
                       extent=[-r, r, -r, r],alpha=0.4)


        plt.text(0.1, 0.9, res, ha='center', va='center', transform=plt.gca().transAxes)
        plt.tight_layout()
    if save_plot:
        assert 'save_path' in measurement.keys(), "no save path known!"
        save_path = measurement['save_path']['filename']
        print(f'saving plot at {save_path}')
        plt.savefig(save_path, dpi=200)
    if not show_plot:
        plt.close()
