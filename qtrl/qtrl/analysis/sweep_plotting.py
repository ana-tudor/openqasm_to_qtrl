# -*- coding: utf-8 -*-
"""
Functions used for plotting sweeps
"""

import decimal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fill_in_mesh(X, Y, Z):
    """
    Converts three arrays used to create a mesh into a single dataframe, which
    can be used for plotting in imshow. The function will fill in blank cells
    with the average value off the entire mesh. The x-array is mapped to the
    x-dimension of an image, the y-array is mapped to the y-dimension of an
    image, and the z-array is mapped to the pixel intensities of an image.

    Args: x, y, z arrays

    Returns: dataframe whose columns are the x-array values, rows are the
             y-array values, and entries are the z-array values.
    """

    # Calculates the x and y step sizes. The rounding to 8 decimal places is
    # to fix float64 rounding errors. This is not a robust fix and could lead
    # to errors at some point.
    x_step = round(np.diff(X[0])[0], 8)
    y_step = round(np.diff(Y.T[0])[0], 8)
    print(x_step, y_step)

    x_precision = abs(decimal.Decimal(str(x_step)).as_tuple().exponent)
    y_precision = abs(decimal.Decimal(str(y_step)).as_tuple().exponent)

    # x_min = round(X.min(), x_precision)
    # x_max = round(X.max(), x_precision)
    y_min = round(Y.min(), y_precision)
    y_max = round(Y.max(), y_precision)

    columns = np.round(X[0], x_precision)
    rows = np.arange(y_min, y_max + y_step, y_step)[::-1]
    img_df = pd.DataFrame(index=rows, columns=columns)

    # Fill in the mesh values while
    for i in range(len(columns)):
        selected_rows = np.round(img_df.index, y_precision).isin(np.round(Y.T[i], y_precision))
        img_df[columns[i]][selected_rows] = Z.T[i, :]

    # Fill in empty cells of dataframe with average of Z_min and Z_max (center of colorbar)
    img_df = img_df.fillna((img_df.max().values.max() + img_df.min().values.min()) / 2)

    return img_df


def plot_two_tone_vs_flux(two_tone_vs_flux_data, title=None, filename=None, figsize=(8, 6), fluxes=None):
    plt.figure(figsize=figsize)
    phases_to_plot = []
    frequencies_to_plot = []
    fluxes_to_plot = []
    for two_tone_meas in two_tone_vs_flux_data:
        qubit = two_tone_meas['config']['qubit']
        phase_temp = np.array(two_tone_meas['phases'])
        phase_temp -= phase_temp.mean()
        phase_temp /= np.max(np.abs(phase_temp))
        phases_to_plot.append(phase_temp)

        frequencies = two_tone_meas['frequencies'] / 1e9
        frequencies_to_plot.append(frequencies)
        current = [two_tone_meas['config'][f'flux_{qubit}']]  # ['fluxes'][f'Q{qubit}']]
        fluxes_to_plot.append(current)
    if fluxes is None:
        fluxes_to_plot = np.hstack([fluxes_to_plot] * len(frequencies))
    else:
        fluxes_to_plot = np.hstack([fluxes] * len(frequencies))
    try:
        img_df = fill_in_mesh(fluxes_to_plot.T, np.array(frequencies_to_plot).T, np.array(phases_to_plot))
        plt.imshow(img_df, aspect='auto', cmap='RdBu',
                   extent=[img_df.columns[0], img_df.columns[-1], img_df.index[-1], img_df.index[0]])
    except:
        plt.pcolormesh(fluxes_to_plot.T, np.array(frequencies_to_plot).T, np.array(phases_to_plot).T, cmap='RdBu')

    # plot formatting stuff
    plt.rc('font', **{'size': 16, 'family': 'Arial'})
    plt.rc('axes', **{'linewidth': 1, 'edgecolor': 'black'})
    plt.rc('grid', **{'linewidth': 2, 'color': '0.8'})

    plt.colorbar(label='Reflected Phase Shift [rad]')
    plt.xlabel('Flux [$\Phi_0$]')
    plt.ylabel('Frequency [GHz]')
    if title is None:
        title = 'Two Tone vs Flux'
    plt.title(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(fname=filename, dpi=400)
