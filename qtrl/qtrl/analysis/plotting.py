# Copyright (c) 2018-2019, UC Regents

import matplotlib.pyplot as plt
import numpy as np
from .sim_tools import vec_to_dm
from matplotlib.font_manager import FontProperties


def plot_dm(dm, show_colorbar=False, show_ticks=True, x_labels=None, y_labels=None, axes=None,
            title=None, scale=None, text_color=None, rounding=2):
    """Plot a density matrix"""
    dm = vec_to_dm(dm)

    x_dim = np.shape(dm)[1]

    if scale is None:
        scale = np.max([x_dim / 1. - 1, 1.3])

    if axes is None:
        fig = plt.figure(figsize=(scale, scale))

    dm_show = dm

    if axes:
        im = axes.imshow(np.real(dm_show), interpolation='none', cmap='RdBu', vmin=-1, vmax=1)
    else:
        im = plt.imshow(np.real(dm_show), interpolation='none', cmap='RdBu', vmin=-1, vmax=1)
    small_font = FontProperties()  # size='small')

    for x in range(dm.shape[1]):
        for y in range(dm.shape[0]):
            if np.abs(np.round(dm_show[y, x], rounding)) > 0:

                # Get the color based on the value, sampled from grey scale
                if text_color is None:
                    current_text_color = plt.get_cmap("Greys")(1-abs(np.real(np.round(dm_show[y, x], rounding))))
                else:
                    current_text_color = text_color

                val = np.round(dm_show[y, x], rounding)
                if not np.isclose(np.imag(val), 0):
                    val_str = "{}\n{}".format(np.real(val), np.imag(val))
                else:
                    val_str = "{}".format(np.real(val))
                if axes:
                    axes.text(x, y, val_str, ha='center', va='center', color=current_text_color, fontproperties=small_font)
                else:
                    plt.text(x, y, val_str, ha='center', va='center', color=current_text_color, fontproperties=small_font)

    if show_ticks:
        if x_labels is None:
            fmt_str = "{0:0" + str(int(np.log2(dm.shape[1]))) + "b}"
            plt.xticks(np.arange(dm.shape[1]),
                       [fmt_str.format(x) for x in np.arange(dm.shape[1])],
                       rotation='vertical',
                       fontproperties=small_font)
        else:
            plt.xticks(np.arange(dm.shape[1]),
                       x_labels,
                       fontproperties=small_font)

        if y_labels is None:
            fmt_str = "{0:0" + str(int(np.log2(dm.shape[0]))) + "b}"
            plt.yticks(np.arange(dm.shape[0]),
                       [fmt_str.format(x) for x in np.arange(dm.shape[0])],
                       fontproperties=small_font)
        else:
            plt.yticks(np.arange(dm.shape[0]),
                       y_labels,
                       fontproperties=small_font)

        from matplotlib.ticker import AutoMinorLocator
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
        plt.tick_params(which='minor', length=0)
        plt.grid(which='minor')
    if title:
        plt.title(title)

    if show_colorbar:
        plt.colorbar(im, fraction=0.03, pad=0.04)

    if not axes:
        return fig
    else:
        return axes
