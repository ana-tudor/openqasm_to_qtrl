# Copyright (c) 2018-2019, UC Regents

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os


def wafer_map_8_ring(data, gridxy=None, locs=None, cmap=None, databounds=None, title=None, cbarlabel=None, save=False):
    """Plot all frequencies found on a wafer of 8 qubit rings. This assumes there are 8x8 chips of 8 qubits rings.

    Args -
        data - 64 x 8  ndarray of qubit frequencies
        gridxy - unused for now, eventually will allow other wafer sized besides 8x8
        locs - which locations to plot, a list of tuples, if None, plots all chips
        cmap - which color map to use, defaults to bwr_r
        databounds - what frequencies will be clipped during plotting, defaults to no clipping
        title - a title which will be added to the plot
        cbarlabel - label for the color bar
        save - boolean, defaults to false
    """

    # quick helper function to convert x-y values to subplot indices for proper display of wafer plaquette
    def xy2ind(x, y, Nx, Ny): return (Ny - y) * Nx + x

    data_8qring = np.array([np.array([f[0:4], f[8:3:-1]]).T for f in data])

    if cmap is None:
        # colormap
        cmap = plt.cm.bwr_r
        cmap.set_under(color='grey')
        cmap.set_over(color='grey')
    else:
        cmap = cmap
    if gridxy is None:
        Nx, Ny = [8, 8]
    else:
        Nx, Ny = gridxy
    if databounds is not None:
        dmin, dmax = databounds
    else:
        dmin = np.min(data)
        dmax = np.max(data)

    fig = plt.figure(figsize=(8, 8))

    if locs is None:
        locs = np.array([[[i, j] for j in range(1, Ny + 1)] for i in range(1, Nx + 1)]).reshape([Nx * Ny, 2])

    for i, loc in enumerate(locs):
        plt.subplot(Nx, Ny, xy2ind(loc[0], loc[1], Nx, Ny))
        plt.pcolormesh(data_8qring[i],
                       cmap=cmap,
                       norm=mpl.colors.Normalize(vmin=dmin, vmax=dmax))
        plt.tick_params(axis='both', bottom=False, top=True)
        plt.xticks([])
        plt.yticks([])
    if title is not None:
        fig.suptitle(title, fontsize=20)

    cax = plt.axes([0.95, 0.1, 0.04, 0.8])
    if cbarlabel is not None:
        plt.colorbar(cax=cax, label=cbarlabel, extend='both', orientation='vertical')
    if save is not False:
        name = save
        plt.savefig(os.getcwd() + '/' + name + '.png', bbox_inches='tight')
