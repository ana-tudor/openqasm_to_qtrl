# Copyright (c) 2018-2019, UC Regents

import matplotlib.pyplot as plt
import numpy as np


def plot_chip(frequencies, filename=None, name='', allowed_detuning=[0.05, 0.20]):
    """Plot the frequencies of a chip in a standard way
        Args:
            - frequencies - a list of frequencies, in units of GHz
            - filename - optional, if provided the plot will be saved
            - name - optional, a name which will be included in the title
            - allowed_detunings - plots bands which illustrate what are good detunings bewteen pairs of qubits
                            defaults to [0.05, 0.2] which means 50MHz and 200Mhz
    """

    # plot it
    plt.figure(figsize=(8, 4))

    # plot the frequency spread between each pair of qubits, assuming a ring.
    plt.subplot(1, 2, 1)

    plt.plot(np.diff(np.hstack([frequencies, frequencies[0]])), 'o')

    plt.fill_between([-0.5, len(frequencies) - .5],
                     2*[allowed_detuning[1]], 2*[allowed_detuning[0]],
                     alpha=0.1)
    plt.fill_between([-0.5, len(frequencies) - .5],
                     2*[-allowed_detuning[1]], 2*[-allowed_detuning[0]],
                     alpha=0.1)
    plt.xlim([-.5, len(frequencies) - .5])
    plt.ylim([-0.35, 0.35])
    x_labels = ['{}/{}'.format(*x) for x in zip(range(0, len(frequencies)), np.arange(1, len(frequencies)+1) % len(frequencies))]

    plt.xticks(range(len(frequencies)), x_labels)
    plt.xlabel("Qubit Pair")
    plt.ylabel("Difference in Freq (GHz)")
    plt.title("{} CR".format(name))

    # plot the frequencies of each qubits
    plt.subplot(1, 2, 2)
    plt.title("{} Frequencies".format(name))
    plt.plot(frequencies, 'o')
    plt.xlim(-0.5, len(frequencies) - .5)
    plt.ylim(4.75, 6.05)
    plt.grid()
    plt.xlabel('Qubit')
    plt.ylabel("Frequency (GHz)")
    plt.tight_layout()

    # save data
    if filename is not None:
        plt.savefig(filename)
    plt.show()
