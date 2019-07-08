# Copyright (c) 2018-2019, UC Regents

from .pulse_library import envelope_lookup
import numpy as np
from collections import namedtuple
from scipy.sparse import linalg, lil_matrix
import copy
from collections import OrderedDict
import matplotlib.pyplot as plt
import warnings

import logging
log = logging.getLogger('qtrl')


__all__ = ['PulseEnvelope',
           'VirtualEnvelope',
           'UniquePulse',
           'Sequence']


PulseEnvelope = namedtuple("PulseEnvelope", ['env_func', 'kwargs', 'width', 't0'])
PulseEnvelope.__new__.__defaults__ = (0,)
VirtualEnvelope = namedtuple("VirtualEnvelope", ['phase', 'width'])
UniquePulse = namedtuple("UniquePulse", ['envelope', 'freq', 'channel', 'subchannel'])


class Sequence:
    def __init__(self, n_elements, sample_rate=1e9, x_axis=None, name='test_sequence'):

        self.modulation_freqs = []
        self._name = name
        self.pulse_envelopes = {}  # dict of the pulse envelopes
        self.abstract_locations = {}  # dict of the abstract location of gates
        self.abstract_virtual_locations = {}  # dict for z gates
        self.unique_locations = {'Start'}  # set of unique locations
        self.n_elements = n_elements
        self.phase_gate_list = []
        self.array = None
        self.compiled_array = None
        self._sample_rate = sample_rate

        if x_axis is None:
            self.x_axis = np.arange(n_elements)
        elif isinstance(x_axis, list) or isinstance(x_axis, np.ndarray):
            assert len(x_axis) == n_elements, 'x_axis is not the same length as n_elements'
            self.x_axis = x_axis
        else:
            self.x_axis = x_axis * np.arange(n_elements)

        # Number of markers and channels are found at compile time
        self._n_markers = 0
        self._n_channels = 1

    def append(self, pulse, element=slice(0, None), start=None, end=None,
               start_delay=0, end_delay=0, **kwargs):
        r"""  start                     |<  width   >|                     end
                |<    start_delay      >|/\/\/\/\/\/\|<    end_delay       >|

            start_delay/end_delay default to 0 samples
            if neither start or end are specified, start defaults to 'Start'

            if you pass an end_delay ref from another pulse, you can specify
               the end_delay length, and it will place the pulse with that
               much spacing
            if you additionally specify a start_delay length, then the
               start_delay reference passed back to you will start_delay that
               far before the pulse
        """

        # in the recursive step this was being passed down
        # and one branch of the recursion would make a change which
        # would effect other branches, better to just make a copy

        if kwargs:
            kwargs = copy.deepcopy(kwargs)

        # Now we have some code that lets us build composite pulses
        # we can pass lists, or lists of lists
        if isinstance(pulse, list):
            if isinstance(pulse[0], list):

                og_start, end = self.append(pulse[0],
                                            start=start,
                                            end=end,
                                            element=element,
                                            start_delay=start_delay,
                                            end_delay=end_delay,
                                            **kwargs)
                for l in pulse[1:]:
                    _, end = self.append(l,
                                         start=end,
                                         element=element,
                                         start_delay=start_delay,
                                         end_delay=end_delay,
                                         **kwargs)
                return og_start, end
            else:
                # if the list is empty, just return
                if len(pulse) == 0:
                    return start, end

                # We need to find the maximum width of the pulses in this list
                max_width = 0
                for p in pulse:
                    if p.envelope.width is not None:
                        max_width = max(p.envelope.width, max_width)

                for l in pulse:
                    # We need to add padding to the pulse so that the
                    # tick is the same size for all operations in this list
                    if l.envelope.width is not None:
                        specific_start = max_width - l.envelope.width + start_delay
                    else:
                        specific_start = max_width + start_delay

                    start, end = self.append(l,
                                             start=start,
                                             end=end,
                                             element=element,
                                             start_delay=specific_start,
                                             end_delay=end_delay,
                                             **kwargs)

                return start, end

        # Reference from the start_delay if not otherwise specified
        if start is None and end is None:
            start = 'Start'

        # see if we have any new definitions for kwargs and update
        env_kargs = OrderedDict()
        if kwargs:
            pulse = pulse._replace(freq=kwargs.pop('freq', pulse.freq),
                                   channel=kwargs.pop('channel', pulse.channel),
                                   subchannel=kwargs.pop('subchannel', pulse.subchannel))
            env_kargs.update(copy.deepcopy(pulse.envelope._asdict()))
            env_kargs.update(copy.deepcopy(kwargs))

        # add the frequency if not seen before!
        if pulse.freq is not None and pulse.freq not in self.modulation_freqs:
            self.modulation_freqs.append(pulse.freq)

        # Construct check validity of our locations, and update 'None's to be something
        start, end, start_delay, end_delay = self._location_check(start, end, start_delay, end_delay)

        from ..utils.config import _ConfigQEnvelop     # TODO: remove, only used for type testing
        if isinstance(pulse.envelope, VirtualEnvelope):
            pulse = pulse._replace(envelope=pulse.envelope._replace(**env_kargs))
            self._append_virtual_pulse(pulse, start_delay, end_delay, start, end, element)
        elif isinstance(pulse.envelope, PulseEnvelope):
            # Now we have to make sure the inside dictionary is updated
            # in the correct order
            if kwargs:
                env_kargs['kwargs'].update(copy.deepcopy(pulse.envelope.kwargs))
                if 'kwargs' in kwargs:
                    env_kargs['kwargs'].update(kwargs['kwargs'])

            # make a copy of the pulse with our overridden properties
            pulse = pulse._replace(envelope=pulse.envelope._replace(**env_kargs))

            self._append_real_pulse(pulse, start_delay, end_delay, start, end, element)
        elif isinstance(pulse.envelope, _ConfigQEnvelop):    # TODO: this makes no sense
            self._append_real_pulse(pulse, start_delay, end_delay, start, end, element)
        else:
            raise TypeError("envelope was not one of the known types! Pulse: {}".format(pulse.envelope))

        return start, end

    def _append_virtual_pulse(self, pulse, start_delay, end_delay, start, end, element):
        """Appends a virtual pulse to the sequence,
            Accepts a width, start_delay and end_delay like a normal pulse.
        """

        # find our start location in samples
        start_sample = int(np.rint(start_delay * self._sample_rate))

        if pulse.envelope.width is not None:
            width = int(np.rint(pulse.envelope.width * self._sample_rate))
        else:
            width = 0

        # pulse = pulse._replace(envelope=pulse.envelope._replace(width=width))

        width += int(np.rint(start_delay * self._sample_rate +
                             end_delay * self._sample_rate))

        # if we have never seen this abstract location before, record it
        if (start, end) not in self.abstract_locations:
            self.abstract_locations[(start, end)] = [width, [], [], []]
        else:
            # if we have, we need to append to this location, not overwrite
            assert self.abstract_locations[(start, end)][0] == width, "Pulse width is now overconstrained {}, {}, {}, {}".format(pulse, start, end, width)

        if start not in self.abstract_virtual_locations:
            self.abstract_virtual_locations[start] = [[pulse], [element], [start_sample]]
        else:
            self.abstract_virtual_locations[start][0].append(pulse)
            self.abstract_virtual_locations[start][1].append(element)
            self.abstract_virtual_locations[start][2].append(start_sample)

    def _append_real_pulse(self, pulse, start_delay, end_delay, start, end, element):
        """Appends a physical pulse to the sequence"""

        # check if we need to make the envelope
        width = self._envelope_check(pulse)

        # find our start location in samples
        end_sample = int(np.rint(end_delay * self._sample_rate))
        start_sample = int(np.rint(start_delay * self._sample_rate))

        # calculate the full width of the pulse
        width = width + end_sample + start_sample

        if pulse.freq is not None:
            assert pulse.freq in self.modulation_freqs, "Pulse {} freq doesnt match known frequencies!".format(pulse)

        # if we have never seen this abstract location before, record it
        if (start, end) not in self.abstract_locations:
            self.abstract_locations[(start, end)] = [width, [pulse], [element], [start_sample]]
        else:
            # if we have, we need to append to this location, not overwrite
            assert self.abstract_locations[(start, end)][0] == width, "Pulse width is now overconstrained {}".format(pulse)

            self.abstract_locations[(start, end)][1].append(pulse)
            self.abstract_locations[(start, end)][2].append(element)
            self.abstract_locations[(start, end)][3].append(start_sample)

    def compile(self):
        """This compiles our pulses down to the final complex array.
        First it takes our abstract locations and finds a real solution, then
        it maps the added pulses into a nicely formatted list of real locations
        Then finally it assembles the compiled_array."""

        log.debug("compiling sequence %s with %d location(s)", self._name, len(self.abstract_locations))

        # First we need to de-abstract our locations
        self._compile_location_mapping()

        # build our final matrix
        self._envelope_compile()

        # now we need to compile our phase multiplication table, then
        # multiply it against our envelope array and we are done.
        self._phase_compile()

        # Multiply out phase array by our envelope array to get the final thing
        self.compiled_array[:, :, :, 0] *= self.reduced_phase_array[:, :, np.newaxis]

        # Old code expects a specific shape, and real numbers
        # here is where that conversion happens
        # np.real takes a view so its faster to do it first before the sum
        # as we will only need to sum the real component of the array.
        # transpose is also a view, and costs us nothing to move before the sum
        # since sum will give us a new array, it's better to do it after the
        # views so all future operations are on nicely formatted data
        self.array = self.compiled_array.real.transpose((1, 2, 0, 4, 3)).sum(0)

        self.compiled_array = True

    def _phase_compile(self, freqs_to_ignore=[]):
        """Take our z-gates and modulation frequencies and generate an array we can
        multiply against our compiled_array
        Args: freqs_to_ignore - list of frequencies to ignore during phase compilation
        """

        self.phase_gate_list = []
        modulation_freqs = self.modulation_freqs[:]
        for freq in freqs_to_ignore:
            try:
                modulation_freqs.remove(freq)
            except ValueError:
                pass
        if len(modulation_freqs) == 0:
            modulation_freqs = [0]

        mod_index_lookup = {freq: freq_i for freq_i, freq in enumerate(modulation_freqs)}
        for location in self.abstract_virtual_locations:
            real_start = self.abstract_mapping[location]
            for pulse, element, start_delay in zip(self.abstract_virtual_locations[location][0],
                                                   self.abstract_virtual_locations[location][1],
                                                   self.abstract_virtual_locations[location][2]):
                if pulse.freq is not None and pulse.freq not in freqs_to_ignore:
                    freq_ind = mod_index_lookup[pulse.freq]
                else:
                    freq_ind = 0

                self.phase_gate_list.append([element, freq_ind, real_start+start_delay, pulse.envelope.phase])

        self.reduced_phase_array = np.ones((self.n_elements,
                                            len(modulation_freqs),
                                            self.max_time),
                                            dtype='complex64')

        i = np.complex64(1j)
        for z in self.phase_gate_list:
            self.reduced_phase_array[z[0], z[1], z[2]:] *= np.exp(i*np.complex64(z[3]))

        # create a temporary modulation array
        modulation = np.exp(2*np.pi*i*np.arange(self.max_time)[np.newaxis] *
                            np.array(np.reshape(modulation_freqs, (-1, 1))/self._sample_rate))

        self.reduced_phase_array *= modulation[np.newaxis]

    def _envelope_compile(self, channels_to_ignore=[]):
        """Take our envelop_index_locs and construct a final complex matrix of envelopes
        Args: channels_to_ignore - list of channels to not compile into the final sequence
        """

        # now we need to go through our pulses and replace
        # abstract locations with real locations, then put them
        # in a nicely formatted list
        self.envelope_index_locs = []
        mod_index_lookup = dict([(i, x) for x, i in enumerate(self.modulation_freqs)])
        for location in self.abstract_locations:
            real_start = self.abstract_mapping[location[0]]
            for pulse, element, start_delay in zip(self.abstract_locations[location][1],
                                                   self.abstract_locations[location][2],
                                                   self.abstract_locations[location][3]):
                if pulse.freq is not None:
                    freq_ind = mod_index_lookup[pulse.freq]
                else:
                    freq_ind = 0

                self._n_channels = np.max([self._n_channels, pulse.channel])
                self._n_markers = np.max([self._n_markers, pulse.subchannel])
                self.envelope_index_locs.append((element,
                                                 freq_ind,
                                                 pulse.channel,
                                                 pulse.subchannel,
                                                 real_start + start_delay,  # where env starts in time
                                                 repr(pulse)))  # pulse_dict key

        # in order to construct our final array, we must know it's size
        # so we need to find out how much time we need
        self.max_time = max(self.abstract_mapping.values())

        # Great, now we build our final array
        self.compiled_array = np.zeros((self.n_elements,
                                        len(self.modulation_freqs),
                                        self._n_channels+1,
                                        self._n_markers+1,
                                        self.max_time),
                                       dtype='complex64')

        # and add our envelopes to it
        for env in self.envelope_index_locs:
            if env[2] not in channels_to_ignore:
                self.compiled_array[env[0],
                                    env[1],
                                    env[2],
                                    env[3],
                                    env[4]:env[4]+len(self.pulse_envelopes[env[5]])] += self.pulse_envelopes[env[5]]

    def _envelope_check(self, pulse):

        pulse_envelope = pulse.envelope

        # First we have to replace the width with a width without units
        width = int(np.rint(pulse_envelope.width * self._sample_rate))

        # Now we check if this is already in the pulse_envelope dict
        # if not present we need to construct it
        env_key = repr(pulse)
        if env_key not in self.pulse_envelopes:
            self.pulse_envelopes[env_key] = envelope_lookup[pulse_envelope.env_func](width=width,
                                                                                     **pulse_envelope.kwargs)

        return width

    def _compile_location_mapping(self):
        """This builds a mapping from the referential locations defined in the abstract_locations
        to absolute time. It does this with sparse matricies and linear regression."""

        # Build a constraint matrix based on our unique locations
        # a LIL matrix is chosen as it has the quickest append time
        constraint_array = lil_matrix((len(self.abstract_locations)+1,
                                       len(self.unique_locations)),
                                       dtype='int32')

        # Fill out our widths from our abstract_location dictionary
        widths = np.zeros(len(self.abstract_locations)+1, dtype='int32')

        # We need to have a way of mapping from an index to
        # a key in our abstract location dictionary, this
        # builds that lookup and an inverse lookup
        lookup = {}
        lookup_i = {}
        for i, location in enumerate(self.unique_locations):
            lookup[location] = i
            lookup_i[i] = location

        # We define a global start location
        constraint_array[0, lookup['Start']] = 1
        widths[0] = 0

        # Fill out our constraint matrix
        for i, val in enumerate(self.abstract_locations):
            constraint_array[i+1, lookup[val[1]]] = -1
            constraint_array[i+1, lookup[val[0]]] = 1
            widths[i+1] = -self.abstract_locations[val][0]

        # Convert the LIL matrix to csc for the linear regression
        constraint_array = constraint_array.tocsc()

        # Solve it with HARD constraints on convergence
        fit = linalg.spsolve(constraint_array, widths)

        # Calculate our errors
        # this is done so that we can have more informational error messages
        # if solving doesn't converge to an error
        errors = constraint_array.dot(fit) - widths
        self.errors = errors
        self.og_fit = fit
        fit -= min(fit)
        self.fit = np.int32(np.round(fit).astype('int32'))

        locs = {}
        for i, v in enumerate(fit):
            locs[lookup_i[i]] = int(v+0)

        self._constraint_array = constraint_array
        self._widths = widths

        self.abstract_mapping = locs

        # A whole lot of error checking, some bugs in this still
        # but it's more helpful then buggy so far
        if np.around(np.sum(np.abs(errors))/len(errors), 3) > 1e-5:
            print("Couldn't Create location table!\n")
            print("Location error might be in reference to:")
            print('start\t\tend\t\terror')
            for i, err in enumerate(np.around(np.abs(errors), 10)):
                if err > 1e-5:
                    j, k = (np.argmax(constraint_array.todense()[i+1]),
                            np.argmin(constraint_array.todense()[i+1]))
                    print("'{}'\t\t'{}'\t\t{}".format(lookup_i[j], lookup_i[k], err))
            raise Exception("Couldn't create a locations table which made sense")

        return locs

    def _generate_random_ref(self):
        """Generates a random string which can be used as a location reference"""
        new_loc_name = "arb_loc_{}".format(np.random.randint(0, 10**9))
        while new_loc_name in self.unique_locations:
            new_loc_name = "arb_loc_{}".format(np.random.randint(0, 10**9))
        return new_loc_name

    def _location_check(self, start_ref, end_ref, start, end):
        """Verify that this is a good location, and update our known locations as
        appropriate"""
        if start_ref is None:
            assert end_ref is not None, "No reference points specified!, {}".format(start_ref)

        # If we don't provide one of the two reference locations,
        # we need to generate a randomly assigned name for this location
        if start_ref is None or end_ref is None:
            new_loc_name = self._generate_random_ref()

            # now we set it appropriately
            if start_ref is None:
                start_ref = new_loc_name
            else:
                end_ref = new_loc_name

        # add our names to our history of unique locations
        self.unique_locations.add(start_ref)
        self.unique_locations.add(end_ref)

        return start_ref, end_ref, start, end

    def plot(self, t_min=None, t_max=None, element=None, fig=None, show_labels=False):
        """Plot a sequence
            Accepts:
                t_min (optional) - labrad time unit of where to clip the display
                t_max (optional) - labrad time unit
                element (Optional) - if not provided, plots a heatmap of all of the sequence elements
                                     if an element number is provided, only plots that single element
                fig (int, optional): figure to plot in. If None, creates a new figure"""
        if t_min is None:
            t_min = 0
        if t_max is None:
            t_max = 1e9

        # the timepoints in natural units
        t_min = max(t_min, 0)
        t_max = min(t_max, self.array.shape[2] / self._sample_rate)

        # The axis in natural units converted
        seq_axis = [self.array.shape[1], 0]

        # the index of the corresponding time points
        t_min_point = int(np.rint(t_min * self._sample_rate))
        t_max_point = int(np.rint(t_max * self._sample_rate))

        t_min *= 1e6
        t_max *= 1e6
        time_axis = [t_min, t_max]

        # makes a table of what is non-zero in the sequence file so we only plot that
        channels_used = np.array(np.max(np.abs(self.array), (2, 1)) != 0)

        # Open a figure
        if fig is None:
            fig = plt.figure(figsize=(10, 5))

        if not channels_used.any():
            warnings.warn("No Waveforms in the sequence yet. Plotting Failed.", stacklevel=2)
            return None

        # are we plotting all elements or only 1
        ax = plt.gca()
        if element is None:
            # set our alpha appropriately
            alpha = 1. / channels_used.sum()
            ax.minorticks_on()
            ax.yaxis.grid(which='both')

            # figure out the extent of the plot for imshow to have the correct axis
            full_axis = np.concatenate([time_axis, seq_axis])

            # for each present in the sequence, plot it
            for chan in np.arange(channels_used.shape[0])[[np.sum(channels_used, 1) != 0]]:
                for elem in np.arange(channels_used.shape[1])[channels_used[chan]]:
                    r = np.max(np.abs(self.array[chan, :, t_min_point:t_max_point, elem]))
                    ax.imshow(self.array[chan, :, t_min_point:t_max_point, elem],
                              vmin=-r,
                              vmax=r,
                              interpolation='none',
                              aspect='auto',
                              alpha=alpha,
                              cmap='bwr',
                              extent=full_axis)

            ax.set_xlabel("Time (us)")
            ax.set_ylabel("Sequence Element")

        else:  # plot a single element
            ax.set_title("Sequence Element {}".format(element))

            for chan in np.arange(channels_used.shape[0])[[np.sum(channels_used, 1) != 0]]:
                for elem in np.arange(channels_used.shape[1])[channels_used[chan]]:
                    label_str = "Channel {}".format(chan)
                    if elem != 0:
                        label_str += " Mkr {}".format(elem)
                    ax.plot(np.linspace(t_min, t_max, t_max_point - t_min_point),
                            self.array[chan, element, t_min_point:t_max_point, elem],
                            label=label_str)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlim(*time_axis)
            ax.set_xlabel("Time (us)")
            ax.legend(loc="lower left")
        return fig

    def generate_seq_table(self, elem_len=None):
        """Split a 4 dimensional matrix into a list of unique elements and a sequence table from which to
        recreate the original.
            Input:
                - elem_len = None - Length of which to chop up the waveforms into smaller pieces in an attempt
                                   to find the list of unique elements.

            Returns:
                - Unique Waveforms - an array containing the unique chunks of all the waveforms, of dimension 3,
                                    Dimension 0 - the number of unique elements found
                                    Dimension 1 - the number of output elements, IE analog, mk1, mk2 etc
                                    Dimension 2 - elem_len and contains the actual waveform chunk.

                - seq_table - is what will become the sequence table.  It is laid out as so:
                                Dimension 0 - Channel of the AWG
                                Dimension 1 - Sequence Element Number
                                Dimension 2 - Chunk Number (this is waveform length/ elem_len)
                                The integer value n listed in the chunk number corresponds to the nth unique element
                                from the unique waveform array above.

            The original waveform can then be reconstructed in full using the results with the command:

                unique_waveforms[seq_table].reshape(num_chans, seq_len, elem_len*wav_len, -1)
            """

        if elem_len is None:
            elem_len = self.shape[-2]

        if len(self.shape) != 4:
            raise Exception(
                "Matrix must be of dimension 4, [0] is channels, [1] is sequence number, [2] is waveform, [3] is output")

        if self.shape[2] % elem_len != 0:
            raise Exception("Waveform length is not a multiple of element length")

        num_chans = self.shape[0]
        seq_len = self.shape[1]
        num_outputs = self.shape[3]
        wav_len = int(self.shape[2] / elem_len)

        self.array = np.round(self.array, 5)

        # Make what will become the sequence table
        seq_table = np.zeros(self.reshape(num_chans, seq_len, -1, elem_len, num_outputs).shape[0:3])

        # Chunk the waveform table into something of the right shape
        chunked_waveforms = self.reshape(num_chans, seq_len, -1, elem_len, num_outputs)

        # Hash the waveforms and stuff the hashes into the sequence table, we will replace these with more sensical
        # numbers later
        for chan in range(seq_table.shape[0]):
            for seq_elem in range(seq_table.shape[1]):
                for wav_elem in range(seq_table.shape[2]):
                    seq_table[chan, seq_elem, wav_elem] = hash(chunked_waveforms[chan, seq_elem, wav_elem].tostring())

        # Now we can count our unique hashes!
        unique_hashes = np.unique(seq_table)
        # print("Found {} unique chunks".format(len(unique_hashes)))
        # ok, lets relabel our hashed values to a more normal numbering from 0- unique number of hashes
        # lets make a little dictionary for this
        relabeling = dict(zip(unique_hashes, np.arange(unique_hashes.shape[0])))

        # change our seq_table to reflect this new numbering
        for chan in range(seq_table.shape[0]):
            for seq_elem in range(seq_table.shape[1]):
                for wav_elem in range(seq_table.shape[2]):
                    seq_table[chan, seq_elem, wav_elem] = relabeling[seq_table[chan, seq_elem, wav_elem]]
        seq_table = seq_table.astype(int)

        # now we need to collect our unique waveforms, we can use our spiffy new seq_table for that
        # here is the array we will stuff these into
        unique_waveforms = np.zeros((unique_hashes.shape[0], elem_len, num_outputs))

        # ok, we can use np.where to get a list of where all the elements are, take the first location
        # and use that location information to get the waveform element from the chunked array,
        # then stuff that into our unique_waveform array
        for i in range(unique_hashes.shape[0]):
            loc = np.where(seq_table == i)
            unique_waveforms[i] = chunked_waveforms[loc[0][0], loc[1][0], loc[2][0]]

        # Verify we built what we wanted
        if np.sum(unique_waveforms[seq_table].reshape(num_chans, seq_len, elem_len * wav_len, -1) != self.array) != 0:
            raise Exception("Sequence chunking failed, might be a hash collision (unlikely).")

        # Huzzah, we have a nice sequence table and list of unique elements
        return unique_waveforms, seq_table

    # Add a few lines so that the sequence object acts like a numpy array,
    # so now we can do seq[0] instead of seq.values[0]
    def __getitem__(self, key):
        return self.array.__getitem__(key)

    def __setitem__(self, key, value):
        self.array.__setitem__(key, value)

    def __getattr__(self, name):
        """Delegate to NumPy array as appropriate."""
        try:
            return getattr(self.array, name)
        except AttributeError:
            raise AttributeError(
                 "Sequence object has no attribute {}".format(name))

    def __getstate__(self):
        # seq.abstract_locations
        # seq.abstract_virtual_locations
        # seq.modulation_freqs
        # seq.n_elements
        # seq.pulse_envelopes
        # seq.unique_locations
        # seq.x_axis
        return {'abstract_locations' : self.abstract_locations,
                'abstract_virtual_locations': self.abstract_virtual_locations,
                'modulation_freqs': self.modulation_freqs,
                'n_elements': self.n_elements,
                'pulse_envelopes': self.pulse_envelopes,
                'unique_locations': self.unique_locations,
                'x_axis': self.x_axis,
                '_sample_rate': self._sample_rate}

    def __setstate__(self, state):
        # Number of markers and channels are found at compile time
        self._n_markers = 0
        self._n_channels = 1
        self.array = None
        self.compiled_array = None
        self.__dict__.update(state)
