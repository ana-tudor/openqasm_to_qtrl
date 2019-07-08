# Copyright (c) 2018-2019, UC Regents

from . import Sequence
import copy

class SequenceWrapper:
    def __init__(self, n_elements, x_axis=None, sample_rate=None):
        """This is a wrapper around a standard Sequence.
        It records the absolute minimum in order to rebuild a full sequence.
        It is almost a direct replacement for the Sequence object.
        The primary change is that the pulses which are appended should be the keys
        in the pulse dictionary which is specified during compile.

        An example of this, 'pulses' must be previously defined:
            # using the Sequence
            >>> seq = Sequence(n_elements=10, sample_rate=1e9)
            >>> start_ref, end_ref = seq.append(pulses['Q6/X180'])
            >>> seq.compile()

            # the equivalent SequenceWrapper
            >>> seq = SequenceWrapper(n_elements=10)
            >>> start_ref, end_ref = seq.append('Q6/X180')
            >>> seq.compile(pulses, sample_rate=1e9)

        The use case for this is when you don't want the end user to have access
        to the raw pulse definitions, or to minimize the representation of an experiment.

        Since it doesn't build full envelopes it should not have any sequence related
        errors until the compile() step.  As a result of this it is probably
        better to use Sequence during development."""

        # same info as the standard Sequence
        self.n_elements = n_elements
        self.x_axis = x_axis
        self.unique_locations = set()
        self._sample_rate = sample_rate

        # all the appended pulses are now just recorded in a List
        # at compile time these will be added in order to a Sequence object
        self.appends = []

    def append(self, pulse, element=slice(0, None), start=None, end=None,
               start_delay=0, end_delay=0, **kwargs):
        """Append exactly the same as the append in Sequence, see that for more details"""
        # check validity of our locations, and update 'None's to be something

        # Reference from the start_delay if not otherwise specified
        if start is None and end is None:
            start = 'Start'
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

                    for l in pulse:
                        start, end = self.append(l,
                                                 start=start,
                                                 end=end,
                                                 element=element,
                                                 end_delay=end_delay,
                                                 **kwargs)

                    return start, end

        (start, end, start_delay, end_delay) = self._location_check(start,
                                                                    end,
                                                                    start_delay,
                                                                    end_delay)

        last_append = {'pulse': pulse,
                       'element': element,
                       'start': start,
                       'end': end,
                       'start_delay': start_delay,
                       'end_delay': end_delay}

        last_append.update(kwargs)

        # wow this looks bad
        self.appends.append(last_append)

        return start, end

    def compile(self, pulse_dict, sample_rate=1e9):
        """Compile exactly like the compile in Sequence, and returns a Sequence object
        However requires a pulse_dictionary like object.  Optionally accepts a sample_rate
        if it was not specified during init."""

        if sample_rate is None:
            sample_rate = self._sample_rate

        assert sample_rate is not None, "No sample rate specified."
        seq = Sequence(self.n_elements, sample_rate, self.x_axis)
        
        # The pulse_dict or Sequence is changing the list when passed
        # going to pass a copy to not have this happen.
        # TODO: Sort this out properly
        
        apps = copy.deepcopy(self.appends)
        for app in apps:
            app['pulse'] = pulse_dict.get(app['pulse'])
            seq.append(**app)

        seq.compile()
        return seq

    # We still need to generate start and end references the same as we would with
    # the standard Sequence object, we we just grab those functions.
    _generate_random_ref = Sequence._generate_random_ref
    _location_check = Sequence._location_check
