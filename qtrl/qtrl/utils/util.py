# Copyright (C) 2018-2019  UC Regents

"""Small utility functions more generic than any single module."""

import collections
import time


class Progress:
    """Progress recorder to give an estimate on the time required to do an experiment.
       Usage:

            prog = progress(100)  # lets say we are doing 100 iterations
            for _ in range(100):
                print next(prog)
                #Other code here

    This prints out time elapsed, time remaining, and expected time of completion
    of the loop.
    """

    def __init__(self, iterations):
        self.iterations = iterations
        self.first_time = None
        self.iteration = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self, ):
        self.iteration += 1
        if self.first_time is None:
            self.first_time = time.localtime()
            return 'Iteration {} of {}, Started at {}'.format(self.iteration,
                                                              self.iterations,
                                                              time.strftime("%I:%M:%S%p", self.first_time))
        else:
            cur_time = time.mktime(time.localtime())
            elapsed_time = cur_time - time.mktime(self.first_time)
            time_left = elapsed_time / float(self.iteration - 1) * self.iterations
            expected_time = time.strftime("%I:%M:%S%p", time.localtime(time.mktime(self.first_time) + time_left))

            m, s = divmod(elapsed_time, 60)
            h, m = divmod(m, 60)

            m2, s2 = divmod(time_left - elapsed_time, 60)
            h2, m2 = divmod(m2, 60)

            return 'Iteration {0} of {1} ({6:.1f}%), Started: {2}, Ending: {3}, Elapsed: {4}, Remaining: {5}'.format(
                self.iteration,
                self.iterations,
                time.strftime("%I:%M:%S%p", self.first_time),
                expected_time,
                "%d:%02d:%02d" % (h, m, s),
                "%d:%02d:%02d" % (h2, m2, s2),
                100. * self.iteration / float(self.iterations))


def replace_vars(config, variables, max_depth=10, depth=0):
    """Recursively traverse a dictionary and convert any string which
    is in the variable dictionary into the value in the variable dictionary
    """

    if depth > max_depth:
        return
    depth += 1

    # 2 cases we care about, if we received a dictionary ...
    if isinstance(config, dict):
        for key in config:
            if isinstance(config[key], dict):
                replace_vars(config[key], variables, max_depth=max_depth, depth=depth)
            elif isinstance(config[key], list):
                replace_vars(config[key], variables, max_depth=max_depth, depth=depth)
            else:
                try:
                    config[key] = variables[config[key]]
                except KeyError:
                    replace_vars(config[key], variables, max_depth=max_depth, depth=depth)

    # ... or a list, which we need to iterate over
    elif isinstance(config, list):
        for i, item in enumerate(config):
            if not isinstance(item, collections.Hashable):
                replace_vars(item, variables, max_depth=max_depth, depth=depth)
            else:
                try:
                    config[i] = variables[item]
                except KeyError:
                    replace_vars(item, variables, max_depth=max_depth, depth=depth)


def update(d, u):
    """Recursively update a dictionary, taken from:
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """

    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
