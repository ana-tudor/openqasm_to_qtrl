# Remote Scheduling of Experiments

## Client Side TODO:

Currently client side support allows for an efficient packing
of the configuration of the measurements, as well as efficient
packing of a sequence object. Need to figure out API standards.

### Sequence and SequenceWrapper
`qtrl.sequencer.Sequence` objects can be created by a
`qtrl.sequencer.SequenceWrapper`, which works very similarly to a
standard Sequence class object. The primary difference is in the
addition of pulses. You do not pass actual pulse objects themselves
anymore, you just pass the key for the pulse dictionary.
At compile time you pass in the pulse dictionary object.

`SequenceWrapper.Compile()` is then the point at which a standard
Sequence object is created, some types of errors may be hidden because
of this later Sequence compilation step.

An Example of the differences:
```python
    # The original Sequence
    seq = Sequence(n_elements=10)
    start_ref, end_ref = seq.append(pulse_dict['Q6/X180'], element=5)
    seq.compile()

    # Now the equivalent SequenceWrapper version
    seq = SequenceWrapper(n_elements=10)
    start_ref, end_ref = seq.append('Q6/X180', element=5)
    seq.compile(pulse_dict)
```

The advantages of this SequenceWrapper object is that is very tiny,
the user doesn't have to know about what pulses are until compile time,
and it is easy to send to a remote server.  This means that it will
be the method by which sequences are sent to a remote server.

## Server Side TODO:

What does the server need to do in general:

- Maintain a default configuration for the server, including
  hardware settings, pulse definitions etc. When a configuration
  is provided, it should copy the default config, and UPDATE it with
  the provided values. This allows the user to not have to provide entire
  copies of the config, just the values the user wishes to update.
- Provide a way to access the default configuration and update
  it if necessary (lower priority)
- Allow for queries from users, including place in the Job Queue,
  and possibly even a job status check. This will mean that the
  acquisition will need to happen concurrently with the queries.
  Since the acquisition is a blocking behavior, I think this means
  that the acquisition must be a threaded process.
- Maintain a PriorityQueue list of all jobs it is currently working on

What constitutes a "Job" in this case:  

- A job should contain all of the information required to preform
  a full acquisition on the hardware. The minimum set should probably
  just be a SequenceWrapper object, and the server can use a default
  configuration for the acquisition. 