# Processing

Processing is managed by the ADC_Manager class.
Acquisition of data is split into 4 distinct steps:

- **prep** - this is before acquisition has started, this can be used
    for checking file structures and adding folders, or other
    processes that should happen before the start of the acquisition.
    
- **batch_start** - acquisitions are done in batches, so if  you want to
    acquire 1M measurements, you cannot do that in one acquisition as
    there is not enough RAM in the computer to hold all of that raw
    data. However you can batch the acquisition process, so you only
    acquire some small number, like 1k measurements at a time. 
    You then acquire 100 batches, and you now have 100k total measurements.
    Example processes that you might run during this time
    are things like turning on the DAC/AWG, etc.
    
- **batch_end** - This gets run after the end of every batch,
    such as turning off of the DAC/AWG.  This is also useful to
    preform Heterodyne here, as that process can be run asynchronously
    with the acquisition.  
    
- **post** - this runs at the completion of all batches. This is then all
    post processing, data saving, plotting, etc etc.
    
Now that we have a general flow for how the ADC manages data
acquisition, we can start organizing our actual data acquisition.

Inside [qtrl/processing/base](./base.py) there is ADCProcess, which
is the base class which should be used for all processing classes.

# ADC_Manager

\# TODO: documentation


# Data Formats

Measurements are recorded as dictionaries,