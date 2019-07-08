# qtrl

A minimal set of control software for the QNL lab.
Reliance on LabRad has been removed, and much of the functionality will be replaced with QCodes.

TODO List:
 - Improve Working Examples
 - Clean up Alazar Code
 - Clean up demod code
 - write documentation of config standards

- **Device Management**   
Qcodes will be used to talk to devices, any new device driver added should be submitted
as a pull request on the original repo to reduce the amount of code maintenance placed on us.
We will not be using the more advanced features of qcodes as they seem very restricting at
this point in time, and not very well documented.

- **Configuration Management**  
Qcodes config files are poorly documented at this time, and from what I have seen, 
are not very human readable. Given this a config management tool from qnl_ctrl has 
been ported over with updates. This allows the creation of Config objects, which 
act exactly like a dictionary with additional benefits. The primary benefit is every 
time a value is accessed or changed it loads from the physical file. Additionally, 
we can specify a variables config, which allows for us to use one config as a reference
for values in other configs. Specifically what this means, is we can have a 'Variables'
config file, which contains values that can be used in other config files, each grabbing
from files whenever anything is accessed.  This will need to be evaluated for how
much of a time hit we are taking on the file read/writes.


