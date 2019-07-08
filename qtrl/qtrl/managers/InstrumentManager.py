# Copyright (c) 2018-2019, UC Regents

from qcodes import VisaInstrument
from .ManagerBase import ManagerBase

import logging
log = logging.getLogger('qtrl.InstrumentManager')


# This is a lookup table to enable the storage of device specific config files
# When a new device needs to be stored in a config, the model should be added here
_DRIVER_LOOKUP = None
def driver_lookup():
    global _DRIVER_LOOKUP
    if _DRIVER_LOOKUP is None:
        _DRIVER_LOOKUP = dict()

        # pre-fill lookup with known drivers, if available
        try:
            from qcodes.instrument_drivers.Keysight import \
                 Keysight_N5183B, Keysight_N5230A, Keysight_E9010A, Keysight_E5063A
            _DRIVER_LOOKUP['N5183B'] = Keysight_N5183B.N5183B
            _DRIVER_LOOKUP['N5230A'] = Keysight_N5230A.N5230A
            _DRIVER_LOOKUP['E9010A'] = Keysight_E9010A.E9010A
            _DRIVER_LOOKUP['E5063A'] = Keysight_E5063A.E5063A
        except ImportError as e:
            log.warning(f"Unable to import hardware drivers, {e}")

        try:
            from qcodes.instrument_drivers.Analog import HMC_T2100
            _DRIVER_LOOKUP['HMC_T2100'] = HMC_T2100.HMC_T2100
        except ImportError as e:
            log.warning(f"Unable to import hardware drivers, {e}")

        try:
            from qcodes.instrument_drivers.Anritsu import Anritsu_MG3692C
            _DRIVER_LOOKUP['MG3692C'] = Anritsu_MG3692C.AnritsuMG3692C
        except ImportError as e:
            log.warning(f"Unable to import hardware drivers, {e}")

        try:
            from qcodes.instrument_drivers.tektronix import Keithley_2400, AWG5014
            _DRIVER_LOOKUP['Keithley_2400']   = Keithley_2400.Keithley_2400
            _DRIVER_LOOKUP['Tektronix_5014C'] = AWG5014.Tektronix_AWG5014,
        except ImportError as e:
            log.warning(f"Unable to import hardware drivers, {e}")

        try:
            from qcodes.instrument_drivers.yokogawa.GS200 import GS200  # DC source
            _DRIVER_LOOKUP['GS200'] = GS200
        except ImportError as e:
            log.warning(f"Unable to import hardware drivers, {e}")

        try:
            from qcodes.instrument_drivers.stahl import stahl
            _DRIVER_LOOKUP['Stahl'] = stahl.Stahl
        except ImportError as e:
            log.warning(f"Unable to import hardware drivers, {e}")

        try:
            from qcodes.instrument_drivers.Minicircuits.RC_SP4T import RC_SP4T
            _DRIVER_LOOKUP['RC_SP4T'] = RC_SP4T
        except ImportError as e:
            log.warning(f"Unable to import hardware drivers, {e}")

        # pylablib
        try:
            from pylablib.aux_libs.devices import Vaunix
            _DRIVER_LOOKUP['LMS'] = Vaunix.LMS
        except ImportError as e:
            log.warning(f"Unable to import hardware drivers, {e}")

        # qtrl drivers
        try:
            from qtrl.instruments.mini_circuits_attenuator import MiniCircuitsAttenuator
            _DRIVER_LOOKUP['MiniCircuitsAttenuator'] = MiniCircuitsAttenuator
        except ImportError as e:
            log.warning(f"Unable to import hardware drivers, {e}")

    return _DRIVER_LOOKUP


class InstrumentManager(ManagerBase):
    
    def __init__(self, config_file='Devices.yaml', variables={}):
        """Thin wrapper over a config_file to help with qcodes device loading.
        Device settings are only set on init and on reset_devices() call.
        """

        super().__init__(config_file, variables)
        self.connections = {}
        self.reset_devices()

    def reset_devices(self):
        self.connections = find_dev_config(self)

    def get_instrument_metadata(self, devices='all'):
        """Return snapshots of all instruments in list-like devices.
        """
        
        instrument_metadata = {}

        if devices == 'all':
            devices = list(self.connections.keys())

        for device in self.connections:
            if device in devices:
                if self[device]['model'] in qcodes_drivers or self[device]['model'] in qtrl_drivers:
                    instrument_metadata[device] = self.connections[device].snapshot()['parameters']

                    # handle qcodes VNAs that store data in parameters for some reason
                    if 'get_trace' in instrument_metadata[device].keys():
                        del instrument_metadata[device]['get_trace']

                elif self[device]['model'] in pylablib_drivers:
                    # only tested for Vaunix LMS!
                    instrument_metadata[device] = self.connections[device].get_settings()
        return instrument_metadata


def find_dev_config(config):
    """Find the devices in the config file and return a dictionary in the same
    structure as the config with all the devices.
    """

    found_devices = {}

    for key in config.keys():
        if is_device(config[key]):
            found_devices[key] = load_device(key, config[key])
        else:
            try:
                config[key].keys()
                found_devices[key] = find_dev_config(config[key])
            except (AttributeError, KeyError):
                pass
    return found_devices


def load_settings(dev, settings):
    for key in settings:
        if isinstance(settings[key],dict):
            load_settings(getattr(dev, key), settings[key])
        else:
            getattr(dev, key)(settings[key])

def is_device(config):
    """test to see if there is a valid device configuration in a given
    dictionary-like object.
    """

    try:
        keys = config.keys()
    except AttributeError:
        return False

    if len(keys) not in [2, 3]:
        return False

    drlu = driver_lookup()
    if 'model' in keys and 'settings' in keys:
        if config['model'] not in drlu:
            return False
        if isinstance(config['settings'], dict):
            return True

    return False


def load_device(name, config):
    """Loads a config of settings to a qcodes device, a unique name must be provided.
    The config must be a dictionary like object which contains only 3 things:
    model, address and settings.
    Settings are assumed to be callable functions of the qcodes object.
    """

    assert is_device(config), 'This does not appear to be a config for a device'

    if config['model'] in qcodes_drivers:
        dev = qcodes_inst(config['model'], name, config.get('address', None))
    elif config['model'] in pylablib_drivers:
        dev = pylablib_inst(config['model'])
    elif config['model'] in qtrl_drivers:
        dev = qtrl_inst(config['model'], config.get('address', None))
    else:
        log.warning(f'cannot find device! add {config["model"]} to driver list.')
        dev = None

    settings = config['settings']
    load_settings(dev, settings)

    return dev


def qcodes_inst(dev_class, name, address):
    """Ask qcodes if we already have a device of the specified name registered,
    if so return that.  Else we return a new qcodes connection to the device at the
    specified address.
    """

    # as a convenience, accept a string and look up the device in the driver_lookup table
    dev_class = qcodes_drivers.get(dev_class, dev_class)

    try:
        return VisaInstrument.find_instrument(name)
    except KeyError:
        if address is not None:
            return dev_class(name, address)
        else:
            return dev_class(name)

def pylablib_inst(dev_class):
    """
    NOTE: THIS HAS ONLY BEEN TESTED ON VAUNIX LABRICK DRIVER AND SHOULD BE UPDATED
    IF OTHER DRIVERS WILL BE USED FROM PYLABLIB

    :param dev_class: model name, should be in pylablib_drivers
    :return: device, with settings to be loaded
    """

    dev_class = pylablib_drivers.get(dev_class, dev_class)
    return dev_class()

def qtrl_inst(dev_class, address):
    """
    :param dev_class: model name, should be in qtrl_drivers dictionary in this module
    :param address: ip address of the instrument
    :return: device, with settings to be loaded in load_device
    """

    dev_class = qtrl_drivers.get(dev_class, dev_class)
    if address is not None:
        return dev_class(ip_address=address)
    else:
        return dev_class()
