# Copyright (c) 2018-2019, UC Regents

import shutil
from ..managers import KeysightDACManager, PulseManager, AlazarADCManager, VariableManager, MetaManager
from ..sequencer import SequenceWrapper
import traceback
import time
from ..utils.util import update


def _acquire_process(jobs, shelf_requests_queue, default_config='./Config/DefaultConfig/', current_config='./Config/CurrentConfig/'):
    """This manages the acquisition process which happens in the server"""
    # Process status:
    # queued - waiting to be run
    # running - Running
    # finished - finished, no error
    # error - finished, error

    # enable logging
    import logging
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger("Acquire-Process")
    logger.debug(f"Default Config Location: {default_config}")
    logger.debug(f"Current Config Location: {current_config}")

    # These are the key allowed in a measurement request, not including the Sequence
    allowed_keys = {'variables',
                    'DAC',
                    'ADC',
                    'pulses',
                    # 'devices'
                    }
    logger.info("Starting MetaManager")

    # Copy the default config into the current directory
    # shutil.copy(default_config + 'Devices.yaml', current_config)
    shutil.copy(default_config + 'DAC.yaml', current_config)
    shutil.copy(default_config + 'Pulses.yaml', current_config)
    shutil.copy(default_config + 'ADC.yaml', current_config)
    shutil.copy(default_config + 'Variables.yaml', current_config)

    var = VariableManager(current_config+'Variables.yaml')
    config = MetaManager({'variables': var,
                          # 'devices': QcodesManager(current_config+'Devices.yaml', var),
                          'DAC': KeysightDACManager(current_config+'DAC.yaml'),
                          'pulses': PulseManager(current_config+'Pulses.yaml', var),
                          'ADC': AlazarADCManager(current_config+'ADC.yaml', var)
                          })

    device_reload = False

    total_acquisitions = 1
    old_job_key = None
    while True:

        logger.info(f"Starting Loop - {total_acquisitions}")
        # Each loop is one complete acquisition

        # an acquisition amounts to copying the default config to the current directory,
        # getting a job out of the jobs dict, updating the current config with the job config,
        # constructing the sequence, writing it, acquiring, and putting the results back in the
        # job dict.

        # Copy the default config into the current directory
        shutil.copy(default_config + 'Devices.yaml', current_config)
        shutil.copy(default_config + 'DAC.yaml', current_config)
        shutil.copy(default_config + 'Pulses.yaml', current_config)
        shutil.copy(default_config + 'ADC.yaml', current_config)
        shutil.copy(default_config + 'Variables.yaml', current_config)
        time.sleep(0.25)
        config.load()

        # if the last job had device settings, reload back to default
        # if device_reload:
        #     logger.info(f"Reloading devices to default")
        #     try:
        #         config.devices.reset_devices()
        #     except Exception as e:
        #         logger.error(f"Hardware was not set, an exception occurred{e}")
        #     device_reload = False
        # config.devices.reset_devices()

        # Delete the last measurement from the job dictionary
        if old_job_key is not None:
            del jobs[old_job_key]

        # now we are ready to start a new Job, we have to get one
        # jobs in the dictionary have a key with 3 entries: (ID, Status, Priority)
        job_key = (None, None, None)
        while str(job_key[1]).lower() != "queued":
            all_jobs = list(jobs.keys())

            # if no jobs found, wait half a second and try again
            if len(all_jobs) == 0:
                time.sleep(0.5)
                continue

            # sort by priority, with a 100000 cost if status is not 0, IE: not complete
            # so this means if the status is not 0, don't run it
            all_jobs.sort(key=lambda x: x[2] + 100000 * (x[1] != 'queued'), reverse=False)
            job_key = all_jobs[0]

            # if the found key isn't one that needs to be run, delay have a second and look for another job
            if str(job_key[1]).lower() != "queued":
                time.sleep(0.5)
            elif str(job_key[1]).lower() == 'shutdown':
                # if the job status is shutdown, kill the server by returning
                logger.info(f"Job with status 'shutdown' found, killing server")
                return

        # update the key to show processing is happening
        new_job_key = list(job_key)
        new_job_key[1] = 'running'

        # pop the current job off the dictionary
        current_job = jobs.get(job_key)

        jobs[tuple(new_job_key)] = jobs.pop(job_key)

        logger.info(f"\nNew Job {new_job_key}\n  keys: {list(current_job.keys())}")

        current_job.update({'errors': [],
                            'measurement': None})

        # now we need to update the current config with the settings from the Job
        for key in allowed_keys:
            manager = config.__getattribute__(key)

            # job_cfg has keys which are tuples ('filename.yaml', 'variables') for example
            # we grab the first entry which has a (*, key) which matches the allowed_keys
            k = [k for k in current_job.keys() if k[1] == key]

            if len(k) > 0:
                cur_dict = current_job.pop(k[0], {})
            else:
                cur_dict = {}
            # we use that first entry to update the current manager we are working on
            update(manager._config_raw, cur_dict)

            # and save it
            manager.save()
            manager.load()

            # # if we have device settings, set them and tell the loop to reset back to default on the next loop
            # if key == 'devices' and len(cur_dict) != 0:
            #     logger.info(f"Setting Hardware - will reload next loop")
            #     try:
            #         config.devices.reset_devices()
            #     except Exception as e:
            #         logger.error(f"Hardware was not set, an exception occurred{e}")
            #         current_job['errors'].append(('Hardware was not set, an exception occurred.', e))
            #     device_reload = True

        config.load()
        logger.info(f"Constructing Sequence")
        sequence = current_job.get('sequence', None)
        if not isinstance(sequence, SequenceWrapper):
            current_job['errors'].append(('Sequence not defined or not valid.', None))
            logger.error(f"Sequence was not defined in current job {new_job_key}")

        try:
            compiled_seq = sequence.compile(config.pulses)
        except Exception as e:
            current_job['errors'].append(("Sequence failed to compile.", e))
            logger.error(f"Sequence failed to compile for job {new_job_key}\n{e}")

        logger.info(f"Writing Sequence")
        try:
            config.write_sequence(compiled_seq)
        except Exception as e:
            current_job['errors'].append(("Sequence failed to write.", e))
            logger.error(f"Sequence failed to write for job {new_job_key}\n{e}")

        logger.info(f"Acquiring")
        try:
            current_job['measurement'] = config.acquire(compiled_seq)
        except Exception as e:
            # record the measurement until it failed
            current_job['measurement'] = config.ADC.measurement
            # record the errors
            current_job['errors'].append(("Acquisition failed.", e))
            # log that shit
            logger.error(f"Acquisition Failed for job {new_job_key}\n{e}")
            logger.error(f"Acquisition Failed for job {new_job_key}\n{traceback.print_tb(e.__traceback__)}")
        logger.info(f"Acquisition complete.")
        old_job_key = tuple(new_job_key)

        # we must always have continuity in the keys of the dictionary
        # in order to achieve this, we put the final result in the location of the
        # old key, which will lead to a race condition where the user
        # might request to get the not completed run and actually receive the completed run
        # this is a preferable situation to asking for the complete run and getting the
        # not completed one.
        #
        # jobs[old_job_key] = current_job
        # # update the key to show processing is happening
        # if len(current_job['errors']) != 0:
        #     new_job_key[1] = 'error'
        # else:
        #     new_job_key[1] = 'finished'

        logger.info(f"Job {old_job_key} complete, returning job to dictionary. ")

        # # put the current job back in the dictionary
        # jobs[tuple(new_job_key)] = jobs[old_job_key]

        # Stuff it into the shelve
        shelf_requests_queue.put(('set', old_job_key[0], current_job))

        total_acquisitions += 1
