# Copyright (c) 2018-2019, UC Regents

import shelve


def shelve_process(shelve_loc, job_input_queue, job_output_dict):
    """Manages the interface between the remote servers and a shelve database which is used to store
    all complete measurements.
        Accepts:
            shelve_loc - File location, do not include a file extension

            job_input_queue - a multiprocessing.Queue object which will be used to define
                operations related to the shelve.
                Commands allowed are:
                    ('get', ID) - loads the job with ID number into the job_output_queue
                                if the ID is not found, then None is returned in it's place
                    ('set', ID, payload) - sets the payload into the shelve
                    ('list', ) - list all available saved measurements
                    ('clear', ) - clears the job_output_dict

            job_output_queue - a thread safe dictionary which will contain the results of the job_input_queue
                Keys will only be the job ID number, or
                'list' - which will contain a list of all available ID numbers in the shelf

    """
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"shelve_storage at {shelve_loc}")

    with shelve.open(shelve_loc) as shelf:
        while True:
            request = job_input_queue.get()
            if len(request) < 1:
                logger.error(f'Invalid command received {request}')
                continue

            if request[0] == 'get':
                if len(request) != 2:
                    logger.error(f'Incomplete command received {request}')
                    continue
                if str(request[1]) not in shelf:
                    # logger.error(f'{request[1]} not found in shelf')
                    job_output_dict[request[1]] = None
                    continue
                logger.info(f'{request} Received')
                job_output_dict[request[1]] = shelf[str(request[1])]
                logger.info(f'{request[1]} Put into dict')

            elif request[0] == 'set':
                if len(request) != 3:
                    logger.error(f'Incomplete command received {request}')
                    continue

                shelf[str(request[1])] = request[2]

            elif request[0] == 'list':
                logger.info(f'List Request {request}')
                job_output_dict['list'] = list(shelf.keys())

            elif request[0] == 'clear':
                for key in job_output_dict:
                    del job_output_dict[key]

            else:
                logger.error(f'Incomplete command received {request}')

