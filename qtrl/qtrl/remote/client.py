# Copyright (c) 2018-2019, UC Regents

import time
from multiprocessing.connection import Client
import warnings
from ..utils.util import update


class Connection:
    def __init__(self, config, ip='127.0.0.1', port=9000):
        """A connection to a server which controls measurements on the experimental setup"""
        self._ip = ip
        self._port = port
        self._config = config
        self._connect()
        self.measurements = []

    def _connect(self):
        self._socket = Client((self._ip, self._port))

    def acquire(self, seq, n_reps=None, n_batches=None, priority=0, blocking=True):
        """Submit a job to the experiment.
        Accepts:
            seq - a qtrl.sequencer.SequenceWrapper object which describes the sequence to be measured
            n_reps - Int - optional, if provided, acquire this many ensemble measurements
                     if not provided use the value in the 'ADC' local Config
                     if 'ADC' is not provided, use the default server 'ADC' Config
            n_batches - Int - optional, how many batches of the ensemble measurements to make.
                        Example: 1000 reps, 10 batches = 10,000 total ensemble measurements
                        behavior same as n_reps
            priority - Int - defaults to 0 - priority to be run on the server
                       lower numbers means the job will be prioritized over all other jobs
            blocking - Boolean - defaults to True - see full description of the qtrl.remote.Connection Class
        Returns:
            Either:
                jobID - float - a unique ID number that is used to identify the measurement on the server
                measurement - dictionary - a dictionary with all of the results of the measurement
        """
        cfg_dict = self._config.config
        cfg_dict['sequence'] = seq

        # If we ADC is not in the config, we need to provide a temp dictionary for it so that
        # the server has the information about n_reps and n_batches
        ADC_key = [x for x in self._config.config.keys() if x[1] == 'ADC']
        ADC_key = ADC_key[0] if len(ADC_key) == 1 else (None, 'ADC')

        if ADC_key not in cfg_dict:
            cfg_dict[ADC_key] = {}

        acquisition_dict = {'acquisition_settings': {}}

        if n_reps is not None:
            acquisition_dict['acquisition_settings']['n_reps'] = int(n_reps)

        if n_batches is not None:
            acquisition_dict['acquisition_settings']['n_batches'] = int(n_batches)

        # if they are not specified, do not update
        if n_reps is not None or n_batches is not None:
            update(cfg_dict[ADC_key], acquisition_dict)

        # This is the end of the n_reps and n_batches update

        # Now we can send our config to the server
        try:
            self._socket.send(('submit', priority, cfg_dict))
            ID = self._socket.recv()[1]
        except ConnectionAbortedError:
            warnings.warn("Connection lost, attempting to reconnect...")
            self._connect()
            return self.acquire(seq, n_reps, n_batches, priority, blocking)

        self.measurements.insert(0, ID)
        if blocking:
            time.sleep(1)
            return self.get(ID, blocking=blocking)
        else:
            return ID

    def status(self, jobID):
        try:
            self._socket.send(('status', jobID))
            result = self._socket.recv()
        except ConnectionAbortedError:
            warnings.warn("Connection lost, attempting to reconnect...")
            self._connect()
            return self.status(jobID)

        if result == 'Job not found':
            raise KeyError("Job not found.")
        return result

    def queued(self):
        try:
            self._socket.send(('queue',))
            result = self._socket.recv()
        except ConnectionAbortedError:
            warnings.warn("Connection lost, attempting to reconnect...")
            self._connect()
            return self.queued()

        return result

    def completed(self):
        try:
            self._socket.send(('completed', ))
            result = self._socket.recv()
        except ConnectionAbortedError:
            warnings.warn("Connection lost, attempting to reconnect...")
            self._connect()
            return self.completed()
        return result

    def get(self, jobID, blocking=False):
        try:
            self._socket.send(('get', jobID))
            result = self._socket.recv()
        except ConnectionAbortedError:
            warnings.warn("Connection lost, attempting to reconnect...")
            self._connect()
            return self.get(jobID, blocking)

        if result == 'Job not found':
            raise KeyError("Job not found.")
        elif result == 'Job running' and blocking:
            time.sleep(0.5)
            return self.get(jobID, blocking)
        if isinstance(result, dict):
            errors = result.get('errors', [])
            if len(errors) > 0:
                warnings.warn(f"Errors occurred during acquisition {errors}")
        return result


def _job_ID():
    """Generate a unique ID"""
    return time.time()


def _client_connection(connection, address, jobs_request_dict, shelve_requests, shelve_response):
    """This handles incoming connections, parses commands, and allows for a remote interface with
    the job dictionary.
        requests from connections should be of the form (cmd, optional payload)
        Allowed commands are
            ('queue', ) - Return a list of the current jobs in the job_request_dict
            ('completed', ) - Returns a list of completed jobs
            ('status', ID) - Returns the status a given ID
                    returns 'Job not found' if job is not found
            ('submit', job) - returns a unique ID number which will be used to idetify the submitted job
            ('get', ID) - returns the result of the job if completed
                    returns 'Job Running' if not completed
                    returns 'Job not found' if job is not found

    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("process-%r" % (address,))
    # connection.settimeout(1)
    try:
        logger.info("Connected %r at %r", connection, address)
        while True:
            try:
                request = connection.recv()
            except EOFError:
                logger.info("Connection lost, closing.")
                break

            if request is None:
                continue

            if not isinstance(request, tuple) or len(request) < 1:
                logger.warning("Received data which was not a tuple! %r", request)
                continue

            cmd = request[0]
            if cmd == 'queue':
                logger.info("queue request")
                connection.send(list(jobs_request_dict.keys()))

            elif cmd == 'completed':
                logger.info("complete request")
                shelve_requests.put(('list', ))

                # Try 100 times to get the respsonse to this command, if it fails return None
                for _ in range(100):
                    logger.info(f"{shelve_response.keys()}")
                    if 'list' in shelve_response:
                        resp = shelve_response.pop('list')
                        logger.info(f"list response - {resp}")
                        connection.send(resp)
                        break
                    else:
                        time.sleep(0.1)

                # else statements after for loops run only when the loop doesn't break
                else:
                    logger.info(f"list response - {shelve_response.keys()}")
                    connection.send(None)

            elif cmd == 'status':
                ID = request[1]
                logger.info(f"status request - {ID}")

                # first check if it is in the job queue
                keys = [i for i in jobs_request_dict.keys() if i[0] == ID]
                if len(keys) != 0:
                    connection.send(keys[0])

                shelve_requests.put(('get', ID))
                # Try 100 times to get the response to this command, if it fails return None
                resp = None
                for _ in range(100):
                    if 'list' in shelve_response:
                        resp = shelve_response.pop('list')
                        print(resp)
                        break
                    else:
                        time.sleep(0.1)

                resp = 'Job not found' if resp is None else resp

                connection.send(resp)

            elif cmd == 'submit':
                ID = _job_ID()
                jobs_request_dict[(ID, 'queued', request[1])] = request[2]
                logger.info(f"submit request - {(ID, 'queued', request[1])}")
                connection.send(("submitted", ID))

            elif cmd == 'get':
                ID = request[1]

                shelve_requests.put(('get', ID))
                # Try 100 times to get the response to this command, if it fails return None
                resp = None
                for _ in range(50):
                    if ID in shelve_response:
                        resp = shelve_response.pop(ID)
                        break
                    else:
                        time.sleep(0.1)

                if resp is not None:
                    connection.send(resp)
                else:
                    keys = [i for i in jobs_request_dict.keys() if i[0] == ID]
                    if len(keys) == 0:
                        connection.send('Job not found')
                    else:
                        connection.send("Job running")

    except:
        logger.exception("Problem handling request, client connection terminated")
