# Copyright (c) 2018-2019, UC Regents

from multiprocessing import Process, Manager, active_children, Queue
from multiprocessing.connection import Listener
from .db_management import shelve_process
import time


class Server:
    def __init__(self, hostname, port, jobs_request_dict, shelve_response, shelve_requests):
        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("server")
        self.hostname = hostname
        self.port = port
        self.socket = None
        self.jobs_dict = jobs_request_dict
        self.shelve_requests = shelve_requests
        self.shelve_response = shelve_response

    def start(self):
        self.logger.debug("listening")
        self.socket = Listener((self.hostname, self.port), "AF_INET")

        while True:
            conn = self.socket.accept()
            address = self.socket.last_accepted
            self.logger.debug("Got connection")
            from .client import _client_connection
            process = Process(target=_client_connection,
                              args=(conn, address, self.jobs_dict, self.shelve_requests, self.shelve_response))
            process.daemon = True
            process.start()
            self.logger.debug("Started process %r", process)


if __name__ == "__main__":
    from .acquisition_management import _acquire_process

    shelve_loc = './database'

    # Creating a thread safe dictionary
    shelf_response = Manager().dict()
    job_requests = Manager().dict()

    # creating a thread safe Queue
    shelf_requests_queue = Queue()

    # start a multiprocess of the acquire_process function, passing our threadsafe dict
    acquire_process = Process(target=_acquire_process,
                              args=(job_requests, shelf_requests_queue,))

    shelf_process = Process(target=shelve_process,
                              args=(shelve_loc, shelf_requests_queue, shelf_response,))

    # Log!
    import logging
    logging.basicConfig(level=logging.INFO)

    # create our server, accepting inputs from everyone at port 9000
    server = Server("127.0.0.1", 9000, job_requests, shelf_response, shelf_requests_queue)

    try:
        # start it up
        acquire_process.start()
        shelf_process.start()
        logging.info("Listening")
        server.start()
    except:
        logging.exception("Unexpected exception")
    finally:
        logging.info("Shutting down")
        for process in active_children():
            logging.info("Shutting down process %r", process)
            process.terminate()
            process.join()
            # shelf.close()

    logging.info("All done")

