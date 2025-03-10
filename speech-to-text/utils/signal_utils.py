import logging
import signal


class SignalManager:
    def __init__(self, logger: logging.Logger):
        self.__can_accept = True
        self.__logger = logger
        signal.signal(signal.SIGTERM, self._reject_new_jobs)
        signal.signal(signal.SIGQUIT, self._reject_new_jobs)
        signal.signal(signal.SIGINT, self._reject_new_jobs)

    def can_accept_new_jobs(self):
        return self.__can_accept

    def _reject_new_jobs(self, signum, frame):
        self.__logger.info(f"exit requested, rejecting new jobs. signum: {signum}")
        self.__can_accept = False
