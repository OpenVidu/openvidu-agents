import logging
import os
import signal
import sys

from utils.redis_utils import (
    RedisUtils,
    REDIS_PREFIX_LOCK,
    REDIS_PREFIX_ACTIVE_JOBS,
    REDIS_PREFIX_WAITING_TO_SHUTDOWN,
)


class SignalManager:
    """Utility class for capturing OS signals and safely managing agent shutdown according to active jobs."""

    def __init__(
        self,
        agent_config: object,
        agent_name: str,
        agent_process_uuid: str,
        agent_main_pid: str,
        register_signals: bool,
    ):
        if register_signals:
            signal.signal(signal.SIGTERM, self.__stop_agent)
            signal.signal(signal.SIGQUIT, self.__stop_agent)
            signal.signal(signal.SIGINT, self.__kill_agent)
        self.__agent_name = agent_name
        self.__agent_process_uuid = agent_process_uuid
        self.__agent_main_pid = agent_main_pid
        self.__redis_utils = RedisUtils(agent_config)

    def increment_active_jobs(self):
        try:
            acquired = self.__redis_utils.acquire_lock(
                REDIS_PREFIX_LOCK + self.__full_agent_id()
            )
            if not acquired:
                logging.error(
                    f"Failed to acquire lock for agent {self.__full_agent_id()}"
                )
                return
            else:
                active_jobs = self.__redis_utils.increment_and_get(
                    REDIS_PREFIX_ACTIVE_JOBS + self.__full_agent_id()
                )
                logging.info(
                    f"Active jobs increased. Agent {self.__full_agent_id()} is now processing {active_jobs} jobs"
                )
        finally:
            self.__redis_utils.release_lock(REDIS_PREFIX_LOCK + self.__full_agent_id())

    async def decrement_active_jobs(self, reason=None):
        try:
            acquired = self.__redis_utils.acquire_lock(
                REDIS_PREFIX_LOCK + self.__full_agent_id()
            )
            if not acquired:
                logging.error(
                    f"Failed to acquire lock for agent {self.__full_agent_id()}"
                )
                return
            else:

                active_jobs = self.__redis_utils.decrement_and_get(
                    REDIS_PREFIX_ACTIVE_JOBS + self.__full_agent_id()
                )
                logging.info(
                    f"Active jobs decreased. Agent {self.__full_agent_id()} is now processing {active_jobs} jobs"
                )
                if active_jobs <= 0:
                    waiting_to_shutdown: int = self.__redis_utils.get(
                        REDIS_PREFIX_WAITING_TO_SHUTDOWN + self.__full_agent_id()
                    )
                    if bool(waiting_to_shutdown):
                        logging.info(
                            f"Agent {self.__full_agent_id()} was waiting to shut down"
                        )
                        self.__shutdown_parent_process()
        finally:
            self.__redis_utils.release_lock(REDIS_PREFIX_LOCK + self.__full_agent_id())

    def can_accept_new_jobs(self) -> bool:
        try:
            acquired = self.__redis_utils.acquire_lock(
                REDIS_PREFIX_LOCK + self.__full_agent_id()
            )
            if not acquired:
                logging.error(
                    f"Failed to acquire lock for agent {self.__full_agent_id()}"
                )
                return False
            waiting_to_shutdown: int = self.__redis_utils.get(
                REDIS_PREFIX_WAITING_TO_SHUTDOWN + self.__full_agent_id()
            )
            return not bool(waiting_to_shutdown)
        finally:
            self.__redis_utils.release_lock(REDIS_PREFIX_LOCK + self.__full_agent_id())

    def __stop_agent(self, signum, frame):
        try:
            acquired = self.__redis_utils.acquire_lock(
                REDIS_PREFIX_LOCK + self.__full_agent_id()
            )
            if not acquired:
                logging.error(
                    f"Failed to acquire lock for agent {self.__full_agent_id()}"
                )
                return
            logging.info(
                f"Exit requested with {signal.Signals(signum).name}. Agent {self.__full_agent_id()} won't process new Rooms. Waiting for current Rooms to finish before shutting down",
            )
            self.__redis_utils.save(
                REDIS_PREFIX_WAITING_TO_SHUTDOWN + self.__full_agent_id(),
                1,
            )
            active_jobs = self.__redis_utils.get(
                REDIS_PREFIX_ACTIVE_JOBS + self.__full_agent_id()
            )
            if active_jobs == 0:
                logging.info(
                    f"Agent {self.__full_agent_id()} has no active jobs. Shutting down."
                )
                self.__shutdown_parent_process()
        finally:
            self.__redis_utils.release_lock(REDIS_PREFIX_LOCK + self.__full_agent_id())

    def __kill_agent(self, signum, frame):
        try:
            acquired = self.__redis_utils.acquire_lock(
                REDIS_PREFIX_LOCK + self.__full_agent_id()
            )
            if not acquired:
                logging.warning(
                    f"Failed to acquire lock for agent {self.__full_agent_id()}"
                )
            logging.info(
                f"Exit requested with {signal.Signals(signum).name}. Killing Agent {self.__full_agent_id()}"
            )
            self.__redis_utils.save(
                REDIS_PREFIX_WAITING_TO_SHUTDOWN + self.__full_agent_id(),
                1,
            )
        finally:
            self.__redis_utils.release_lock(REDIS_PREFIX_LOCK + self.__full_agent_id())
        # Make sure to always shut down
        self.__shutdown_parent_process()

    def __shutdown_parent_process(self):
        logging.info(
            f"Shutting down Agent {self.__full_agent_id()} from pid {os.getpid()}"
        )
        self.__clean_redis()
        logging.info("Redis clean up complete")
        self.__redis_utils.release_lock(REDIS_PREFIX_LOCK + self.__full_agent_id())
        try:
            os.kill(int(self.__agent_main_pid), signal.SIGTERM)
            logging.info(f"Sent SIGTERM to parent process {self.__agent_main_pid}")
        except OSError as e:
            logging.error(f"Error killing parent process: {e}")
        logging.info("Exiting own process")
        sys.exit(0)

    def __clean_redis(self):
        logging.info("Cleaning up Redis keys for agent " + self.__full_agent_id())
        self.__redis_utils.delete(REDIS_PREFIX_ACTIVE_JOBS + self.__full_agent_id())
        self.__redis_utils.delete(
            REDIS_PREFIX_WAITING_TO_SHUTDOWN + self.__full_agent_id()
        )

    def __full_agent_id(self) -> str:
        return f"{self.__agent_name}:{self.__agent_process_uuid}"
