import logging
import os
import signal
import threading
import asyncio
import time

from .redis_utils import (
    RedisUtils,
    REDIS_PREFIX_LOCK,
    REDIS_PREFIX_ACTIVE_JOBS,
    REDIS_PREFIX_WAITING_TO_SHUTDOWN,
    REDIS_PREFIX_SHUTDOWN_FLAG,
)


class SignalManager:
    """Utility class for capturing OS signals and safely managing agent shutdown according to active jobs."""

    def __init__(
        self,
        agent_config: object,
        agent_name: str,
        agent_process_uuid: str,
        agent_main_pid: str,
        is_main_process: bool,
    ):
        self.__agent_name = agent_name
        self.__agent_process_uuid = agent_process_uuid
        self.__agent_main_pid = agent_main_pid
        self.__redis_utils = RedisUtils(agent_config)

        if is_main_process:
            # Register synchronous signal handlers that will schedule async functions
            signal.signal(signal.SIGTERM, self.__stop_agent_sync)
            signal.signal(signal.SIGQUIT, self.__stop_agent_sync)
            signal.signal(signal.SIGINT, self.__kill_agent_sync)
            # Start the shutdown watcher thread
            threading.Thread(
                target=self.__shutdown_watcher,
                args=(),
                daemon=True,
            ).start()

    def __shutdown_watcher(self):
        while True:
            if self.__redis_utils.get(
                REDIS_PREFIX_SHUTDOWN_FLAG + self.__full_agent_id()
            ):
                logging.info(
                    f"Agent {self.__full_agent_id()} received shutdown signal, exiting..."
                )
                self.__clean_redis()
                os._exit(0)
            time.sleep(1)

    def __stop_agent_sync(self, signum, frame):
        """Synchronous signal handler that schedules the async __stop_agent function"""
        logging.info(
            f"Signal {signal.Signals(signum).name} received, scheduling async handler"
        )

        # Try to get the current event loop and schedule the async function
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.__stop_agent_async(signum, frame))
        except RuntimeError:
            # No event loop running, use a thread to run the async function
            logging.warning(
                "No asyncio event loop found, using thread for signal handling"
            )
            threading.Thread(
                target=self.__run_async_in_thread,
                args=(self.__stop_agent_async, signum, frame),
                daemon=True,
            ).start()

    def __kill_agent_sync(self, signum, frame):
        """Synchronous signal handler that schedules the async __kill_agent function"""
        logging.info(
            f"Signal {signal.Signals(signum).name} received, scheduling async handler"
        )

        # Try to get the current event loop and schedule the async function
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.__kill_agent_async(signum, frame))
        except RuntimeError:
            # No event loop running, use a thread to run the async function
            logging.warning(
                "No asyncio event loop found, using thread for signal handling"
            )
            threading.Thread(
                target=self.__run_async_in_thread,
                args=(self.__kill_agent_async, signum, frame),
                daemon=True,
            ).start()

    def __run_async_in_thread(self, async_func, *args):
        """Helper to run async function in a new thread with its own event loop"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(async_func(*args))
        except Exception as e:
            logging.error(f"Error running async function in thread: {e}")
        finally:
            try:
                loop.close()
            except:
                pass

    def increment_active_jobs(self):
        """Increment the active jobs counter for this agent instance"""
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
        """Decrement the active jobs counter for this agent instance"""
        logging.info(
            f"Decrementing active jobs for agent {self.__full_agent_id()}. Reason: {reason}"
        )
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
                            f"Agent {self.__full_agent_id()} was waiting to shut down. Shutting down now."
                        )
                        await self.__shutdown_parent_process()
        finally:
            self.__redis_utils.release_lock(REDIS_PREFIX_LOCK + self.__full_agent_id())

    def is_waiting_to_shut_down(self) -> bool:
        """Check if this agent instance is flagged as waiting to shut down"""
        try:
            acquired = self.__redis_utils.acquire_lock(
                REDIS_PREFIX_LOCK + self.__full_agent_id()
            )
            if not acquired:
                logging.error(
                    f"Failed to acquire lock for agent {self.__full_agent_id()}"
                )
                # If we can't acquire the lock, let's be conservative and assume we cannot take new jobs
                return True
            waiting_to_shutdown: int = self.__redis_utils.get(
                REDIS_PREFIX_WAITING_TO_SHUTDOWN + self.__full_agent_id()
            )
            return bool(waiting_to_shutdown)
        finally:
            self.__redis_utils.release_lock(REDIS_PREFIX_LOCK + self.__full_agent_id())

    async def __stop_agent_async(self, signum, frame):
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
                f"Exit requested with {signal.Signals(signum).name}. Agent {self.__full_agent_id()} won't process new Rooms. Waiting for current jobs to finish before shutting down",
            )
            self.__redis_utils.save(
                REDIS_PREFIX_WAITING_TO_SHUTDOWN + self.__full_agent_id(),
                1,
            )
            active_jobs = self.__redis_utils.get(
                REDIS_PREFIX_ACTIVE_JOBS + self.__full_agent_id()
            )
            if active_jobs == 0 or active_jobs is None:
                logging.info(
                    f"Agent {self.__full_agent_id()} has no active jobs. Shutting down immediately."
                )
                await self.__shutdown_parent_process()
            else:
                logging.info(
                    f"Agent {self.__full_agent_id()} has still {active_jobs} active jobs. Waiting for them to finish before shutting down."
                )
        finally:
            self.__redis_utils.release_lock(REDIS_PREFIX_LOCK + self.__full_agent_id())

    async def __kill_agent_async(self, signum, frame):
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
        await self.__shutdown_parent_process()

    async def __shutdown_parent_process(self):
        OWN_PID = os.getpid()

        logging.info(f"Shutting down Agent {self.__full_agent_id()} from pid {OWN_PID}")

        await asyncio.sleep(0)  # yield control so logs can flush
        # Flush logs
        for handler in logging.root.handlers:
            handler.flush()

        if OWN_PID == int(self.__agent_main_pid):
            logging.info(
                f"Main process {self.__agent_main_pid} is shutting down. Exiting immediately."
            )
            self.__clean_redis()
            logging.info("Redis clean up complete")
            # Release the lock before exiting, as os._exit(0) will bypass any upper finally blocks
            self.__redis_utils.release_lock(REDIS_PREFIX_LOCK + self.__full_agent_id())
            os._exit(0)
        else:
            logging.info(
                f"Child process {OWN_PID} is shutting down. Notifying main process through Redis."
            )
            self.__redis_utils.save(
                REDIS_PREFIX_SHUTDOWN_FLAG + self.__full_agent_id(),
                1,
            )
            logging.info("Exiting own child process")
            os._exit(0)

    def __clean_redis(self):
        logging.info("Cleaning up Redis keys for agent " + self.__full_agent_id())
        self.__redis_utils.delete(REDIS_PREFIX_ACTIVE_JOBS + self.__full_agent_id())
        self.__redis_utils.delete(
            REDIS_PREFIX_WAITING_TO_SHUTDOWN + self.__full_agent_id()
        )
        self.__redis_utils.delete(REDIS_PREFIX_SHUTDOWN_FLAG + self.__full_agent_id())

    def __full_agent_id(self) -> str:
        return f"{self.__agent_name}:{self.__agent_process_uuid}"
