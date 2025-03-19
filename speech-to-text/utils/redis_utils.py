import time
import redis
import logging
from redis.sentinel import Sentinel
import redis.lock
from redis.retry import Retry
from redis.exceptions import TimeoutError, ConnectionError, LockNotOwnedError
from redis.backoff import ExponentialBackoff

REDIS_PREFIX_LOCK: str = "ov_agent_lock:"
REDIS_PREFIX_ACTIVE_JOBS: str = "ov_agent_active_jobs:"
REDIS_PREFIX_WAITING_TO_SHUTDOWN: str = "ov_agent_waiting_to_shutdown:"


class RedisUtils:
    """Utility class for managing Redis interactions."""

    __TTL_MS: int = 10000
    __locks = {}

    def __init__(self, agent_config):
        if "sentinel_addresses" in agent_config["redis"]:
            logging.info("Using Redis Sentinel")
            sentinels = []
            for sentinel_address in agent_config["redis"]["sentinel_addresses"]:
                host, port = sentinel_address.split(":")
                sentinels.append((host, int(port)))
            sentinel = self.__redis_client = Sentinel(
                sentinels=sentinels,
                sentinel_kwargs={
                    "password": agent_config["redis"]["sentinel_password"]
                },
                db=agent_config["redis"]["db"],
                password=agent_config["redis"]["sentinel_password"],
            )
            self.__redis_client = sentinel.master_for(
                agent_config["redis"]["sentinel_master_name"],
                redis_class=redis.Redis,
            )
        else:
            logging.info("Using Redis standalone")
            redis_address = agent_config["redis"]["address"]
            host, port = redis_address.split(":")
            self.__redis_client = redis.Redis(
                host=host,
                port=port,
                db=agent_config["redis"]["db"],
                username=agent_config["redis"]["username"],
                password=agent_config["redis"]["password"],
                retry=Retry(ExponentialBackoff(cap=10, base=1), -1),  # Retry forever
                retry_on_error=[ConnectionError, TimeoutError, ConnectionResetError],
            )

        self.__test_connection()

    def __test_connection(self):
        backoff = 1  # Initial backoff in seconds
        max_backoff = 10  # Maximum backoff in seconds
        attempt = 1
        while True:
            try:
                # Test connection with ping
                response = self.__redis_client.ping()
                if response:
                    logging.info("Successfully connected to Redis")
                break
            except Exception as e:
                logging.error(
                    f'Connection with Redis failed (attempt {attempt}): "{str(e)}"'
                )
                logging.error(f"Retrying in {backoff} seconds...")
                time.sleep(backoff)
                # Exponential backoff with maximum cap
                backoff = min(backoff * 2, max_backoff)
                attempt += 1

    def save(self, key, value):
        self.__redis_client.set(key, value)

    def get(self, key):
        return self.__redis_client.get(key)

    def delete(self, key):
        self.__redis_client.delete(key)

    def exists(self, key):
        return self.__redis_client.exists(key)

    def increment_and_get(self, key):
        return self.__redis_client.incr(key)

    def decrement_and_get(self, key):
        return self.__redis_client.decr(key)

    def acquire_lock(self, key: str, ttl: int = 0) -> bool:
        if ttl == 0:
            ttl = self.__TTL_MS
        lock: redis.lock.Lock = self.__redis_client.lock(
            name=key,
            timeout=ttl,  # The time-to-live (TTL) for the lock in milliseconds
            sleep=0.1,  # Time between retries to acquire the lock
            blocking=True,  # Block the operation until the lock is acquired
            blocking_timeout=None,  # Infinite retries until the lock is acquired
            lock_class=redis.lock.Lock,
        )
        acquired: bool = lock.acquire()
        if acquired:
            self.__locks[key] = lock
        return acquired

    def release_lock(self, key: str) -> None:
        if key in self.__locks:
            lock: redis.lock.Lock = self.__locks[key]
            if lock:
                del self.__locks[key]
                try:
                    lock.release()
                except LockNotOwnedError as e:
                    logging.warning(f"Lock for key {key} not owned: {str(e)}")
            else:
                logging.warning(f"Lock for key {key} not found")
