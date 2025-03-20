import logging
import os
import re
import sys
import yaml
from typing import TypeVar
from enum import Enum
from dotenv import load_dotenv

T = TypeVar("T", bound=Enum)


class ConfigLoader:
    """Utility class to load agent configuration from environment variables or a YAML file."""

    def load_agent_config(self) -> tuple[object, str]:
        agent_config: object = None
        agent_name: str = None

        # Try to load environment variables from file
        self.__load_env_vars_from_file()

        # Try to load configuration from env vars
        config_body = os.environ.get("AGENT_CONFIG_BODY")
        if config_body is not None:
            try:
                agent_config = yaml.safe_load(config_body)
            except yaml.YAMLError as exc:
                logging.error(
                    "Error loading configuration from AGENT_CONFIG_BODY env var"
                )
                logging.error(exc)
                exit(1)
            try:
                agent_name = agent_config["agent_name"]
            except Exception as exc:
                logging.error(
                    'Property "agent_name" is missing. It must be defined when providing agent configuration through env var AGENT_CONFIG_BODY'
                )
                exit(1)
            if agent_config["agent_name"] is None:
                logging.error(
                    'Property "agent_name" is missing. It must be defined when providing agent configuration through env var AGENT_CONFIG_BODY'
                )
                exit(1)
            agent_name = agent_config["agent_name"]
        else:
            config_file = os.environ.get("AGENT_CONFIG_FILE")
            if config_file is not None:
                if not self.__is_agent_config_file(
                    os.path.dirname(config_file), os.path.basename(config_file)
                ):
                    logging.error(
                        f"Env var AGENT_CONFIG_FILE set to {config_file}, but file is not a valid agent configuration file. It must exist and be named agent-AGENT_NAME.yml"
                    )
                    exit(1)
            else:
                # If env vars are not defined, try to find the config file in the current working directory
                # This is useful for development purposes

                # Possible paths for the config file are any file agent-AGENT_NAME.y(a)ml
                # in the current working directory or in the location of the python entrypoint
                # script, being AGENT_NAME a unique string identifying the agent.
                cwd = os.getcwd()
                for f in os.listdir(cwd):
                    # First search in the same location as this file
                    if self.__is_agent_config_file(cwd, f):
                        config_file = os.path.join(cwd, f)
                        break
                if config_file is None:
                    # If not found, search in the location of the python entrypoint script
                    try:
                        entry_point_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
                        for f in os.listdir(entry_point_dir):
                            if self.__is_agent_config_file(entry_point_dir, f):
                                config_file = os.path.join(entry_point_dir, f)
                                break
                    except Exception as e:
                        logging.error(
                            "Error searching for the agent-AGENT_NAME.yml file in the location of the python entrypoint script:"
                        )
                        logging.error(e)
                        pass

            if config_file is None:
                logging.error(
                    "\nAgent configuration not found. One of these must be defined:\n    - env var AGENT_CONFIG_FILE with the path to the YAML configuration file.\n    - env var AGENT_CONFIG_BODY with the configuration YAML as a string.\n    - A file agent-AGENT_NAME.yml in the current working directory"
                )
                exit(1)

            with open(config_file) as stream:
                try:
                    agent_config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    logging.error(f"Error loading configuration file {config_file}")
                    logging.error(exc)
                    exit(1)

            # Load agent name from the file name
            agent_name = os.path.basename(config_file).split(".")[0].split("agent-")[1]
            # If property "agent_name" of the config file is defined check that it matches the value inside the file name
            agent_name_in_config_file = None
            try:
                agent_name_in_config_file = agent_config["agent_name"]
            except Exception as exc:
                # Do nothing
                pass
            if (
                agent_name_in_config_file is not None
                and agent_name_in_config_file != agent_name
            ):
                logging.error(
                    f"Agent name is defined as \"{agent_config['agent_name']}\" inside configuration file {config_file} and it does not match the value of the file name \"{agent_name}\""
                )
                exit(1)

        if not self.__prop_exists_and_is_string_not_empty(agent_config, "api_key"):
            if not os.environ.get("LIVEKIT_API_KEY"):
                logging.error(
                    "api_key not defined in agent configuration or LIVEKIT_API_KEY env var"
                )
                sys.exit(1)
            agent_config["api_key"] = os.environ["LIVEKIT_API_KEY"]
        if not self.__prop_exists_and_is_string_not_empty(agent_config, "api_secret"):
            if not os.environ.get("LIVEKIT_API_SECRET"):
                logging.error(
                    "api_secret not defined in agent configuration or LIVEKIT_API_SECRET env var"
                )
                exit(1)
            agent_config["api_secret"] = os.environ["LIVEKIT_API_SECRET"]
        if not self.__prop_exists_and_is_string_not_empty(agent_config, "ws_url"):
            if not os.environ.get("LIVEKIT_URL"):
                logging.error(
                    "ws_url not defined in agent configuration or LIVEKIT_URL env var"
                )
                exit(1)
            agent_config["ws_url"] = os.environ["LIVEKIT_URL"]

        agent_config = self.__load_redis_config(agent_config)

        return agent_config, agent_name

    def __load_env_vars_from_file(self) -> None:
        ENV_VARS_FILE = "ENV_VARS_FILE"
        if not ENV_VARS_FILE in os.environ:
            logging.debug(ENV_VARS_FILE + " not defined")
            return
        env_vars_file_path = os.getenv(ENV_VARS_FILE)
        if not os.path.isfile(env_vars_file_path):
            logging.warning(
                ENV_VARS_FILE + " defined but is not a file: " + env_vars_file_path
            )
            return
        if not os.access(env_vars_file_path, os.R_OK):
            logging.warning(
                ENV_VARS_FILE + " defined but is not readable: " + env_vars_file_path
            )
            return
        # Environment variable file exists and is readable
        logging.info("Loading environment variables from file: " + env_vars_file_path)
        if not load_dotenv(env_vars_file_path):
            logging.warning("No env vars available at file: " + env_vars_file_path)

    # Redis configuration must be:
    #
    # For standalone Redis using YAML configuration:
    # redis:
    #     address: 127.0.0.1:6379
    #     db: 0
    #     username: user
    #     password: pass
    #
    # For standalone Redis using environment variables:
    # REDIS_ADDRESS=127.0.0.1:6379
    # REDIS_USERNAME=user
    # REDIS_PASSWORD=pass
    # REDIS_DB=0
    #
    # For Redis Sentinel using YAML configuration:
    # redis:
    #     sentinel_master_name: openvidu
    #     sentinel_addresses:
    #         - sentinel.address.1:26379
    #         - sentinel.address.2:26379
    #     sentinel_username: user
    #     sentinel_password: pass
    #
    # For Redis Sentinel using environment variables:
    # REDIS_SENTINEL_MASTER_NAME=openvidu
    # REDIS_SENTINEL_ADDRESSES=sentinel.address.1:26379,sentinel.address.2:26379
    # REDIS_SENTINEL_USERNAME=user
    # REDIS_SENTINEL_PASSWORD=pass
    #
    def __load_redis_config(self, agent_config: object) -> tuple[object, str]:
        if "redis" not in agent_config:
            # Try to load directly from the agent configuration
            self.__load_redis_config_from_env(agent_config)

        try:
            self.__check_redis_config(agent_config)
        except ValueError as e:
            logging.error(e)
            exit(1)

        return agent_config

    def __check_redis_config(self, agent_config: object) -> None:
        """
        Validates Redis configuration either from agent_config or environment variables.

        Requirements if agent_config.redis exists:
        - redis.db must be defined
        - Either redis.address OR redis.sentinel_addresses must be defined
        - If using sentinel_addresses, must have sentinel_master_name and sentinel_password
        - If using address, must have password

        Requirements if agent_config.redis doesn't exist:
        - Environment variables must be set appropriately for either standalone Redis connection
        or Redis Sentinel connection

        Raises:
            ValueError: If configuration requirements are not met
        """
        if "redis" in agent_config:
            redis_config = agent_config["redis"]

            if not "db" in redis_config:
                raise ValueError(
                    "Missing required field in agent configuration: redis.db"
                )

            # The attribute exists and is not None or empty string
            has_address = self.__prop_exists_and_is_string_not_empty(
                redis_config, "address"
            )
            has_sentinel = "sentinel_addresses" in redis_config

            if not (has_address or has_sentinel):
                raise ValueError(
                    "Missing required field in agent configuration: either redis.address or redis.sentinel_addresses must be defined"
                )

            if has_sentinel:
                if not self.__prop_exists_and_is_string_not_empty(
                    redis_config, "sentinel_master_name"
                ):
                    raise ValueError(
                        "Missing required field in agent configuration: redis.sentinel_master_name"
                    )
                if not self.__prop_exists_and_is_string_not_empty(
                    redis_config, "sentinel_password"
                ):
                    raise ValueError(
                        "Missing required field in agent configuration: redis.sentinel_password"
                    )

            if has_address and not self.__prop_exists_and_is_string_not_empty(
                redis_config, "password"
            ):
                raise ValueError(
                    "Missing required field in agent configuration: redis.password"
                )

        # If no redis config in agent_config, check environment variables
        else:
            # Check for standalone Redis connection env vars
            has_direct_env = os.getenv("REDIS_ADDRESS") and os.getenv("REDIS_PASSWORD")
            # Check for Sentinel connection env vars
            has_sentinel_env = (
                os.getenv("REDIS_SENTINEL_ADDRESSES")
                and os.getenv("REDIS_SENTINEL_MASTER_NAME")
                and os.getenv("REDIS_SENTINEL_PASSWORD")
            )
            if not (has_direct_env or has_sentinel_env):
                raise ValueError(
                    "Required environment variables not set. Need either:\n"
                    "- REDIS_ADDRESS and REDIS_PASSWORD for standalone connection\n"
                    "- REDIS_SENTINEL_ADDRESSES and REDIS_SENTINEL_MASTER_NAME and "
                    "REDIS_SENTINEL_PASSWORD for sentinel connection"
                )

    def __load_redis_config_from_env(self, agent_config: object) -> object:
        agent_config["redis"] = {}
        agent_config["redis"]["db"] = os.environ.get("REDIS_DB") or 0
        if os.environ.get("REDIS_SENTINEL_ADDRESSES"):
            agent_config["redis"]["sentinel_addresses"] = os.environ.get(
                "REDIS_SENTINEL_ADDRESSES"
            ).split(",")
            agent_config["redis"]["sentinel_master_name"] = os.environ.get(
                "REDIS_SENTINEL_MASTER_NAME"
            )
            agent_config["redis"]["sentinel_password"] = os.environ.get(
                "REDIS_SENTINEL_PASSWORD"
            )
            agent_config["redis"]["sentinel_username"] = (
                os.environ.get("REDIS_SENTINEL_USERNAME") or None
            )
        else:
            agent_config["redis"]["address"] = os.environ.get("REDIS_ADDRESS")
            agent_config["redis"]["password"] = os.environ.get("REDIS_PASSWORD")
            agent_config["redis"]["username"] = os.environ.get("REDIS_USERNAME") or None

        self.__check_redis_config(agent_config)

        return agent_config

    def __is_agent_config_file(self, file_folder: str, file_name: str) -> bool:
        return (
            os.path.isfile(os.path.join(file_folder, file_name))
            and re.match(r"agent-[a-zA-Z0-9-_]+\.ya?ml", file_name) is not None
        )

    def __prop_exists_and_is_string_not_empty(self, obj: object, prop: str) -> bool:
        return (
            prop in obj
            and obj[prop] != None
            and isinstance(obj[prop], str)
            and obj[prop] != ""
        )
