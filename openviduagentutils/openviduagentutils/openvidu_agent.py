import logging
import os
import re
import sys
import yaml
from typing import TypeVar
from enum import Enum
from dotenv import load_dotenv

T = TypeVar("T", bound=Enum)

# Singleton pattern for OpenViduAgent
openvidu_agent = None


class OpenViduAgent:
    """Utility class to load agent configuration from environment variables or a YAML file."""

    def __init__(self):
        agent_config, agent_name = self.__load_agent_config()
        self.__agent_config = agent_config
        self.__agent_name = agent_name

    def get_instance() -> "OpenViduAgent":
        """Get the singleton instance of OpenViduAgent"""
        global openvidu_agent
        if openvidu_agent is None:
            openvidu_agent = OpenViduAgent()
        return openvidu_agent

    def get_agent_config(self) -> object:
        """Get the agent configuration"""
        return self.__agent_config

    def get_agent_name(self) -> str:
        """Get the agent name"""
        return self.__agent_name

    def __load_agent_config(self) -> tuple[object, str]:
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
