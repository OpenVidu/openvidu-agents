# openvidu-agents

> [!WARNING]  
> This repository is actively being developed and is subject to change.

- [Introduction](#introduction)
- [Developing an agent](#developing-an-agent)
  - [Create a new agent](#create-a-new-agent)
  - [Build as a Docker image](#build-as-a-docker-image)
  - [Local development](#local-development)
  - [Debugging in VSCode](#debugging-in-vscode)
- [Shared common library](#shared-common-library)
  - [Developing the common library](#developing-the-common-library)
- [Agent configuration](#agent-configuration)
  - [Environment variables](#environment-variables)
  - [YAML configuration](#yaml-configuration)
- [Run tests](#run-tests)

## Introduction

This is a collection of pre-configured and ready-to-use AI Agents for OpenVidu. They are built using the [LiveKit Agents framework](https://docs.livekit.io/agents/). They are designed to easily be added to an OpenVidu deployment and provide useful AI services.

The list of available agents is:

- [speech-processing](speech-processing/README.md): AI services for transcribing, translating and summarizing video conference conversations.

## Developing an agent

### Create a new agent

> Folder `minimal` contains a copy-paste template for creating a new agent with the minimum required files.

All OpenVidu agents must have at least the following files:

- `Dockerfile`: The Dockerfile to build the agent image. Generally this file is the same for all agents, except if extra files are needed in the image (such as AI models).
- `requirements.txt`: The Python dependencies of the agent.
- `main.py`: The Python entrypoint of the agent.
- `agent-AGENT_NAME.yml`: The YAML configuration of the agent. This file defines the common and agent-specific configurations.

All OpenVidu agents must:

- Have the following dependency in their `requirements.txt` file:

  ```
  openviduagentutils @ git+https://github.com/OpenVidu/openvidu-agents#egg=openviduagentutils&subdirectory=openviduagentutils
  ```

- Import and properly use the `openviduagentutils` package in `main.py`. This package atuomatically loads and checks mandatory agent configuration, provides utilities to load custom configurations and allows managing the agent lifecycle.

  ```python
  from openviduagentutils.openvidu_agent import OpenViduAgent

  async def entrypoint(ctx: JobContext) -> None:
    openvidu_agent = OpenViduAgent.get_instance()
    openvidu_agent.new_active_job(ctx)
    # DO SOME WORK IN THE ROOM...

  async def request_fnc(req: JobRequest) -> None:
    openvidu_agent = OpenViduAgent.get_instance()
    if not openvidu_agent.can_accept_new_jobs():
        await req.reject()
    else:
        agent_name = openvidu_agent.get_agent_name()
        await req.accept(
            name=agent_name
        )

  if __name__ == "__main__":
    openvidu_agent = OpenViduAgent.get_instance(True) # True only in the "main" program
    agent_config = openvidu_agent.get_agent_config()
    agent_name = openvidu_agent.get_agent_name()
    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=request_fnc,
        api_key=agent_config["api_key"],
        api_secret=agent_config["api_secret"],
        ws_url=agent_config["ws_url"],
        worker_type=WorkerType.ROOM,
    )
    if agent_config["live_captions"]["processing"] == "manual":
        worker_options.agent_name = agent_name
    cli.run_app(worker_options)
  ```

### Build as a Docker image

To build the Docker image (in this example for agent `speech-processing`):

```bash
cd speech-processing
docker build --no-cache -f Dockerfile.base -t openvidu/agent-speech-processing-base:main .
docker build --no-cache -f Dockerfile.cloud -t openvidu/agent-speech-processing-cloud:main .
```

> `--no-cache` is required to bring latest changes from the shared utils library hosted in the repository.

### Local development

To prepare the agent for development (in this example agent `speech-processing`), it is necessary to install its dependencies:

```bash
# Create a virtual environment
python3 -m venv .venv
. .venv/bin/activate
# Install agent dependencies
cd speech-processing
pip3 install -r requirements-cloud.txt
```

### Debugging in VSCode

To debug the agent in VSCode, there is a configuration at `.vscode/launch.json`. Simply run the "Debug agent" configuration in the debug panel while having the .py entrypoint opened for the desired agent.

> Make sure you have completed the steps in the [Local development](#local-development) section before debugging with VSCode.

## Shared common library

All agents need a common utils library to work. This library is located at folder `openviduagentutils` and it is a Python package. All agents need to import it as a dependency in their `requirements.txt` like this:

```
openviduagentutils @ git+https://github.com/OpenVidu/openvidu-agents#egg=openviduagentutils&subdirectory=openviduagentutils
```

Agents may bring any change pushed to the common library simply by reinstalling their dependencies:

```bash
cd speech-processing
pip3 install -r requirements.txt --force-reinstall
```

### Developing the common library

To make it easier to develop the common library, the best way is to install it as an [editable local package](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) in the agent's virtual environment. For example, to develop the common library against agent `speech-processing`:

```bash
cd speech-processing
python3 -m pip install -e ../openviduagentutils/
```

After that, any change done to the common library will be reflected in the agent without the need to reinstall the dependencies. Just restart the agent for the changes to take effect. Remember to push any desired changes to the common library to the remote repository, so agents are able to include them when building their Docker images.

> This strategy works great with the [VSCode debugger](#debugging-in-vscode)!

To remove the editable package from the agent and start using the remote package again, simply reinstall dependencies:

```bash
cd speech-processing
pip3 install -r requirements.txt
```

## Agent configuration

Agents require a set of environment variables and a YAML configuration.

### Environment variables

These mandatory environment variables allow the agent to connect to OpenVidu:

```sh
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_URL=ws://127.0.0.1:7880/
# For Redis standalone
REDIS_ADDRESS=127.0.0.1:6379
REDIS_PASSWORD=redispassword
# For Redis Sentinel
# REDIS_SENTINEL_MASTER_NAME=openvidu
# REDIS_SENTINEL_ADDRESSES=10.5.0.3:7001,10.5.0.4:7001,10.5.0.5:7001,10.5.0.6:7001
# REDIS_SENTINEL_PASSWORD=fUhZxagsL4evGRgQEeTyhrD5w4cHeNgja04g2iGh1rlD
```

> Env vars can be provided via a file that is itself specified with env variable `ENV_VARS_FILE` (e.g. `ENV_VARS_FILE=/etc/.env`).

### YAML configuration

This configuration is dependant on each agent. It can be provided in three different ways. In order of precedence:

- Env var `AGENT_CONFIG_BODY` with the YAML configuration as a string. In this case, it must contain property `agent_name` to define the agent's name.
- Env var `AGENT_CONFIG_FILE` with the path to a YAML file containing the agent configuration. In this case, the agent name will be obtained from the file name itself (`agent-AGENT_NAME.yml`).
- For development purposes, the agent will look for a file named `agent-*.yml` in CWD. In this case, the agent name will be obtained from the file name itself (`agent-AGENT_NAME.yml`).

## Run tests

Run specific test:

```bash
python3 -m unittest test_stt_impl.py
```

Run tests in a folder:

```bash
python3 -m unittest discover -s .
```

Run all tests:

```bash
find . -maxdepth 4 -name "test_*.py" -type f -exec sh -c 'cd $(dirname {}) && python3 -m unittest $(basename {})' \;
```
