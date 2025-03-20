# openvidu-agents

This is a collection of pre-configured and ready-to-use AI Agents for OpenVidu. They are built using the [LiveKit Agents framework](https://docs.livekit.io/agents/). They are designed to easily be added to an OpenVidu deployment and provide useful AI services.

The list of available agents is:

- [speech-to-text](speech-to-text/README.md): Transcribes the audio of a Room to text in real-time.

## Developing an agent

### Build as a Docker image

To build the Docker image (in this example for agent `speech-to-text`):

```bash
cd speech-to-text
docker build --no-cache -t openvidu/agent-speech-to-text:3.2.0 .
```

> `--no-cache` is required to bring latest changes from the shared utils library hosted in the repository.

### Local development

To prepare the agent for development (in this example agent `speech-to-text`), it is necessary to install its dependencies:

```bash
# Create a virtual environment
python3 -m venv .venv
. .venv/bin/activate
# Install agent dependencies
cd speech-to-text
pip3 install -r requirements.txt
```

### Debugging in VSCode

To debug the agent in VSCode, there is a configuration at `.vscode/launch.json`. Simply run the "Debug agent" configuration in the debug panel while having the .py entrypoint opened for the desired agent.

> Make sure you have completed the steps in the [Local development](#local-development) section before debugging with VSCode.

## Shared common library

All agents need a common utils library to work. This library is located at folder `openviduagentutils` and it is a Python package. All agents need to import it as a dependency in their `requirements.txt` like this:

```
openviduagentutils @ git+https://github.com/OpenVidu/openvidu-agents#egg=openviduagentutils&subdirectory=openviduagentutils
```

Any change done to the common library must be pushed to the remote repository, so agents are able to pull the latest version. To do so, the agent must simply reinstall the dependencies:

```bash
pip3 install -r requirements.txt --force-reinstall
```

### Developing the common library

Install it as an [editable local package](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) in the agent's virtual environment. For example, to develop the common library against agent `speech-to-text`:

```bash
cd speech-to-text
python3 -m pip install -e ../openviduagentutils/
```

After that, any change done to the common library will be reflected in the agent without the need to reinstall the dependencies. Just restart the agent for the changes to take effect.

> This strategy works great with the [VSCode debugger](#debugging-in-vscode).

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
python -m unittest test_stt_impl.py
```

Run tests in a folder:

```bash
python -m unittest discover -s .
```

Run all tests:

```bash
find . -name "test_*.py" -type f -maxdepth 4 -exec sh -c 'cd $(dirname {}) && python -m unittest $(basename {})' \;
```
