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

To prepare the agent for development (in this example agent `speech-to-text`), it is necessary to install its depdendencies:

```bash
# Create a virtual environment
python3 -m venv .venv
. .venv/bin/activate
# Install agent dependencies
cd speech-to-text
pip3 install -r requirements.txt
```

### Debugging in VSCode

To debug the agent in VSCode, there are preconfigured configurations at `.vscode/launch.json`. Simply click on the "Run" button in the debug panel with the proper configuration selected.

### Shared common library

All agents need a common utils library to work. This library is located at folder `openviduagentutils` and it is a Python package. All agents need to import it as a dependency in their `requirements.txt` like this:

```
openviduagentutils @ git+https://github.com/OpenVidu/openvidu-agents#egg=openviduagentutils&subdirectory=openviduagentutils
```

Any change done to the common library must be pushed to the remote repository, so agents are able to pull the latest version. To do so, the agent must simply reinstall the dependencies:

```bash
pip3 install -r requirements.txt --force-reinstall
```

## Agent configuration

Agents require a set of environment variables, and a YAML configuration.

In order:

- Env var `AGENT_CONFIG_BODY` with the YAML configuration as a string. In this case, it must contain property `agent_name` to define the agent's name.
- Env var `AGENT_CONFIG_FILE` with the path to a YAML file containing the agent configuration. In this case, the agent name will be obtained from the file name itself (`agent-AGENT_NAME.yml`).
- For development purposes, the agent will look for a file named `agent-*.yml` in CWD. In this case, the agent name will be obtained from the file name itself (`agent-AGENT_NAME.yml`).

## Run unit tests

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
find . -name "test_*.py" -type f -not -path '*/venv/*' -exec sh -c 'cd $(dirname {}) && python -m unittest $(basename {})' \;
```
