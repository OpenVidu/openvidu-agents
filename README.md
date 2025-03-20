# openvidu-agents

This is a collection of pre-configured and ready-to-use AI Agents for OpenVidu. They are built using the [LiveKit Agents framework](https://docs.livekit.io/agents/). They are designed to easily be added to an OpenVidu deployment and provide with useful AI services.

The list of available agents is:

- [speech-to-text](speech-to-text/README.md): Transcribes the audio of a Room to text in real-time.

## Building an agent

To build the Docker image (in this example for agent `speech-to-text`):

```bash
cd speech-to-text
docker build --no-cache -t openvidu/agent-speech-to-text:3.2.0 .
```

> `--no-cache` is required to bring latest changes from the shared utils library hosted in the repository.

To prepare the agent for development (in this example agent `speech-to-text`), it is necessary to install its depdendencies:

```bash
# Create a virtual environment
python3 -m venv .venv
. .venv/bin/activate
# Install agent dependencies
cd speech-to-text
pip3 install -r requirements.txt
```

To debug the agent in VSCode (in this example agent `speech-to-text`), there is a preconfigured configuration in `.vscode/launch.json`. Simply click on the "Run" button in the debug panel with the proper configuration selected.

<img src="data:image/png;base64,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" />

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
