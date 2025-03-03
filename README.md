# agents-catalog

This is a collection of pre-configured and ready-to-use AI Agents for OpenVidu. They are built using the [LiveKit Agents framework](https://docs.livekit.io/agents/). They are designed to easily be added to an OpenVidu deployment.

The list of available agents is:

- [speech-to-text](speech-to-text/README.md): Transcribes the audio of a session to text in real-time.

# Agent configuration

In order:

- Env var `AGENT_CONFIG_BODY` with the YAML configuration as a string. In this case, it must contain property `agent_name` to define the agent's name.
- Env var `AGENT_CONFIG_FILE` with the path to a YAML file containing the agent configuration. In this case, the agent name will be obtained from the file name itself (`agent-AGENT_NAME.yml`).
- For development purposes, the agent will look for a file named `agent-*.yml` in CWD. In this case, the agent name will be obtained from the file name itself (`agent-AGENT_NAME.yml`).

# Run unit tests

Run specific agent tests:

```bash
python -m unittest discover -s <AGENT_FOLDER> -p 'test_*.py'
```

Run all tests:

```bash
for d in */ ; do python -m unittest discover -s $d -p 'test_*.py'; done
```
