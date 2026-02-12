# LiveKit Plugins Vosk

Agent Framework plugin for offline Speech-to-Text using [Vosk](https://alphacephei.com/vosk/).

## Features

- **Offline Speech Recognition**: Runs locally without requiring an internet connection
- **No API Key Required**: Completely free to use with no usage limits
- **Multiple Language Support**: Supports 10+ languages with downloadable models
- **Real-time Streaming**: Optimized for real-time, provides interim results during speech recognition

## Installation

```bash
pip install git+https://github.com/OpenVidu/agents.git#egg=livekit-plugins-vosk&subdirectory=livekit-plugins/livekit-plugins-vosk
```

## Model Download

Before using this plugin, you need to download a Vosk model. Models are available at:
https://alphacephei.com/vosk/models

### Recommended Models

| Language | Model | Size | Description |
|----------|-------|------|-------------|
| English (US) | vosk-model-en-us-0.22 | 1.8 GB | Large model, best accuracy |
| English (US) | vosk-model-small-en-us-0.15 | 40 MB | Small model, faster |
| Spanish | vosk-model-es-0.42 | 1.4 GB | Large Spanish model |
| German | vosk-model-de-0.21 | 1.9 GB | Large German model |
| French | vosk-model-fr-0.22 | 1.4 GB | Large French model |
| Chinese | vosk-model-cn-0.22 | 1.3 GB | Large Chinese model |

### Download Example

```bash
# Download and extract a model
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

## Usage

### Basic Usage

```python
from livekit.plugins import vosk

# Create STT instance with model path
stt = vosk.STT(model_path="/path/to/vosk-model-small-en-us-0.15")
```

### Using Environment Variable

```bash
export VOSK_MODEL_PATH=/path/to/vosk-model-small-en-us-0.15
```

```python
from livekit.plugins import vosk

# Model path will be read from VOSK_MODEL_PATH environment variable
stt = vosk.STT()
```

### With LiveKit Agent

```python
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from livekit.plugins import vosk

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    session = AgentSession(
        stt=vosk.STT(model_path="/path/to/vosk-model"),
    )
    
    # Start the session
    await session.start(room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### Configuration Options

```python
stt = vosk.STT(
    language="en-us",           # Language code (for reference)
    model_path="/path/to/model", # Path to Vosk model directory
    sample_rate=16000,          # Audio sample rate in Hz
    words=False,                # Include word-level timestamps
    partial_results=True,       # Enable interim results
)
```

## Supported Languages

Vosk supports 30+ languages including:
- English (US, UK, Indian)
- Spanish
- German
- French
- Chinese
- Russian
- Portuguese
- Italian
- Dutch
- Japanese
- Korean
- Arabic
- And many more...

See the full list at: https://alphacephei.com/vosk/models

## License

Apache-2.0
