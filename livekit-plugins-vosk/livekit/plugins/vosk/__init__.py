"""Vosk plugin for LiveKit Agents

Offline speech-to-text using [Vosk](https://alphacephei.com/vosk/).

Vosk is an offline speech recognition toolkit that runs locally without
requiring an internet connection or API key. Download models from:
https://alphacephei.com/vosk/models

Example usage:
    ```python
    from livekit.plugins import vosk

    stt = vosk.STT(model_path="/path/to/vosk-model")
    ```
"""

from .stt import STT, SpeechStream
from .version import __version__

__all__ = ["STT", "SpeechStream", "__version__"]


from livekit.agents import Plugin

from .log import logger


class VoskPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(VoskPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
