from __future__ import annotations

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from vosk import Model, KaldiRecognizer, SetLogLevel

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIConnectionError,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger


# Endpointer configuration constants for Kaldi recognizer
# See: https://github.com/alphacep/vosk-api/blob/625e44c62607a3b48fec6d72a30118448909e83f/src/recognizer.cc#L244-L261
ENDPOINTER_T_START_MAX = 0.25  # Maximum initial silence before speech is recognized (seconds)
ENDPOINTER_T_END = 0.4         # Trailing silence timeout after speech to detect end of utterance (seconds)
ENDPOINTER_T_MAX = 20.0        # Maximum utterance length timeout, not silence! (seconds)


# Module-level model cache to share models across STT instances
# This prevents loading the same model multiple times when transcribing different participants
class _ModelCache:
    """Thread-safe cache for Vosk models.
    
    Models are cached by their absolute path to ensure the same model
    is not loaded multiple times across different STT instances.
    """
    
    def __init__(self):
        self._models: dict[str, Model] = {}
        self._lock = threading.Lock()
    
    def get_or_load(self, model_path: str) -> Model:
        """Get a cached model or load it if not cached.
        
        Args:
            model_path: Path to the Vosk model directory.
            
        Returns:
            The loaded Vosk Model instance.
        """
        # Normalize path for consistent cache keys
        abs_path = os.path.abspath(model_path)
        
        # Check if already cached (without lock for performance)
        if abs_path in self._models:
            logger.info(f"Reusing cached Vosk model from: {model_path}")
            return self._models[abs_path]
        
        # Load with lock to prevent duplicate loading
        with self._lock:
            # Double-check after acquiring lock
            if abs_path in self._models:
                logger.info(f"Reusing cached Vosk model from: {model_path} (loaded by another thread)")
                return self._models[abs_path]
            
            logger.info(f"Loading Vosk model into memory from: {model_path}")
            model = Model(model_path)
            self._models[abs_path] = model
            logger.info(f"Vosk model loaded and cached successfully (path: {model_path})")
            return model
    
    def is_cached(self, model_path: str) -> bool:
        """Check if a model is already cached."""
        abs_path = os.path.abspath(model_path)
        return abs_path in self._models


# Global model cache instance
_model_cache = _ModelCache()

# Shared ThreadPoolExecutor for all STT instances
_shared_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_shared_executor() -> ThreadPoolExecutor:
    """Get or create the shared ThreadPoolExecutor.
    
    Uses all available CPU cores for maximum performance.
    The executor is shared across all STT instances to avoid thread explosion.
    """
    global _shared_executor
    if _shared_executor is not None:
        return _shared_executor
    
    with _executor_lock:
        if _shared_executor is not None:
            return _shared_executor
        
        cpu_count = utils.hw.get_cpu_monitor().cpu_count()
        logger.info(f"Creating shared Vosk ThreadPoolExecutor with max_workers={cpu_count}")
        _shared_executor = ThreadPoolExecutor(max_workers=cpu_count)
        return _shared_executor


def _extract_confidence(result: dict) -> float:
    """Extract confidence from Vosk result.
    
    When words are enabled, Vosk provides per-word confidence in the 'result' array.
    Each word has a 'conf' field. We compute the average confidence across all words.
    If no word-level data is available, returns 1.0 as default.
    """
    words = result.get("result", [])
    if not words:
        return 1.0
    
    confidences = [w.get("conf", 1.0) for w in words]
    return sum(confidences) / len(confidences) if confidences else 1.0


def _extract_start_time(result: dict, offset: float = 0.0) -> float:
    """Extract start time from Vosk result.
    
    When words are enabled, Vosk provides per-word timing in the 'result' array.
    Each word has a 'start' field (time in seconds). Returns the start time of the first word.
    If no word-level data is available, returns the offset.
    """
    words = result.get("result", [])
    if not words:
        return offset
    return words[0].get("start", 0.0) + offset


def _extract_end_time(result: dict, offset: float = 0.0) -> float:
    """Extract end time from Vosk result.
    
    When words are enabled, Vosk provides per-word timing in the 'result' array.
    Each word has an 'end' field (time in seconds). Returns the end time of the last word.
    If no word-level data is available, returns the offset.
    """
    words = result.get("result", [])
    if not words:
        return offset
    return words[-1].get("end", 0.0) + offset


@dataclass
class STTOptions:
    model_path: str | None
    sample_rate: int
    partial_results: bool
    language: str


class STT(stt.STT):
    def __init__(
        self,
        *,
        model_path: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 16000,
        partial_results: bool = True,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of Vosk STT.

        Vosk is an offline speech recognition toolkit that runs locally without
        requiring an internet connection or API key.

        Args:
            model_path: Path to the Vosk model directory. You can download models from
                       https://alphacephei.com/vosk/models
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            partial_results: Whether to return interim/partial results during recognition.
                           Defaults to True.
            language: Optional language code for metadata purposes (e.g., "en-US", "es").
                     This is included in the SpeechData output but does not affect recognition.
                     The model determines the recognition language.

        Note:
            You need to download a Vosk model before using this STT.
            Models can be downloaded from: https://alphacephei.com/vosk/models
            
            Set the model path either:
            - Using the model_path parameter
            - Using the VOSK_MODEL_PATH environment variable
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=partial_results,
            )
        )

        # Reduce Vosk log verbosity
        SetLogLevel(-1)

        self._opts = STTOptions(
            model_path=model_path if is_given(model_path) else None,
            sample_rate=sample_rate,
            partial_results=partial_results,
            language=language if is_given(language) else "",
        )

        # Use shared executor for all STT instances (avoids thread explosion)
        self._executor = _get_shared_executor()

    @property
    def model(self) -> str:
        return self._get_model_path()

    @property
    def provider(self) -> str:
        return "Vosk"

    def _get_model_path(self) -> str:
        """Determine the model path to use."""
        if self._opts.model_path:
            return self._opts.model_path

        # Check environment variable
        env_path = os.environ.get("VOSK_MODEL_PATH")
        if env_path:
            return env_path

        raise ValueError(
            "Vosk model path is required. Either:\n"
            "  - Set the model_path parameter\n"
            "  - Set the VOSK_MODEL_PATH environment variable\n"
            "Download models from: https://alphacephei.com/vosk/models"
        )

    async def _ensure_model(self) -> Model:
        """Get the Vosk model from the shared cache.
        
        Models are cached at the module level and shared across all STT instances
        to avoid loading the same model multiple times for different participants.
        """
        model_path = self._get_model_path()
        
        # Check if already cached (fast path)
        if _model_cache.is_cached(model_path):
            return _model_cache.get_or_load(model_path)
        
        # Load in executor to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            model = await loop.run_in_executor(
                self._executor, lambda: _model_cache.get_or_load(model_path)
            )
        except Exception as e:
            raise APIConnectionError(
                f"Failed to load Vosk model from {model_path}: {e}"
            ) from e
        
        return model

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Recognize speech from an audio buffer (non-streaming)."""
        try:
            model = await self._ensure_model()
            frame = rtc.combine_audio_frames(buffer)

            # Ensure audio is at the correct sample rate
            if frame.sample_rate != self._opts.sample_rate:
                resampler = rtc.AudioResampler(
                    frame.sample_rate,
                    self._opts.sample_rate,
                    quality=rtc.AudioResamplerQuality.HIGH,
                )
                resampled_frames = resampler.push(frame)
                resampled_frames.extend(resampler.flush())
                if resampled_frames:
                    frame = rtc.combine_audio_frames(resampled_frames)

            # Create recognizer and process audio
            loop = asyncio.get_event_loop()

            def recognize_sync() -> str:
                recognizer = KaldiRecognizer(model, self._opts.sample_rate)
                recognizer.SetWords(True)

                # Process audio data
                audio_bytes = frame.data.tobytes()
                recognizer.AcceptWaveform(audio_bytes)

                # Get final result
                return recognizer.FinalResult()

            result_json = await loop.run_in_executor(self._executor, recognize_sync)
            result = json.loads(result_json)

            text = result.get("text", "")
            alternatives = []

            if text:
                alternatives.append(
                    stt.SpeechData(
                        language=self._opts.language,
                        text=text,
                        start_time=_extract_start_time(result),
                        end_time=_extract_end_time(result),
                        confidence=_extract_confidence(result),
                    )
                )

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=alternatives,
            )

        except Exception as e:
            raise APIConnectionError(f"Vosk recognition failed: {e}") from e

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Create a streaming speech recognition session."""
        return SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=self._opts,
            executor=self._executor,
            model_getter=self._ensure_model,
        )

    def update_options(
        self,
        *,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        partial_results: NotGivenOr[bool] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Update STT options.
        
        Note: model_path cannot be changed after initialization as models are
        cached at the module level for memory efficiency.
        """
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(partial_results):
            self._opts.partial_results = partial_results
        if is_given(language):
            self._opts.language = language

    async def aclose(self) -> None:
        """Close the STT and release resources."""
        # Don't shutdown executor - it's shared across all STT instances
        await super().aclose()


class SpeechStream(stt.SpeechStream):
    """Streaming speech recognition using Vosk."""

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        opts: STTOptions,
        executor: ThreadPoolExecutor,
        model_getter,
    ) -> None:
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=opts.sample_rate,
        )
        self._opts = opts
        self._executor = executor
        self._model_getter = model_getter
        self._speaking = False

    async def _run(self) -> None:
        """Main loop for processing streaming audio."""
        try:
            model = await self._model_getter()
        except Exception as e:
            raise APIConnectionError(f"Failed to load Vosk model: {e}") from e

        loop = asyncio.get_event_loop()

        # Create recognizer
        def create_recognizer():
            logger.debug("[Vosk SpeechStream] Creating KaldiRecognizer")
            r = KaldiRecognizer(model, self._opts.sample_rate)
            r.SetWords(True)
            try:
                r.SetEndpointerDelays(ENDPOINTER_T_START_MAX, ENDPOINTER_T_END, ENDPOINTER_T_MAX)
                logger.debug(
                    f"[Vosk SpeechStream] Endpointer configured: "
                    f"delays=({ENDPOINTER_T_START_MAX}, {ENDPOINTER_T_END}, {ENDPOINTER_T_MAX})"
                )
            except Exception as e:
                logger.warning(f"[Vosk SpeechStream] Failed to set endpointer delays: {e}")
                    
            return r

        recognizer = await loop.run_in_executor(self._executor, create_recognizer)

        # Buffer for accumulating audio
        audio_buffer = bytearray()
        chunk_size = int(self._opts.sample_rate * 0.1 * 2)  # 100ms chunks, 16-bit audio

        async for frame in self._input_ch:
            if isinstance(frame, self._FlushSentinel):
                logger.debug(
                    f"[Vosk SpeechStream] FlushSentinel received: "
                    f"buffer_size={len(audio_buffer)} bytes, speaking={self._speaking}"
                )
                # Process remaining audio and finalize
                if audio_buffer:
                    logger.debug(f"[Vosk SpeechStream] Processing remaining {len(audio_buffer)} bytes before flush")
                    await self._process_audio_chunk(
                        loop, recognizer, bytes(audio_buffer)
                    )
                    audio_buffer.clear()

                # Get final result
                final_result_json = await loop.run_in_executor(
                    self._executor, recognizer.FinalResult
                )
                final_result = json.loads(final_result_json)
                text = final_result.get("text", "")
                logger.debug(f"[Vosk SpeechStream] FinalResult: '{text}'")

                if text:
                    confidence = _extract_confidence(final_result)
                    start_time = _extract_start_time(final_result, self.start_time_offset)
                    end_time = _extract_end_time(final_result, self.start_time_offset)
                    logger.debug(
                        f"[Vosk SpeechStream] FINAL (from flush): '{text}' "
                        f"(confidence={confidence:.2f}, start={start_time:.2f}s, end={end_time:.2f}s)"
                    )
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[
                                stt.SpeechData(
                                    language=self._opts.language,
                                    text=text,
                                    start_time=start_time,
                                    end_time=end_time,
                                    confidence=confidence,
                                )
                            ],
                        )
                    )
                else:
                    logger.debug("[Vosk SpeechStream] FinalResult was empty after flush")

                # Send END_OF_SPEECH if we were speaking
                if self._speaking:
                    logger.debug("[Vosk SpeechStream] Emitting END_OF_SPEECH after flush")
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    )
                    self._speaking = False

                # Reset recognizer for next segment
                logger.debug("[Vosk SpeechStream] Resetting recognizer for next segment")
                await loop.run_in_executor(
                    self._executor, recognizer.Reset
                )
                continue

            if isinstance(frame, rtc.AudioFrame):
                # Add audio data to buffer
                audio_buffer.extend(frame.data.tobytes())

                # Process in chunks
                while len(audio_buffer) >= chunk_size:
                    chunk = bytes(audio_buffer[:chunk_size])
                    audio_buffer = audio_buffer[chunk_size:]

                    await self._process_audio_chunk(loop, recognizer, chunk)

    async def _process_audio_chunk(
        self,
        loop: asyncio.AbstractEventLoop,
        recognizer: KaldiRecognizer,
        audio_data: bytes,
    ) -> None:
        """Process an audio chunk and emit events."""
        # Process audio in executor to avoid blocking
        is_final = await loop.run_in_executor(
            self._executor,
            lambda: recognizer.AcceptWaveform(audio_data),
        )
        
        if is_final:
            logger.debug(f"[Vosk SpeechStream] AcceptWaveform returned True (internal endpointing triggered)")
            # Get final result
            result_json = await loop.run_in_executor(
                self._executor, recognizer.Result
            )
            result = json.loads(result_json)
            text = result.get("text", "")

            if text:
                # Emit START_OF_SPEECH if this is the first speech
                if not self._speaking:
                    logger.debug("[Vosk SpeechStream] Emitting START_OF_SPEECH (first speech detected via internal)")
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
                    self._speaking = True

                confidence = _extract_confidence(result)
                start_time = _extract_start_time(result, self.start_time_offset)
                end_time = _extract_end_time(result, self.start_time_offset)
                logger.info(
                    f"[Vosk SpeechStream] FINAL (from internal endpointing): '{text}' "
                    f"(confidence={confidence:.2f}, start={start_time:.2f}s, end={end_time:.2f}s)"
                )
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                language=self._opts.language,
                                text=text,
                                start_time=start_time,
                                end_time=end_time,
                                confidence=confidence,
                            )
                        ],
                    )
                )
        elif self._opts.partial_results:
            # Get partial result
            partial_json = await loop.run_in_executor(
                self._executor, recognizer.PartialResult
            )
            partial = json.loads(partial_json)
            text = partial.get("partial", "")

            if text:
                # Emit START_OF_SPEECH if this is the first speech
                if not self._speaking:
                    logger.debug("[Vosk SpeechStream] Emitting START_OF_SPEECH (first partial detected)")
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
                    self._speaking = True

                logger.debug(f"[Vosk SpeechStream] INTERIM: '{text}'")
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                language=self._opts.language,
                                text=text,
                                start_time=self.start_time_offset,
                                end_time=self.start_time_offset,
                                confidence=0.0,
                            )
                        ],
                    )
                )
