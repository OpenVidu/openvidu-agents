"""VAD-triggered STT wrapper for faster final transcripts.

This wrapper coordinates VAD and STT to flush the STT stream immediately
when VAD detects end of speech, rather than waiting for the STT's internal
endpointing/silence detection.

This is particularly useful for local STT models like Vosk and Sherpa that
have slower internal endpointing compared to cloud-based STTs.

Usage:
    from vad_stt_wrapper import VADTriggeredSTT
    from livekit.plugins import vosk, silero
    
    vad_model = silero.VAD.load()
    base_stt = vosk.STT(model_path="...")
    
    # Create a VAD-triggered STT that flushes on VAD END_OF_SPEECH
    stt_impl = VADTriggeredSTT(stt_impl=base_stt, vad_impl=vad_model)
    
    # Use in your agent - no external VAD needed since it's integrated
    session.start(
        agent=Transcriber(
            stt_impl=stt_impl,
            turn_detection="stt",  # Use STT-based since VAD is integrated
            vad_model=None,
        )
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from livekit import rtc
from livekit.agents import stt, vad, utils, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import aio

if TYPE_CHECKING:
    from livekit.agents.stt.stt import SpeechStream

logger = logging.getLogger(__name__)

# Enable debug logging for this module
# logger.setLevel(logging.DEBUG)


class VADTriggeredSTT(stt.STT):
    """STT wrapper that uses VAD to trigger immediate flushes for faster finals.
    
    When VAD detects END_OF_SPEECH, this wrapper immediately flushes the
    underlying STT stream to get a final transcript, rather than waiting
    for the STT's internal silence detection.
    """
    
    def __init__(
        self,
        *,
        stt_impl: stt.STT,
        vad_impl: vad.VAD,
        flush_delay: float = 0.0,
    ) -> None:
        """Create a VAD-triggered STT wrapper.
        
        Args:
            stt_impl: The underlying STT to use for transcription.
            vad_impl: The VAD to use for speech boundary detection.
            flush_delay: Optional delay (in seconds) after VAD END_OF_SPEECH
                        before flushing. Can help with edge cases where speech
                        continues briefly after VAD triggers.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=stt_impl.capabilities.streaming,
                interim_results=stt_impl.capabilities.interim_results,
                aligned_transcript=stt_impl.capabilities.aligned_transcript,
                offline_recognize=stt_impl.capabilities.offline_recognize,
            )
        )
        self._stt = stt_impl
        self._vad = vad_impl
        self._flush_delay = flush_delay
        
        # Get sample rate from the underlying STT if available
        self._sample_rate = getattr(stt_impl, '_opts', None)
        if self._sample_rate and hasattr(self._sample_rate, 'sample_rate'):
            self._sample_rate = self._sample_rate.sample_rate
        else:
            self._sample_rate = 16000
        
        logger.debug(
            f"[VADTriggeredSTT] Created wrapper - "
            f"underlying_stt={stt_impl.provider}, "
            f"sample_rate={self._sample_rate}, "
            f"flush_delay={flush_delay}s"
        )
    
    @property
    def model(self) -> str:
        return self._stt.model
    
    @property
    def provider(self) -> str:
        return f"vad-triggered/{self._stt.provider}"
    
    async def _recognize_impl(
        self,
        buffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        # Delegate to underlying STT for non-streaming recognition
        return await self._stt._recognize_impl(buffer, language=language, conn_options=conn_options)
    
    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "VADTriggeredSpeechStream":
        """Create a VAD-triggered streaming speech recognition session."""
        logger.debug("[VADTriggeredSTT] Creating new VADTriggeredSpeechStream")
        return VADTriggeredSpeechStream(
            stt=self,
            base_stt=self._stt,
            vad=self._vad,
            flush_delay=self._flush_delay,
            sample_rate=self._sample_rate,
            conn_options=conn_options,
        )
    
    def update_options(self, **kwargs) -> None:
        self._stt.update_options(**kwargs)
    
    async def aclose(self) -> None:
        await self._stt.aclose()
        await super().aclose()


class VADTriggeredSpeechStream(stt.SpeechStream):
    """Speech stream that uses VAD to trigger STT flushes."""
    
    def __init__(
        self,
        *,
        stt: VADTriggeredSTT,
        base_stt: stt.STT,
        vad: vad.VAD,
        flush_delay: float,
        sample_rate: int,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=sample_rate,
        )
        self._base_stt = base_stt
        self._vad = vad
        self._flush_delay = flush_delay
        self._stt_stream: SpeechStream | None = None
        self._vad_stream: vad.VADStream | None = None
        self._speaking = False
        self._stream_id = id(self)  # Unique ID for logging
        self._audio_frame_count = 0
        self._flush_count = 0
        self._start_time = time.monotonic()
        logger.debug(f"[VADTriggeredStream:{self._stream_id}] Stream initialized")
    
    async def _run(self) -> None:
        """Main loop coordinating VAD and STT."""
        logger.info(f"[VADTriggeredStream:{self._stream_id}] Starting _run() - coordinating VAD and STT")
        
        # Create streams
        self._stt_stream = self._base_stt.stream(conn_options=self._conn_options)
        self._vad_stream = self._vad.stream()
        logger.debug(f"[VADTriggeredStream:{self._stream_id}] Created underlying STT stream and VAD stream")
        
        # Channel to coordinate flush signals
        flush_event = asyncio.Event()
        
        @utils.log_exceptions(logger=logging.getLogger(__name__))
        async def vad_task() -> None:
            """Monitor VAD and signal flushes on END_OF_SPEECH."""
            logger.debug(f"[VADTriggeredStream:{self._stream_id}] vad_task started - listening for VAD events")
            vad_event_count = 0
            inference_done_count = 0
            async for ev in self._vad_stream:
                vad_event_count += 1
                elapsed = time.monotonic() - self._start_time
                
                if ev.type == vad.VADEventType.START_OF_SPEECH:
                    self._speaking = True
                    logger.debug(
                        f"[VADTriggeredStream:{self._stream_id}] VAD START_OF_SPEECH "
                        f"(event #{vad_event_count}, elapsed={elapsed:.2f}s)"
                    )
                    # Forward START_OF_SPEECH from VAD
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
                    logger.debug(f"[VADTriggeredStream:{self._stream_id}] Forwarded START_OF_SPEECH to event channel")
                
                elif ev.type == vad.VADEventType.END_OF_SPEECH:
                    self._speaking = False
                    logger.debug(
                        f"[VADTriggeredStream:{self._stream_id}] VAD END_OF_SPEECH "
                        f"(event #{vad_event_count}, elapsed={elapsed:.2f}s, "
                        f"speech_duration={ev.speech_duration:.3f}s, silence_duration={ev.silence_duration:.3f}s)"
                    )
                    
                    # Signal flush after optional delay
                    if self._flush_delay > 0:
                        logger.debug(f"[VADTriggeredStream:{self._stream_id}] Waiting {self._flush_delay}s before flush signal")
                        await asyncio.sleep(self._flush_delay)
                    
                    logger.debug(f"[VADTriggeredStream:{self._stream_id}] Setting flush_event to trigger STT flush")
                    flush_event.set()
                    
                    # Forward END_OF_SPEECH from VAD
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    )
                    logger.debug(f"[VADTriggeredStream:{self._stream_id}] Forwarded END_OF_SPEECH to event channel")
                
                elif ev.type == vad.VADEventType.INFERENCE_DONE:
                    # Log only 1 out of every 10 inference events to reduce noise
                    inference_done_count += 1
                    if inference_done_count % 10 == 1:
                        logger.debug(
                            f"[VADTriggeredStream:{self._stream_id}] VAD INFERENCE_DONE "
                            f"(#{inference_done_count}, showing 1/10)"
                        )
                
                else:
                    logger.debug(f"[VADTriggeredStream:{self._stream_id}] VAD event: {ev.type}")
            
            logger.debug(f"[VADTriggeredStream:{self._stream_id}] vad_task ended - processed {vad_event_count} events")
        
        @utils.log_exceptions(logger=logging.getLogger(__name__))
        async def stt_forward_task() -> None:
            """Forward STT events, filtering out STT's own START/END_OF_SPEECH."""
            logger.debug(f"[VADTriggeredStream:{self._stream_id}] stt_forward_task started - listening for STT events")
            stt_event_count = 0
            filtered_count = 0
            async for ev in self._stt_stream:
                stt_event_count += 1
                elapsed = time.monotonic() - self._start_time
                
                # Filter out START/END_OF_SPEECH from STT (VAD handles these)
                if ev.type in (
                    stt.SpeechEventType.START_OF_SPEECH,
                    stt.SpeechEventType.END_OF_SPEECH,
                ):
                    filtered_count += 1
                    logger.debug(
                        f"[VADTriggeredStream:{self._stream_id}] Filtered out STT {ev.type} "
                        f"(VAD handles these, filtered #{filtered_count})"
                    )
                    continue
                
                # Forward transcription events
                if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    text = ev.alternatives[0].text if ev.alternatives else ""
                    confidence = ev.alternatives[0].confidence if ev.alternatives else 0.0
                    logger.debug(
                        f"[VADTriggeredStream:{self._stream_id}] STT FINAL_TRANSCRIPT "
                        f"(event #{stt_event_count}, elapsed={elapsed:.2f}s): "
                        f"'{text}' (confidence={confidence:.2f})"
                    )
                elif ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    text = ev.alternatives[0].text if ev.alternatives else ""
                    logger.debug(
                        f"[VADTriggeredStream:{self._stream_id}] STT INTERIM: '{text}'"
                    )
                else:
                    logger.debug(
                        f"[VADTriggeredStream:{self._stream_id}] STT event: {ev.type}"
                    )
                
                self._event_ch.send_nowait(ev)
            
            logger.debug(
                f"[VADTriggeredStream:{self._stream_id}] stt_forward_task ended - "
                f"processed {stt_event_count} events, filtered {filtered_count}"
            )
        
        @utils.log_exceptions(logger=logging.getLogger(__name__))
        async def audio_forward_task() -> None:
            """Forward audio to both VAD and STT, flush STT on VAD signal."""
            logger.debug(f"[VADTriggeredStream:{self._stream_id}] audio_forward_task started - forwarding audio")
            frame_count = 0
            external_flush_count = 0
            vad_triggered_flush_count = 0
            total_audio_duration = 0.0
            
            async for frame in self._input_ch:
                if isinstance(frame, self._FlushSentinel):
                    external_flush_count += 1
                    logger.debug(
                        f"[VADTriggeredStream:{self._stream_id}] External FlushSentinel received "
                        f"(#{external_flush_count}) - forwarding to STT"
                    )
                    # External flush - forward to STT
                    self._stt_stream.flush()
                    continue
                
                if isinstance(frame, rtc.AudioFrame):
                    frame_count += 1
                    frame_duration = frame.samples_per_channel / frame.sample_rate
                    total_audio_duration += frame_duration
                    
                    # Log every 100 frames to avoid spam
                    if frame_count % 100 == 0:
                        logger.debug(
                            f"[VADTriggeredStream:{self._stream_id}] Audio progress: "
                            f"{frame_count} frames, {total_audio_duration:.1f}s total"
                        )
                    
                    # Forward to VAD
                    self._vad_stream.push_frame(frame)
                    
                    # Forward to STT
                    self._stt_stream.push_frame(frame)
                    
                    # Check if VAD signaled a flush
                    if flush_event.is_set():
                        flush_event.clear()
                        vad_triggered_flush_count += 1
                        self._flush_count += 1
                        elapsed = time.monotonic() - self._start_time
                        logger.info(
                            f"[VADTriggeredStream:{self._stream_id}] VAD-triggered flush #{vad_triggered_flush_count} "
                            f"(elapsed={elapsed:.2f}s, frames={frame_count}, audio={total_audio_duration:.1f}s) "
                            f"- calling stt_stream.flush()"
                        )
                        self._stt_stream.flush()
                        logger.debug(f"[VADTriggeredStream:{self._stream_id}] flush() called on STT stream")
            
            # End of input. Close streams
            logger.debug(
                f"[VADTriggeredStream:{self._stream_id}] Input ended - "
                f"processed {frame_count} frames ({total_audio_duration:.1f}s audio), "
                f"{vad_triggered_flush_count} VAD-triggered flushes, {external_flush_count} external flushes"
            )
            self._stt_stream.end_input()
            self._vad_stream.end_input()
            logger.debug(f"[VADTriggeredStream:{self._stream_id}] end_input() called on both streams")
        
        # Run all tasks concurrently
        logger.info(f"[VADTriggeredStream:{self._stream_id}] Starting 3 concurrent tasks: vad_task, stt_forward_task, audio_forward_task")
        tasks = [
            asyncio.create_task(vad_task(), name=f"vad_task_{self._stream_id}"),
            asyncio.create_task(stt_forward_task(), name=f"stt_forward_task_{self._stream_id}"),
            asyncio.create_task(audio_forward_task(), name=f"audio_forward_task_{self._stream_id}"),
        ]
        
        try:
            await asyncio.gather(*tasks)
            logger.info(f"[VADTriggeredStream:{self._stream_id}] All tasks completed normally")
        except Exception as e:
            logger.error(f"[VADTriggeredStream:{self._stream_id}] Task error: {e}", exc_info=True)
            raise
        finally:
            logger.debug(f"[VADTriggeredStream:{self._stream_id}] Cleaning up - cancelling tasks and closing streams")
            await aio.cancel_and_wait(*tasks)
            if self._stt_stream:
                await self._stt_stream.aclose()
                logger.debug(f"[VADTriggeredStream:{self._stream_id}] STT stream closed")
            if self._vad_stream:
                await self._vad_stream.aclose()
                logger.debug(f"[VADTriggeredStream:{self._stream_id}] VAD stream closed")
            
            total_elapsed = time.monotonic() - self._start_time
            logger.info(
                f"[VADTriggeredStream:{self._stream_id}] Stream ended - "
                f"total_elapsed={total_elapsed:.2f}s, total_flushes={self._flush_count}"
            )
