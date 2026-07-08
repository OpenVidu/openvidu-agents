"""Teardown fixes for livekit-plugins-azure's streaming STT.

Why this exists (verified empirically with an object census, memray native
traces and coroutine await-chain dumps under a chaos soak):

livekit-plugins-azure 1.6.3/1.6.4 tears its native Azure SpeechRecognizer down
by calling only ``stop_continuous_recognition()`` and dropping the Python
references (SpeechStream._run's ``finally``). The ``PushAudioInputStream`` is
closed only on the clean "input ended" path (azure stt.py line 321), which is
unreachable in the voice-agent pipeline: session close CANCELS ``_run`` at the
``asyncio.wait`` and the error path raises before it. Without the stream
``close()`` the native audio pump never sees end-of-stream, so under Azure
connection errors (e.g. 4429 "parallel requests exceeded" / 1007 "quota
exceeded" remote closes) the blocking native stop can take tens of seconds.
The surrounding ``AgentSession.aclose()`` timeouts then abandon the coroutine,
and when the job's event loop stops, the coroutine freezes forever at
``await asyncio.to_thread(_cleanup)`` — pinning the native recognizer, the
audio buffers, and (via the task's contextvars Context -> JobContext ->
_shutdown_callbacks) every AgentSession of the room.

The patched ``_run`` below is upstream's implementation with a hardened
``finally`` block:
  1. ``self._stream.close()`` FIRST on every exit path (native EOS - this is
     what Microsoft's own samples do and what lets a stop on a dead session
     return promptly),
  2. a BOUNDED native stop (``asyncio.wait_for`` + ``asyncio.shield`` so a
     second cancellation cannot skip the cleanup),
  3. ``disconnect_all()`` on every connected EventSignal (breaks the
     recognizer -> signal -> bound-method -> stream -> recognizer reference
     cycle so refcounting frees the native handles immediately instead of
     waiting for a gen2 GC pass),
  4. ``_session_started_event`` is cleared on every loop iteration, so a
     retry waits for the NEW Azure session to start instead of pouring audio
     into a not-yet-connected native stream.

Remove once fixed upstream (tracked for livekit-plugins-azure > 1.6.4).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

_NATIVE_STOP_TIMEOUT_S = 15.0
_EOS_FLUSH_TIMEOUT_S = 15.0

_patched = False


def apply_azure_stt_teardown_fix() -> None:
    """Monkey-patch livekit.plugins.azure.stt.SpeechStream._run with fixed teardown."""
    global _patched
    if _patched:
        return

    import azure.cognitiveservices.speech as speechsdk  # type: ignore
    from livekit import rtc
    from livekit.agents import APIConnectionError, utils
    from livekit.plugins.azure import stt as azure_stt

    async def _run(self) -> None:  # noqa: C901 - mirrors upstream structure
        while True:
            self._session_stopped_event.clear()
            # upstream bug: never cleared, so retries skip the session-start wait
            self._session_started_event.clear()

            self._stream = speechsdk.audio.PushAudioInputStream(
                stream_format=speechsdk.audio.AudioStreamFormat(
                    samples_per_second=self._opts.sample_rate,
                    bits_per_sample=16,
                    channels=self._opts.num_channels,
                )
            )
            self._recognizer = azure_stt._create_speech_recognizer(
                config=self._opts, stream=self._stream
            )
            self._recognizer.recognizing.connect(self._on_recognizing)
            self._recognizer.recognized.connect(self._on_recognized)
            self._recognizer.speech_start_detected.connect(self._on_speech_start)
            self._recognizer.speech_end_detected.connect(self._on_speech_end)
            self._recognizer.session_started.connect(self._on_session_started)
            self._recognizer.session_stopped.connect(self._on_session_stopped)
            self._recognizer.canceled.connect(self._on_canceled)
            self._recognizer.start_continuous_recognition()

            try:
                try:
                    await asyncio.wait_for(
                        self._session_started_event.wait(), self._conn_options.timeout
                    )
                except asyncio.TimeoutError as e:
                    # must be an APIError so the base class retry/backoff logic
                    # applies; with _session_started_event now cleared per
                    # iteration this wait is real on every retry, and a bare
                    # TimeoutError would kill the stream AND the AgentSession
                    raise APIConnectionError(
                        "timed out waiting for Azure Speech session to start"
                    ) from e

                async def process_input() -> None:
                    async for input in self._input_ch:
                        if isinstance(input, rtc.AudioFrame):
                            self._audio_duration += input.duration
                            self._maybe_emit_recognition_usage()
                            self._stream.write(input.data.tobytes())
                        elif isinstance(input, self._FlushSentinel):
                            self._emit_recognition_usage()
                    self._emit_recognition_usage()

                process_input_task = asyncio.create_task(process_input())
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                wait_stopped_task = asyncio.create_task(self._session_stopped_event.wait())

                input_ended = False
                try:
                    done, _ = await asyncio.wait(
                        [process_input_task, wait_reconnect_task, wait_stopped_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        if task not in [wait_reconnect_task, wait_stopped_task]:
                            task.result()

                    if wait_stopped_task in done:
                        raise APIConnectionError("SpeechRecognition session stopped")

                    input_ended = wait_reconnect_task not in done
                    if not input_ended:
                        self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(process_input_task, wait_reconnect_task)

                # flush finals for buffered audio before teardown, on BOTH the
                # input-ended and the reconnect (update_options) paths, like
                # upstream — but bounded so a dead session cannot hang teardown
                self._stream.close()
                try:
                    await asyncio.wait_for(
                        self._session_stopped_event.wait(), _EOS_FLUSH_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    logging.warning(
                        "azure stt: session did not stop within %.0fs after EOS; "
                        "final transcripts may be truncated",
                        _EOS_FLUSH_TIMEOUT_S,
                    )
                if input_ended:
                    break
            finally:
                recognizer = self._recognizer
                stream = self._stream
                del self._recognizer

                def _cleanup() -> None:
                    # EOS first: unblocks the native audio pump so the stop
                    # below returns promptly even when Azure killed the
                    # connection (4429/1007); safe to call more than once.
                    with contextlib.suppress(Exception):
                        stream.close()
                    recognizer.stop_continuous_recognition()
                    # break the recognizer->EventSignal->bound-method->stream
                    # cycle so native handles release via refcounting
                    for sig in (
                        recognizer.recognizing,
                        recognizer.recognized,
                        recognizer.speech_start_detected,
                        recognizer.speech_end_detected,
                        recognizer.session_started,
                        recognizer.session_stopped,
                        recognizer.canceled,
                    ):
                        with contextlib.suppress(Exception):
                            sig.disconnect_all()

                try:
                    # shield: a second cancellation (e.g. AgentSession.aclose
                    # timeout) must not skip the native cleanup
                    await asyncio.wait_for(
                        asyncio.shield(asyncio.to_thread(_cleanup)),
                        timeout=_NATIVE_STOP_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    logging.warning(
                        "azure stt: native recognizer stop timed out after %.0fs; "
                        "abandoning cleanup thread",
                        _NATIVE_STOP_TIMEOUT_S,
                    )
                except asyncio.CancelledError:
                    # cleanup itself is shielded; re-raise after it was started
                    raise

    azure_stt.SpeechStream._run = _run
    _patched = True
    logging.info(
        "Applied azure STT teardown fix (stream EOS + bounded native stop + "
        "signal disconnect on all exit paths)"
    )
