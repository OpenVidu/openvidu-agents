#!/usr/bin/env python3
"""Tests how a VADTriggeredSpeechStream ends and cleans up its inner streams.

The wrapper owns an inner STT stream and an inner VAD stream. Both must end
their input exactly once (livekit-agents raises RuntimeError on a second
end_input()) and be closed, and the wrapper must deregister itself from
VADTriggeredSTT._active_streams, however the stream ends:

- the caller ends input and drains it, then leaves `async with` (aclose());
- the caller leaves `async with` without ending input, which is what the
  default livekit-agents stt_node does when the session closes.

The STT and VAD are fakes: nothing here loads a model.
"""

import asyncio
import unittest

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, stt, vad

from vad_stt_wrapper import VADTriggeredSTT

SAMPLE_RATE = 16000
FRAME_SAMPLES = SAMPLE_RATE // 100  # 10 ms


def _frames(count: int = 50):
    for _ in range(count):
        yield rtc.AudioFrame(b"\0" * FRAME_SAMPLES * 2, SAMPLE_RATE, 1, FRAME_SAMPLES)


class _InnerStreamProbe:
    """Counts end_input() calls and records aclose() on an inner stream."""

    def __init__(self) -> None:
        self.end_input_calls = 0
        self.closed = False

    def record_end_input(self) -> None:
        self.end_input_calls += 1

    def record_aclose(self) -> None:
        self.closed = True


class FakeSpeechStream(stt.SpeechStream):
    """Emits one FINAL_TRANSCRIPT per flush that follows some audio."""

    def __init__(self, *, stt: "FakeSTT", conn_options) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self.probe = _InnerStreamProbe()
        self.frames_received = 0

    def end_input(self) -> None:
        self.probe.record_end_input()
        super().end_input()

    async def aclose(self) -> None:
        self.probe.record_aclose()
        await super().aclose()

    async def _run(self) -> None:
        pending_audio = False
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                if pending_audio:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[stt.SpeechData(language="en", text="final")],
                        )
                    )
                    pending_audio = False
            else:
                self.frames_received += 1
                pending_audio = True


class FakeSTT(stt.STT):
    def __init__(self) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=False)
        )
        self.streams: list[FakeSpeechStream] = []

    async def _recognize_impl(self, buffer, *, language=None, conn_options=None):
        raise NotImplementedError

    def stream(self, *, language=None, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        s = FakeSpeechStream(stt=self, conn_options=conn_options)
        self.streams.append(s)
        return s


class FakeVADStream(vad.VADStream):
    """Consumes audio and never reports speech boundaries."""

    def __init__(self, vad_impl: "FakeVAD") -> None:
        super().__init__(vad_impl)
        self.probe = _InnerStreamProbe()

    def end_input(self) -> None:
        self.probe.record_end_input()
        super().end_input()

    async def aclose(self) -> None:
        self.probe.record_aclose()
        await super().aclose()

    async def _main_task(self) -> None:
        async for _ in self._input_ch:
            pass


class FakeVAD(vad.VAD):
    def __init__(self) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.01))
        self.streams: list[FakeVADStream] = []

    def stream(self) -> FakeVADStream:
        s = FakeVADStream(self)
        self.streams.append(s)
        return s


class VADTriggeredStreamShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.base_stt = FakeSTT()
        self.base_vad = FakeVAD()
        self.wrapped = VADTriggeredSTT(stt_impl=self.base_stt, vad_impl=self.base_vad)

    def _inner_probes(self) -> tuple[_InnerStreamProbe, _InnerStreamProbe]:
        self.assertEqual(len(self.base_stt.streams), 1)
        self.assertEqual(len(self.base_vad.streams), 1)
        return self.base_stt.streams[0].probe, self.base_vad.streams[0].probe

    def _assert_cleaned_up(self) -> None:
        stt_probe, vad_probe = self._inner_probes()
        self.assertEqual(stt_probe.end_input_calls, 1)
        self.assertEqual(vad_probe.end_input_calls, 1)
        self.assertTrue(stt_probe.closed)
        self.assertTrue(vad_probe.closed)
        self.assertEqual(self.wrapped._active_streams, [])

    async def _wait_until_inner_stt_received(self, frames: int) -> None:
        async def poll():
            while not self.base_stt.streams or self.base_stt.streams[0].frames_received < frames:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(poll(), timeout=5)

    async def test_caller_ending_input_ends_inner_input_once_and_cleans_up(self):
        finals = []
        async with self.wrapped.stream() as stream:
            for frame in _frames():
                stream.push_frame(frame)
            stream.end_input()

            async def drain():
                async for ev in stream:
                    if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        finals.append(ev.alternatives[0].text)

            # Before the fix the stream failed here with
            # "RuntimeError: ...FakeSpeechStream input ended"
            await asyncio.wait_for(drain(), timeout=5)

        self.assertEqual(finals, ["final"])
        self._assert_cleaned_up()

    async def test_aclose_without_ending_input_still_ends_inner_input(self):
        async with self.wrapped.stream() as stream:
            for frame in _frames():
                stream.push_frame(frame)
            await self._wait_until_inner_stt_received(50)

        self._assert_cleaned_up()


if __name__ == "__main__":
    unittest.main()
