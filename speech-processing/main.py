import asyncio
import contextlib
import ctypes
import ctypes.util
import gc
import logging
import multiprocessing
import os
import signal
import sys
import time


def _is_job_subprocess() -> bool:
    """True when called inside a job child process (JobExecutorType.PROCESS).

    Must be evaluated at CALL time, never at import time: multiprocessing
    re-imports this module (as __mp_main__) in every job subprocess during the
    bootstrap `prepare()` step, and parent_process() is only populated after
    that import completes (verified empirically under the Linux forkserver
    start method) — at import time a child looks like the worker.
    """
    return multiprocessing.parent_process() is not None


# Configure basic logging early so preload messages are visible.
# cli.run_app() will reconfigure this with proper formatting later.
# The handler basicConfig adds is remembered so job subprocesses can remove it
# in prewarm(): livekit's IPC log forwarding already re-emits child records in
# the parent, so a local stderr handler in a child would duplicate every log
# line. (Child detection is impossible at import time — see _is_job_subprocess.)
_handlers_before_basic_config = len(logging.getLogger().handlers)
logging.basicConfig(level=logging.INFO)
_early_log_handler = (
    logging.getLogger().handlers[-1]
    if len(logging.getLogger().handlers) > _handlers_before_basic_config
    else None
)

from livekit.agents import (
    AgentServer,
    AutoSubscribe,
    JobContext,
    JobExecutorType,
    JobProcess,
    cli,
    stt,
    WorkerPermissions,
    Agent,
    AgentSession,
)
from livekit.agents.voice.room_io import RoomOptions, TextOutputOptions
from livekit.agents.worker import ServerType
from livekit import rtc

from stt_impl import (
    get_stt_impl,
    set_cached_silero_vad,
    stt_provider_requires_vad,
)
from vad_stt_wrapper import VADTriggeredSTT
from openviduagentutils.openvidu_agent import OpenViduAgent
from openviduagentutils.config_manager import ConfigManager
from livekit.agents.types import NotGiven


def _resolve_malloc_trim():
    """Resolve glibc's malloc_trim, or None if unavailable.

    malloc_trim is a glibc extension (present in this agent's Debian-based image).
    On other libc implementations it won't exist and we degrade gracefully to a
    plain gc.collect() in _release_memory_to_os().
    """
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        return malloc_trim
    except (OSError, AttributeError):
        logging.warning(
            "malloc_trim not available; freed memory may not be returned to the OS"
        )
        return None


_malloc_trim = _resolve_malloc_trim()

# The job's MultiUserTranscriber is attached to the JobContext itself
# (ctx._transcriber) rather than a module-level dict keyed by room name: with
# JobExecutorType.THREAD, a room can be deleted and re-created with the same
# name while the previous job is still draining, and a name-keyed dict would
# cross the two jobs' transcribers.


def _release_memory_to_os() -> None:
    """Return memory freed by finished jobs/sessions back to the operating system.

    Closing an AgentSession drops its STT recognizer, but the underlying native
    memory is only returned to glibc's allocator free lists, not to the kernel, so
    the container RSS stays pinned at its high-water mark. Under
    JobExecutorType.THREAD all jobs run as threads in a single long-lived process,
    so that memory is never reclaimed by process exit either. We force a garbage collection so the
    recognizer's finalizer runs, then call malloc_trim() to hand the now-free heap
    pages back to the OS. Meant to be run in an executor to avoid blocking the loop.

    In a job subprocess (JobExecutorType.PROCESS) the full gc pass is skipped: the
    process exits when the room ends, which reclaims everything, and touching every
    object's refcount would dirty the copy-on-write pages shared with the
    forkserver, inflating per-process memory for no benefit.
    """
    if not _is_job_subprocess():
        gc.collect()
    if _malloc_trim is not None:
        _malloc_trim(0)


# ######################################
# TODO: use turn detection when required
# ######################################
# from livekit.plugins.turn_detector.english import EnglishModel
# from livekit.plugins.turn_detector.multilingual import MultilingualModel
# from stt_impl import get_best_turn_detector


class Transcriber(Agent):
    def __init__(
        self,
        *,
        participant_identity: str,
        stt_impl: stt.STT,
        turn_detection: object = "stt",
        vad_model: object | None = None,
    ):
        super().__init__(
            instructions="not-needed",
            stt=stt_impl,
            turn_detection=turn_detection,
            vad=vad_model,
        )
        self.participant_identity = participant_identity
        logging.info(
            f"[Transcriber] Transcriber initialized for {participant_identity} (stt_provider={stt_impl.provider}, "
            f"turn_detection={turn_detection}, vad_model={'None' if vad_model is None else type(vad_model).__name__})"
        )

    # async def on_user_turn_completed(
    #     self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage
    # ):
    #     import time
    #     user_transcript = new_message.text_content
    #     logging.info(
    #         f"[Transcriber] Turn completed for {self.participant_identity}: '{user_transcript}' "
    #         f"(timestamp={time.time():.3f})"
    #     )
    #     logging.debug(
    #         f"[Transcriber] Full message details: text_content='{new_message.text_content}', "
    #         f"role={new_message.role}"
    #     )

    #     raise StopResponse()


class MultiUserTranscriber:
    def __init__(self, ctx: JobContext, agent_config: object):
        self.ctx = ctx
        self.agent_config = agent_config
        # Maps participant_identity → (AgentSession, stt_impl). stt_impl is stored
        # so force_close_all() can be called on it when sess.aclose() times out.
        self._sessions: dict[str, tuple[AgentSession, stt.STT | None]] = {}
        self._tasks: set[asyncio.Task] = set()
        self._pending_sessions: set[str] = set()
        self._vad_model = None
        self._aclose_started = False

    def start(self):
        self.ctx.room.on("participant_connected", self.on_participant_connected)
        self.ctx.room.on("participant_disconnected", self.on_participant_disconnected)

    async def aclose(self):
        if self._aclose_started:
            logging.debug("[aclose] already in progress, skipping")
            return
        self._aclose_started = True
        logging.debug("[aclose] starting")

        # Await ongoing tasks with a timeout instead of cancelling them:
        # _close_session tasks MUST complete to release STT/VAD resources.
        # _start_session tasks will either complete or be abandoned after the timeout.
        if self._tasks:
            logging.debug(f"[aclose] waiting for {len(self._tasks)} task(s)")
            _, pending = await asyncio.wait(list(self._tasks), timeout=10.0)
            if pending:
                logging.warning(
                    f"[aclose] {len(pending)} task(s) still running after 10s timeout, continuing"
                )

        # Close sessions that never received a disconnect event
        if self._sessions:
            items = list(self._sessions.items())
            self._sessions.clear()
            logging.debug(
                f"[aclose] Closing {len(items)} remaining session(s) "
                f"with no prior disconnect event"
            )
            await asyncio.gather(
                *[self._close_session(sess, stt_impl) for _, (sess, stt_impl) in items],
                return_exceptions=True,
            )
            # Drop the stt_impl references held by `items` immediately.  Combined with
            # the `stt_impl = None` inside _close_session, this ensures all
            # VADTriggeredSTT objects are released before the gc pass below.
            del items

        self.ctx.room.off("participant_connected", self.on_participant_connected)
        self.ctx.room.off("participant_disconnected", self.on_participant_disconnected)

        self._pending_sessions.clear()

        # Final gc+malloc_trim pass now that all of this transcriber's sessions
        # are closed, returning the freed per-session memory to the OS.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _release_memory_to_os)

        logging.debug("[aclose] done")

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        if (
            participant.identity in self._sessions
            or participant.identity in self._pending_sessions
        ):
            return

        logging.info(f"starting session for {participant.identity}")
        task = asyncio.create_task(self._start_session(participant))
        self._tasks.add(task)
        self._pending_sessions.add(participant.identity)

        def on_task_done(task: asyncio.Task):
            try:
                sess, stt_impl = task.result()
                self._sessions[participant.identity] = (sess, stt_impl)
            finally:
                self._tasks.discard(task)
                self._pending_sessions.discard(participant.identity)
                logging.info(f"session started for {participant.identity}")

        task.add_done_callback(on_task_done)

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        self._pending_sessions.discard(participant.identity)

        entry = self._sessions.pop(participant.identity, None)
        if entry is None:
            return
        session, stt_impl = entry

        logging.info(f"closing session for {participant.identity}")
        task = asyncio.create_task(self._close_session(session, stt_impl))
        self._tasks.add(task)

        def on_close_done(_task: asyncio.Task):
            self._tasks.discard(_task)

        task.add_done_callback(on_close_done)

    async def _start_session(
        self, participant: rtc.RemoteParticipant
    ) -> tuple[AgentSession, stt.STT | None]:
        if participant.identity in self._sessions:
            return self._sessions[participant.identity]

        stt_impl = get_stt_impl(self.agent_config)

        vad_model = None
        turn_detection = "manual"

        # ######################################
        # TODO: use turn detection when required
        # ######################################
        # try:
        #     # Get cached turn detector from proc.userdata to avoid loading per participant
        #     turn_detection = self._get_turn_detector()
        #     logging.info(
        #         f"Determined optimal turn detector for participant {participant.identity}: {turn_detection}"
        #     )
        # except Exception as exc:
        #     logging.warning(
        #         "Failed to determine optimal turn detector, defaulting to 'vad': %s",
        #         exc,
        #     )
        #     turn_detection = "vad"
        # if turn_detection is NotGiven:
        #     turn_detection = "stt"

        if not stt_impl.capabilities.streaming:
            logging.info(
                f"Provider {stt_impl.provider} does not support streaming. Wrapping with StreamAdapter"
            )
            vad_model = self._get_vad_model()
            stt_impl = stt.StreamAdapter(stt=stt_impl, vad=vad_model)
            # ######################################
            # TODO: use turn detection when required
            # ######################################
            # if turn_detection == "stt":
            #     turn_detection = "vad"

        # If STT is VAD-wrapped (use_silero_vad=true), VAD is already integrated
        if stt_impl.provider.lower().startswith("vad-triggered/"):
            vad_model = None

        if turn_detection == "vad" and vad_model is None:
            vad_model = self._get_vad_model()

        session = AgentSession()

        # @session.on("user_input_transcribed")
        # def on_transcript(event):
        #     logging.info(f"[EVENT HANDLER CALLED] {participant.identity}")
        #     if event.is_final:
        #         logging.info(
        #             f"[EVENT] {participant.identity} FINAL -> {event.transcript}"
        #         )
        #     else:
        #         logging.info(
        #             f"[EVENT] {participant.identity} PARTIAL -> {event.transcript}"
        #         )

        logging.info(
            f"[MultiUserTranscriber] Starting Transcriber agent for {participant.identity} - "
            f"stt_provider={stt_impl.provider}, turn_detection={turn_detection}, "
            f"vad_model={'None' if vad_model is None else type(vad_model).__name__}"
        )
        await session.start(
            agent=Transcriber(
                participant_identity=participant.identity,
                stt_impl=stt_impl,
                turn_detection=turn_detection,
                vad_model=vad_model,
            ),
            room=self.ctx.room,
            room_options=RoomOptions(
                participant_identity=participant.identity,
                text_input=False,
                video_input=False,
                audio_output=False,
                text_output=TextOutputOptions(sync_transcription=False),
                close_on_disconnect=True,
            ),
        )
        return session, stt_impl

    async def _close_session(
        self,
        sess: AgentSession,
        stt_impl: stt.STT | None = None,
    ) -> None:
        t0 = time.monotonic()
        logging.debug("[_close_session] starting drain()")
        try:
            await asyncio.wait_for(sess.drain(), timeout=5.0)
        except asyncio.TimeoutError:
            logging.warning(
                "[_close_session] drain() timed out after 5s, proceeding to aclose()"
            )
        except RuntimeError as e:
            if "isn't running" not in str(e):
                raise
            # livekit's RoomIO close_on_disconnect already closed the session
            # ("AgentSession isn't running"); continue so the memory release
            # below still runs instead of leaking this task's exception.
            logging.debug("[_close_session] session already closed, skipping drain")
        try:
            await asyncio.wait_for(sess.aclose(), timeout=10.0)
        except asyncio.TimeoutError:
            logging.warning(
                "[_close_session] aclose() timed out after 10s, proceeding to memory release"
            )
            # sess.aclose() timed out: the VAD/STT pipeline tasks are still running
            # because the audio input was never cleanly closed (participant never
            # sent a proper disconnect signal). Force-cancel them.
            if isinstance(stt_impl, VADTriggeredSTT):
                logging.warning("[_close_session] force-closing VAD stream tasks")
                await stt_impl.force_close_all()
        # Drop stt_impl NOW so CPython's reference counter can immediately collect
        # the VADTriggeredSTT object and decrement the shared Silero VAD model's
        # refcount.  If this was the last holder, gc.collect() below will free it.
        stt_impl = None  # type: ignore[assignment]
        # Drop sess so the AgentSession (and its internal STT pipeline node, which
        # holds a reference to stt_impl via Agent.stt) is no longer reachable from
        # this frame when gc.collect() runs below.  Without this, the AgentSession
        # object graph keeps VADTriggeredSTT / VoskSpeechStream alive even after
        # stt_impl = None above, because AgentSession still references the Agent
        # whose .stt field points at the same VADTriggeredSTT instance.
        sess = None  # type: ignore[assignment]
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _release_memory_to_os)
        logging.debug(
            f"[_close_session] done, total close time {time.monotonic() - t0:.2f}s"
        )

    def _get_vad_model(self):
        if self._vad_model is None:
            proc = getattr(self.ctx, "proc", None)
            userdata = getattr(proc, "userdata", None)
            if isinstance(userdata, dict):
                self._vad_model = userdata.get("vad")

        if self._vad_model is None:
            # Fallback: use the module-level cached VAD, loading on-demand if needed
            # This handles edge cases where VAD is required at runtime but wasn't preloaded
            from stt_impl import _get_cached_silero_vad

            self._vad_model = _get_cached_silero_vad(load_if_missing=True)

        return self._vad_model

    # ######################################
    # TODO: use turn detection when required
    # ######################################
    # def _get_turn_detector(self):
    #     """Get cached turn detector from proc.userdata to share across participants."""
    #     proc = getattr(self.ctx, "proc", None)
    #     userdata = getattr(proc, "userdata", None)
    #     if isinstance(userdata, dict):
    #         turn_detectors = userdata.get("turn_detectors", {})
    #         if turn_detectors:
    #             # Return cached turn detector based on config
    #             try:
    #                 return get_best_turn_detector(self.agent_config, preloaded_models=turn_detectors)
    #             except Exception:
    #                 pass

    #     # Fallback: create new instance if cache unavailable
    #     return get_best_turn_detector(self.agent_config)


# ######################################
# TODO: use turn detection when required
# ######################################
# def _preload_turn_detector_models() -> dict[str, object]:
#     """Preload turn detector models. Must be called within a job context."""
#     loaded_models: dict[str, object] = {}

#     try:
#         loaded_models["english"] = EnglishModel()
#         logging.info("Preloaded English turn detector model")
#     except Exception as exc:
#         logging.warning("Failed to preload English turn detector: %s", exc)

#     try:
#         loaded_models["multilingual"] = MultilingualModel()
#         logging.info("Preloaded Multilingual turn detector model")
#     except Exception as exc:
#         logging.warning("Failed to preload multilingual turn detector: %s", exc)

#     return loaded_models

# ######################################
# TODO: use turn detection when required
# ######################################
# def _ensure_turn_detectors_loaded(ctx: JobContext) -> None:
#     """Ensure turn detector models are loaded (called once from first entrypoint).

#     Turn detector models require job context to initialize, so they cannot be
#     preloaded in prewarm(). This function loads them on the first job and caches
#     them in proc.userdata for all subsequent participants to share.
#     """
#     proc = getattr(ctx, "proc", None)
#     userdata = getattr(proc, "userdata", None)

#     if not isinstance(userdata, dict):
#         return

#     # Check if already loaded
#     turn_detectors = userdata.get("turn_detectors", {})
#     if turn_detectors:
#         logging.debug("Turn detector models already loaded, skipping")
#         return

#     logging.info("Preloading turn detector models (first job, requires job context)...")
#     userdata["turn_detectors"] = _preload_turn_detector_models()


async def main_entrypoint(ctx: JobContext) -> None:
    """Job entrypoint. Defined at module level (NOT inside the __main__ guard) so
    it stays importable/picklable in job child processes when the server runs
    with JobExecutorType.PROCESS (the child re-imports this module with
    __name__ != "__main__", so anything defined inside the guard is missing)."""
    # Add custom log context fields
    ctx.log_context_fields = {
        "worker_id": ctx.worker_id,
        "room_name": ctx.room.name,
    }
    await entrypoint(ctx)


async def session_end(ctx: JobContext) -> None:
    """Close all sessions BEFORE the framework calls room.disconnect().

    room.disconnect() blocks until every AudioStream subscription is released by the
    FFI.  AudioStreams are owned by AgentSession objects, so if any session is still
    open when room.disconnect() runs, the FFI deadlocks for the full close_timeout
    (30 s).  This callback is invoked by the framework *before* room.disconnect(), so
    closing sessions here lets the FFI proceed without the timeout.
    """
    transcriber = getattr(ctx, "_transcriber", None)
    ctx._transcriber = None
    try:
        if transcriber is not None:
            await transcriber.aclose()
    finally:
        await _release_room_ffi_subscription(ctx.room)


async def _release_room_ffi_subscription(room: rtc.Room) -> None:
    """Work around a livekit rtc leak, still present in 1.1.12: when a room is
    disconnected server-side (room deleted / connection lost), Room.disconnect()
    early-returns without unsubscribing the room's process-global FFI event
    queue, and because no FFI 'eos' event is emitted on a remote close, the
    room's listen task never ends. With the THREAD executor the job's event loop
    object additionally outlives the job un-closed, so the leaked subscription
    keeps accumulating EVERY FFI event in the process (audio-frame events of all
    other rooms' tracks, ~100/s per track) into the dead loop forever.

    livekit>=1.1.10 (python-sdks PR #699) unsubscribes when the listen task
    ENDS, and 1.1.12 also unsubscribes in Room.__del__ — but NEITHER fires on
    this path: without 'eos' the listen task never ends, and the Room can never
    be garbage-collected because the global subscriber list transitively pins it
    (subscribers -> queue -> pending get() future -> listen task -> coroutine ->
    Room). Keep this until upstream emits 'eos' on remote disconnect or drops
    the disconnect() early-return; to re-check, remove the calls and verify the
    FFI subscriber count returns to baseline after an azure chaos soak.
    """
    if room.isconnected():
        # a proper room.disconnect() (which the framework calls right after
        # session_end) unsubscribes the queue itself
        return
    try:
        from livekit.rtc._ffi_client import FfiClient

        task = getattr(room, "_task", None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
            # run the cleanups _listen_task's cancellation skipped (they only
            # run after an 'eos' break upstream)
            for drain_name in (
                "_drain_rpc_invocation_tasks",
                "_drain_data_stream_tasks",
            ):
                drain = getattr(room, drain_name, None)
                if drain is not None:
                    with contextlib.suppress(Exception):
                        await drain()
        ffi_queue = getattr(room, "_ffi_queue", None)
        if ffi_queue is not None:
            queue = FfiClient.instance.queue
            with queue._lock:
                was_subscribed = any(q is ffi_queue for q, _, _ in queue._subscribers)
            if was_subscribed:
                queue.unsubscribe(ffi_queue)
                logging.info(
                    f"[leak-fix] released leaked FFI queue subscription of "
                    f"disconnected room {room.name}"
                )
    except Exception as e:  # noqa: BLE001 - defensive cleanup must never fail the job
        logging.warning(f"[leak-fix] room FFI cleanup failed: {e}")


async def entrypoint(ctx: JobContext):
    # ######################################
    # TODO: use turn detection when required
    # ######################################
    # Preload turn detector models on first job (they require job context)
    # These will be cached in proc.userdata for all subsequent participants
    # _ensure_turn_detectors_loaded(ctx)

    openvidu_agent = OpenViduAgent.get_instance()

    agent_config = openvidu_agent.get_agent_config()

    transcriber = MultiUserTranscriber(ctx, agent_config)
    # Attach so session_end() can close it before room.disconnect() is called.
    ctx._transcriber = transcriber
    transcriber.start()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    for participant in ctx.room.remote_participants.values():
        # handle all existing participants
        transcriber.on_participant_connected(participant)

    async def cleanup():
        # session_end() normally runs first (before room.disconnect()), so
        # _aclose_started will be True here and this becomes a fast no-op.
        ctx._transcriber = None
        await transcriber.aclose()
        # Shutdown callbacks run AFTER the framework's room.disconnect(), so
        # this also covers the race where the room disconnects between
        # session_end()'s check and the framework's disconnect() call.
        await _release_room_ffi_subscription(ctx.room)

    ctx.add_shutdown_callback(cleanup)


# async def _forward_transcription(
#     stt_stream: stt.SpeechStream,
#     stt_forwarder: transcription.STTSegmentsForwarder,
# ):
#     """Forward the transcription to the client and log the transcript in the console"""
#     async for ev in stt_stream:
#         if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
#             # you may not want to log interim transcripts, they are not final and may be incorrect
#             logging.info(
#                 f"{stt_forwarder._participant_identity} is saying -> {ev.alternatives[0].text}"
#             )
#         elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
#             logging.info(
#                 f"{stt_forwarder._participant_identity} said -> {ev.alternatives[0].text}"
#             )
#         elif ev.type == stt.SpeechEventType.RECOGNITION_USAGE:
#             logging.debug(f"metrics: {ev.recognition_usage}")

#         stt_forwarder.update(ev)


# async def entrypoint(ctx: JobContext) -> None:
#     openvidu_agent = OpenViduAgent.get_instance()
#     openvidu_agent.new_active_job(ctx)

#     agent_config = openvidu_agent.get_agent_config()
#     agent_name = openvidu_agent.get_agent_name()

#     print(f"Agent {agent_name} joining room {ctx.room.name}")

#     stt_impl = get_stt_impl(agent_config)

#     if not stt_impl.capabilities.streaming:
#         # wrap with a stream adapter to use streaming semantics
#         stt_impl = stt.StreamAdapter(
#             stt=stt_impl,
#             vad=silero.VAD.load(
#                 min_silence_duration=0.2,
#             ),
#         )

#     async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
#         audio_stream = rtc.AudioStream(track)
#         stt_forwarder = transcription.STTSegmentsForwarder(
#             room=ctx.room, participant=participant, track=track
#         )

#         print(
#             f"Agent {agent_name} transcribing audio track {track.sid} from participant {participant.identity}"
#         )

#         stt_stream = stt_impl.stream()
#         asyncio.create_task(_forward_transcription(stt_stream, stt_forwarder))

#         async for ev in audio_stream:
#             stt_stream.push_frame(ev.frame)

#         stt_stream.end_input()

#     @ctx.room.on("track_subscribed")
#     def on_track_subscribed(
#         track: rtc.Track,
#         publication: rtc.TrackPublication,
#         participant: rtc.RemoteParticipant,
#     ):
#         # spin up a task to transcribe each track
#         if track.kind == rtc.TrackKind.KIND_AUDIO:
#             asyncio.create_task(transcribe_track(participant, track))

#     await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)


def prewarm(proc: JobProcess):
    in_job_subprocess = _is_job_subprocess()
    if in_job_subprocess and _early_log_handler is not None:
        # Drop the import-time stderr handler: livekit's LogQueueHandler forwards
        # this child's records to the parent, which re-emits them — keeping the
        # local handler would duplicate every log line.
        logging.getLogger().removeHandler(_early_log_handler)

    # Reuse the preloaded Silero VAD model from the main process, if it was preloaded.
    # VAD is only preloaded when the STT provider requires it (non-streaming or use_silero_vad=true)
    from stt_impl import _get_cached_silero_vad

    cached_vad = _get_cached_silero_vad()
    if cached_vad is None and in_job_subprocess:
        # Under JobExecutorType.PROCESS this runs inside a fresh (warm) job
        # subprocess where the parent's __main__ preload is invisible, so load
        # the VAD here when the configured provider needs it. This runs before
        # any job is assigned to the process, keeping the load off the caption
        # critical path (+~23 MB RSS per process).
        try:
            agent_config = OpenViduAgent.get_instance().get_agent_config()
            if stt_provider_requires_vad(agent_config):
                logging.info("Prewarm: loading Silero VAD in job subprocess")
                cached_vad = _get_cached_silero_vad(load_if_missing=True)
        except (
            Exception,
            SystemExit,
        ) as e:  # noqa: BLE001 - prewarm must never kill the process (config loader calls exit())
            logging.warning(f"Prewarm VAD load failed (will load on-demand): {e}")

    if cached_vad is not None:
        proc.userdata["vad"] = cached_vad
        logging.debug("Using preloaded Silero VAD model in prewarm")
    else:
        # VAD not preloaded - this is expected when provider doesn't need VAD
        # Don't load it here to save memory. It will be loaded on-demand if needed.
        logging.debug("Silero VAD not preloaded - will be loaded on-demand if needed")

    # ######################################
    # TODO: use turn detection when required
    # ######################################
    # Turn detector models will be preloaded in the first entrypoint call
    # because they require job context to initialize
    # proc.userdata["turn_detectors"] = {}


def _preload_silero_vad() -> None:
    """Preload Silero VAD model into memory for sharing across threads.

    When using JobExecutorType.THREAD, all agent threads share the same process memory.
    This function loads the Silero VAD model once at startup so all subsequent uses
    reuse the cached model via stt_impl's internal cache.
    """
    try:
        logging.info("Preloading Silero VAD model for shared thread-based execution...")
        from livekit.plugins import silero

        vad_model = silero.VAD.load()
        set_cached_silero_vad(vad_model)
        logging.info(
            "Silero VAD model preloaded successfully. Will be shared across all agent threads"
        )
    except Exception as e:
        logging.warning(
            f"Failed to preload Silero VAD model: {e}. Model will be loaded on first use."
        )


def _preload_vosk_model(agent_config) -> None:
    """Preload Vosk model into memory for sharing across threads.

    When using JobExecutorType.THREAD, all agent threads share the same process memory.
    This function loads the Vosk model once at startup so all subsequent STT instances
    reuse the cached model via livekit-plugins-vosk's internal _ModelCache.
    """
    try:
        stt_provider = agent_config.get("live_captions", {}).get("provider")
        if stt_provider == "vosk":
            logging.info("Preloading Vosk model for shared thread-based execution...")
            # Creating an STT instance triggers model loading into the cache
            stt_impl = get_stt_impl(agent_config)
            # Force model loading by calling a method that requires the model
            # The model will be cached and shared across all thread-based jobs
            logging.info(
                "Vosk model preloaded successfully. Will be shared across all agent threads"
            )
    except Exception as e:
        logging.warning(
            f"Failed to preload Vosk model: {e}. Model will be loaded on first use."
        )


def _preload_sherpa_model(agent_config) -> None:
    """Preload sherpa model into memory for sharing across threads.

    When using JobExecutorType.THREAD, all agent threads share the same process memory.
    This function loads the sherpa model once at startup so all subsequent STT instances
    reuse the cached recognizer via livekit-plugins-sherpa's internal _RecognizerCache.
    """
    try:
        stt_provider = agent_config.get("live_captions", {}).get("provider")
        if stt_provider == "sherpa":
            logging.info("Preloading sherpa model for shared thread-based execution...")
            # Creating an STT instance triggers recognizer loading into the cache
            stt_impl = get_stt_impl(agent_config)

            # If wrapped in VADTriggeredSTT, get the underlying sherpa STT
            from vad_stt_wrapper import VADTriggeredSTT

            if isinstance(stt_impl, VADTriggeredSTT):
                sherpa_stt = stt_impl._stt
            else:
                sherpa_stt = stt_impl

            # Force model loading by ensuring the recognizer is created
            # The _ensure_recognizer() method loads the model into _RecognizerCache
            asyncio.run(sherpa_stt._ensure_recognizer())
            logging.info(
                "sherpa model preloaded successfully. Will be shared across all agent threads"
            )
    except Exception as e:
        logging.warning(
            f"Failed to preload sherpa model: {e}. Model will be loaded on first use."
        )


def _preload_nemotron_model(agent_config) -> None:
    """Preload the Nemotron model into memory for sharing across threads.

    When using JobExecutorType.THREAD, all agent threads share the same process
    memory. This loads the ~2.4 GB GPU model once at startup so all subsequent
    STT instances reuse the cached model via livekit-plugins-nemotron's internal
    _ModelCache, instead of paying the multi-second load on the caption path.
    """
    try:
        stt_provider = agent_config.get("live_captions", {}).get("provider")
        if stt_provider == "nemotron":
            logging.info(
                "Preloading nemotron model for shared thread-based execution..."
            )
            # Creating an STT instance triggers model loading into the cache.
            stt_impl = get_stt_impl(agent_config)

            # If wrapped in VADTriggeredSTT, get the underlying nemotron STT.
            from vad_stt_wrapper import VADTriggeredSTT

            if isinstance(stt_impl, VADTriggeredSTT):
                nemo_stt = stt_impl._stt
            else:
                nemo_stt = stt_impl

            # Force model loading by ensuring it is in the shared cache.
            asyncio.run(nemo_stt._ensure_model())
            # Absorb the one-time lazy-init cost (numba JIT of the RNNT greedy decoder, ~3.5s on CPU)
            # with a dummy utterance, so the first real caption is as fast as subsequent ones.
            # Non-fatal: on failure the first utterance simply pays that cost instead.
            try:
                asyncio.run(nemo_stt.warmup())
            except Exception as warmup_error:
                logging.warning(
                    f"nemotron warm-up failed (non-fatal, first utterance will "
                    f"pay the one-time JIT cost): {warmup_error}"
                )
            logging.info(
                "nemotron model preloaded successfully. Will be shared across all agent threads"
            )
    except Exception as e:
        logging.warning(
            f"Failed to preload nemotron model: {e}. Model will be loaded on first use."
        )


if __name__ == "__main__":

    # If calling "python main.py download-files" do not initialize the OpenViduAgent
    if len(sys.argv) > 1 and sys.argv[1] == "download-files":
        from livekit.plugins import silero

        silero.VAD.load()

        # ######################################
        # TODO: use turn detection when required
        # ######################################
        # _preload_turn_detector_models()

        # Create a minimal server just for download-files
        server = AgentServer()

        @server.rtc_session()
        async def download_entrypoint(ctx: JobContext):
            await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        cli.run_app(server)
        logging.info("Files downloaded for all plugins")
        sys.exit(0)

    openvidu_agent = OpenViduAgent.get_instance()
    agent_config = openvidu_agent.get_agent_config()
    agent_name = openvidu_agent.get_agent_name()

    config_manager = ConfigManager(agent_config, "")
    load_threshold = config_manager.optional_numeric_value("load_threshold", 0.7)
    if load_threshold < 0 or load_threshold > 1:
        logging.error("load_threshold must be a number between 0 and 1")
        sys.exit(1)

    # Decide the job executor type from the configured provider:
    #  - PROCESS for every cloud provider. Each Room job runs in its own OS
    #    process, so ALL resources (including native SDK memory) are reclaimed
    #    by the kernel when the Room closes, hung jobs can be killed, and
    #    job_memory_warn_mb/job_memory_limit_mb are actually enforced.
    #    Non-streaming cloud providers (openai, azure_openai, groq, fal, clova,
    #    spitch, simplismart, and the default elevenlabs/mistralai models) need
    #    the Silero VAD StreamAdapter wrapper: each warm job subprocess loads
    #    its own VAD copy in prewarm() (+~23 MB per concurrent Room, loaded off
    #    the caption critical path).
    #  - THREAD only for vosk/sherpa/nemotron (with or without VAD): their large
    #    local ASR models are shared across jobs by design, and one process is
    #    what lets every job reuse a single copy.
    # Env override for testing: JOB_EXECUTOR_TYPE=thread|process. Note that
    # forcing 'process' for vosk/sherpa/nemotron makes every job child load its
    # own copy of the ASR model on the job critical path — testing escape hatch only.
    provider_requires_vad = stt_provider_requires_vad(agent_config)
    stt_provider = agent_config.get("live_captions", {}).get("provider")

    _executor_override = os.getenv("JOB_EXECUTOR_TYPE", "").strip().lower()
    if _executor_override in ("thread", "process"):
        job_executor_type = (
            JobExecutorType.THREAD
            if _executor_override == "thread"
            else JobExecutorType.PROCESS
        )
        logging.info(
            f"Job executor type overridden via JOB_EXECUTOR_TYPE env: "
            f"{job_executor_type.name}"
        )
        if job_executor_type == JobExecutorType.PROCESS and stt_provider in (
            "vosk",
            "sherpa",
            "nemotron",
        ):
            logging.warning(
                "JOB_EXECUTOR_TYPE=process forced for a local-model provider: "
                "every Room job subprocess will load its own copy of the ASR "
                "model on the job critical path"
            )
    elif _executor_override:
        logging.warning(
            f"Unrecognized JOB_EXECUTOR_TYPE value '{_executor_override}' - "
            f"ignoring and selecting automatically"
        )
        job_executor_type = (
            JobExecutorType.THREAD
            if stt_provider in ("vosk", "sherpa", "nemotron")
            else JobExecutorType.PROCESS
        )
    elif stt_provider in ("vosk", "sherpa", "nemotron"):
        job_executor_type = JobExecutorType.THREAD
    else:
        job_executor_type = JobExecutorType.PROCESS

    logging.info(
        f"Using job executor type {job_executor_type.name} for provider "
        f"'{stt_provider}' (local ASR model: {stt_provider in ('vosk', 'sherpa', 'nemotron')}, "
        f"needs Silero VAD: {provider_requires_vad})"
    )

    server_kwargs = {}
    if job_executor_type == JobExecutorType.PROCESS:
        # The prod default is one warm process per CPU core, which multiplies
        # the idle footprint on large machines for no benefit here: a small
        # warm pool keeps job assignment instant while the pool refills
        # asynchronously in the background.
        server_kwargs["num_idle_processes"] = 2
        # Job subprocess initialization (prewarm) may import the provider
        # plugin and load the Silero VAD; give it comfortable headroom over
        # livekit's 10s default so a CPU-busy host cannot kill warm children
        # mid-initialization.
        server_kwargs["initialize_process_timeout"] = 60.0
        if provider_requires_vad:
            # Import (only) the silero plugin in the supervisor so it gets
            # registered with livekit and therefore included in the forkserver
            # preload list: job subprocesses then inherit the module pages via
            # copy-on-write instead of each paying the full cold import. The
            # VAD model itself is still loaded per child in prewarm().
            import livekit.plugins.silero  # noqa: F401

    server = AgentServer(
        job_executor_type=job_executor_type,
        **server_kwargs,
        ws_url=agent_config["ws_url"],
        api_key=agent_config["api_key"],
        api_secret=agent_config["api_secret"],
        load_threshold=load_threshold,
        max_retry=sys.maxsize,
        drain_timeout=sys.maxsize,
        # Local models may require sizable memory
        job_memory_warn_mb=2048,
        permissions=WorkerPermissions(
            # no need to publish tracks
            can_publish=False,
            # must subscribe to audio tracks
            can_subscribe=True,
            # mandatory to send transcription events
            can_publish_data=True,
            # when set to true, the agent won't be visible to others in the room.
            # when hidden, it will also not be able to publish tracks to the room as it won't be visible.
            hidden=True,
        ),
    )

    # Set agent name for explicit dispatch only in manual processing mode.
    if agent_config["live_captions"]["processing"] == "manual":
        server.rtc_session(
            type=ServerType.ROOM, agent_name=agent_name, on_session_end=session_end
        )(main_entrypoint)
    else:
        server.rtc_session(type=ServerType.ROOM, on_session_end=session_end)(
            main_entrypoint
        )

    # Preload local models into memory before starting the server.
    # THREAD only: it ensures all thread-based agents share the same model
    # instance. Under PROCESS the supervisor never runs jobs, so preloading here
    # would only bloat it — each warm job subprocess prewarms its own resources
    # (see prewarm()).
    if job_executor_type == JobExecutorType.THREAD:
        if provider_requires_vad:
            _preload_silero_vad()
        else:
            logging.info(
                "Skipping Silero VAD preload. Not needed for configured provider"
            )
        _preload_vosk_model(agent_config)
        _preload_sherpa_model(agent_config)
        _preload_nemotron_model(agent_config)
    else:
        logging.info(
            "Skipping model preloads: PROCESS executor prewarms per-job subprocesses"
        )

    # Set up prewarm function
    server.setup_fnc = prewarm

    logging.info(
        f"Starting agent {agent_name} with processing configured to {agent_config['live_captions']['processing']}"
    )

    # Redirect signal SIGQUIT as SIGTERM to allow graceful shutdown using livekit/agents mechanism
    signal.signal(
        signal.SIGQUIT, lambda signum, frame: os.kill(int(os.getpid()), signal.SIGTERM)
    )

    cli.run_app(server)
