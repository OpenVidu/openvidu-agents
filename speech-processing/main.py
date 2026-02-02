import asyncio
import logging
import os
import signal
import sys

from livekit.agents import (
    AgentServer,
    AutoSubscribe,
    JobContext,
    JobExecutorType,
    JobProcess,
    cli,
    stt,
    llm,
    WorkerType,
    WorkerPermissions,
    Agent,
    AgentSession,
    RoomOutputOptions,
    RoomInputOptions,
    RoomIO,
    utils,
    StopResponse,
)
from livekit import rtc
from livekit.plugins import silero

from stt_impl import get_stt_impl
from openviduagentutils.openvidu_agent import OpenViduAgent
from openviduagentutils.config_manager import ConfigManager
from livekit.agents.types import NotGiven

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
        logging.info(f"[Transcriber] Transcriber initialized for {participant_identity} (stt_provider={stt_impl.provider}, "
                     f"turn_detection={turn_detection}, vad_model={'None' if vad_model is None else type(vad_model).__name__})")

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
        self._sessions: dict[str, AgentSession] = {}
        self._tasks: set[asyncio.Task] = set()
        self._pending_sessions: set[str] = set()
        self._vad_model = None

    def start(self):
        self.ctx.room.on("participant_connected", self.on_participant_connected)
        self.ctx.room.on("participant_disconnected", self.on_participant_disconnected)

    async def aclose(self):
        await utils.aio.cancel_and_wait(*self._tasks)

        await asyncio.gather(
            *[self._close_session(session) for session in self._sessions.values()]
        )

        self.ctx.room.off("participant_connected", self.on_participant_connected)
        self.ctx.room.off("participant_disconnected", self.on_participant_disconnected)

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
                self._sessions[participant.identity] = task.result()
            finally:
                self._tasks.discard(task)
                self._pending_sessions.discard(participant.identity)
                logging.info(f"session started for {participant.identity}")

        task.add_done_callback(on_task_done)

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        self._pending_sessions.discard(participant.identity)

        if (session := self._sessions.pop(participant.identity, None)) is None:
            return

        logging.info(f"closing session for {participant.identity}")
        task = asyncio.create_task(self._close_session(session))
        self._tasks.add(task)
        task.add_done_callback(lambda _: self._tasks.discard(task))

    async def _start_session(self, participant: rtc.RemoteParticipant) -> AgentSession:
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

        room_io = RoomIO(
            agent_session=session,
            room=self.ctx.room,
            participant=participant,
            input_options=RoomInputOptions(
                text_enabled=False,
                audio_enabled=True,
                video_enabled=False,
                close_on_disconnect=True,
                delete_room_on_close=False,
            ),
            output_options=RoomOutputOptions(
                transcription_enabled=True,
                audio_enabled=False,
                sync_transcription=False,
            ),
        )
        await room_io.start()
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
            )
        )
        return session

    async def _close_session(self, sess: AgentSession) -> None:
        await sess.drain()
        await sess.aclose()

    def _get_vad_model(self):
        if self._vad_model is None:
            proc = getattr(self.ctx, "proc", None)
            userdata = getattr(proc, "userdata", None)
            if isinstance(userdata, dict):
                self._vad_model = userdata.get("vad")

        if self._vad_model is None:
            self._vad_model = silero.VAD.load(min_silence_duration=0.2)

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
    transcriber.start()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    for participant in ctx.room.remote_participants.values():
        # handle all existing participants
        transcriber.on_participant_connected(participant)

    async def cleanup():
        await transcriber.aclose()

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
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.2)
    # ######################################
    # TODO: use turn detection when required
    # ######################################
    # Turn detector models will be preloaded in the first entrypoint call
    # because they require job context to initialize
    # proc.userdata["turn_detectors"] = {}


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
            logging.info(
                "Preloading sherpa model for shared thread-based execution..."
            )
            # Creating an STT instance triggers recognizer loading into the cache
            stt_impl = get_stt_impl(agent_config)
            # Force model loading by ensuring the recognizer is created
            # The _ensure_recognizer() method loads the model into _RecognizerCache
            asyncio.run(stt_impl._ensure_recognizer())
            logging.info(
                "sherpa model preloaded successfully. Will be shared across all agent threads"
            )
    except Exception as e:
        logging.warning(
            f"Failed to preload sherpa model: {e}. Model will be loaded on first use."
        )


if __name__ == "__main__":

    # If calling "python main.py download-files" do not initialize the OpenViduAgent
    if len(sys.argv) > 1 and sys.argv[1] == "download-files":
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

    server = AgentServer(
        # Create AgentServer with THREAD executor to share local models across all agents
        # Significantly reduces memory usage when running multiple concurrent transcriptions
        job_executor_type=JobExecutorType.THREAD,
        ws_url=agent_config["ws_url"],
        api_key=agent_config["api_key"],
        api_secret=agent_config["api_secret"],
        load_threshold=load_threshold,
        max_retry=sys.maxsize,
        drain_timeout=sys.maxsize,
        # Vosk models may require sizable memory
        job_memory_warn_mb=1024,
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

    # Set agent name for explicit dispatch if configured
    if agent_config["live_captions"]["processing"] == "manual":
        server._agent_name = agent_name

    # Set worker type (ROOM = one agent per room)
    server._server_type = WorkerType.ROOM

    # Register the entrypoint
    @server.rtc_session()
    async def main_entrypoint(ctx: JobContext):
        await entrypoint(ctx)

    # Set up prewarm function
    server.setup_fnc = prewarm

    # Preload local STT models into memory before starting the server
    # This ensures all thread-based agents share the same model instance
    _preload_vosk_model(agent_config)
    _preload_sherpa_model(agent_config)

    logging.info(
        f"Starting agent {agent_name} with processing configured to {agent_config['live_captions']['processing']}"
    )
    logging.info(
        "Using JobExecutorType.THREAD for shared Vosk model memory across all agents"
    )

    # Redirect signal SIGQUIT as SIGTERM to allow graceful shutdown using livekit/agents mechanism
    signal.signal(
        signal.SIGQUIT, lambda signum, frame: os.kill(int(os.getpid()), signal.SIGTERM)
    )

    cli.run_app(server)
