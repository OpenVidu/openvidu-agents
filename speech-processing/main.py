import asyncio
import logging
import os
import signal
import sys

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    stt,
    llm,
    WorkerType,
    WorkerPermissions,
    JobProcess,
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

from stt_impl import get_stt_impl, get_best_turn_detector
from openviduagentutils.openvidu_agent import OpenViduAgent
from openviduagentutils.config_manager import ConfigManager
from livekit.agents.types import NotGiven


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

    async def on_user_turn_completed(
        self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ):
        user_transcript = new_message.text_content
        logging.info(f"{self.participant_identity} -> {user_transcript}")

        raise StopResponse()


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
        try:
            turn_detection = get_best_turn_detector(self.agent_config)
        except Exception as exc:
            logging.warning(
                "Failed to determine optimal turn detector, defaulting to 'vad': %s",
                exc,
            )
            turn_detection = "vad"

        if turn_detection is NotGiven:
            turn_detection = "stt"

        if not stt_impl.capabilities.streaming:
            vad_model = self._get_vad_model()
            stt_impl = stt.StreamAdapter(stt=stt_impl, vad=vad_model)
            if turn_detection == "stt":
                turn_detection = "vad"

        if turn_detection == "vad" and vad_model is None:
            vad_model = self._get_vad_model()

        session = AgentSession()
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


async def entrypoint(ctx: JobContext):
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
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":

    # If calling "python main.py download-files" do not initialize the OpenViduAgent
    if len(sys.argv) > 1 and sys.argv[1] == "download-files":
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
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

    worker_options = WorkerOptions(
        prewarm_fnc=prewarm,
        entrypoint_fnc=entrypoint,
        api_key=agent_config["api_key"],
        api_secret=agent_config["api_secret"],
        ws_url=agent_config["ws_url"],
        load_threshold=load_threshold,
        max_retry=sys.maxsize,
        drain_timeout=sys.maxsize,
        # For speech transcription, we want to initiate a new instance of the agent for each room
        worker_type=WorkerType.ROOM,
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
    if agent_config["live_captions"]["processing"] == "manual":
        worker_options.agent_name = agent_name

    logging.info(
        f"Starting agent {agent_name} with processing configured to {agent_config['live_captions']['processing']}"
    )

    # Redirect signal SIGQUIT as SIGTERM to allow graceful shutdown using livekit/agents mechanism
    signal.signal(
        signal.SIGQUIT, lambda signum, frame: os.kill(int(os.getpid()), signal.SIGTERM)
    )

    cli.run_app(worker_options)
