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
    WorkerType,
    WorkerPermissions,
    JobProcess,
    Agent,
    AgentSession,
    RoomOutputOptions,
    RoomInputOptions,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from stt_impl import get_stt_impl
from openviduagentutils.openvidu_agent import OpenViduAgent


class Transcriber(Agent):
    def __init__(self, stt: stt.STT):
        super().__init__(
            instructions="not-needed",
            stt=stt,
        )


async def entrypoint(ctx: JobContext):
    openvidu_agent = OpenViduAgent.get_instance()

    agent_config = openvidu_agent.get_agent_config()
    agent_name = openvidu_agent.get_agent_name()

    try:
        stt_impl = get_stt_impl(agent_config)
    except ValueError as e:
        logging.error(f"Failed to initialize provider: {e}")
        sys.exit(3)

    print(f"Agent {agent_name} joining room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        # vad is needed for non-streaming STT implementations
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=Transcriber(stt_impl),
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            # The agent will only generate text transcriptions as output
            transcription_enabled=True,
            audio_enabled=False,
            sync_transcription=False,
        ),
        room_input_options=RoomInputOptions(
            # The agent will only receive audio tracks as input
            text_enabled=False,
            video_enabled=False,
            audio_enabled=True,
            pre_connect_audio=True,
            pre_connect_audio_timeout=3.0,
        ),
    )


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

    worker_options = WorkerOptions(
        prewarm_fnc=prewarm,
        entrypoint_fnc=entrypoint,
        api_key=agent_config["api_key"],
        api_secret=agent_config["api_secret"],
        ws_url=agent_config["ws_url"],
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
