import asyncio
import logging
import os
import sys
import uuid

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    stt,
    transcription,
    WorkerType,
    WorkerPermissions,
    JobRequest,
)
from livekit.plugins import silero
import psutil

from stt_impl import get_stt_impl
from openviduagentutils.config_loader import ConfigLoader
from openviduagentutils.signal_manager import SignalManager

signal_manager: SignalManager = None
config_loader: ConfigLoader = None


# Singleton pattern for SignalManager
def get_signal_manager(
    agent_config: object, agent_name: str, register_signals: bool
) -> SignalManager:
    """Ensure that SignalManager is initialized properly per process"""
    global signal_manager
    if signal_manager is None:
        agent_main_pid = os.environ["AGENT_MAIN_PID"]
        agent_process_uuid = os.environ["AGENT_UUID"]
        signal_manager = SignalManager(
            agent_config,
            agent_name,
            agent_process_uuid,
            agent_main_pid,
            register_signals,
        )
    return signal_manager


def get_top_level_parent_id() -> int:
    """Get the top-level parent process ID"""
    current_pid = os.getpid()

    try:
        process = psutil.Process(current_pid)
        parent = process.parent()

        # Keep traversing up until we reach init (PID 1) or can't go further
        while parent and parent.pid != 1:
            try:
                process = parent
                parent = process.parent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

        return process.pid

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return current_pid


# Singleton pattern for agent configuration
def get_config_loader() -> ConfigLoader:
    """Ensure that ConfigLoader is initialized properly per process"""
    global config_loader
    if config_loader is None:
        config_loader = ConfigLoader()
    return config_loader


async def _forward_transcription(
    stt_stream: stt.SpeechStream, stt_forwarder: transcription.STTSegmentsForwarder
):
    """Forward the transcription to the client and log the transcript in the console"""
    async for ev in stt_stream:
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            # you may not want to log interim transcripts, they are not final and may be incorrect
            logging.info(
                f"{stt_forwarder._participant_identity} is saying -> {ev.alternatives[0].text}"
            )
        elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            logging.info(
                f"{stt_forwarder._participant_identity} said -> {ev.alternatives[0].text}"
            )
        elif ev.type == stt.SpeechEventType.RECOGNITION_USAGE:
            logging.debug(f"metrics: {ev.recognition_usage}")

        stt_forwarder.update(ev)


async def entrypoint(ctx: JobContext) -> None:
    agent_config, agent_name = get_config_loader().load_agent_config()

    signal_manager: SignalManager = get_signal_manager(
        agent_config, agent_name, register_signals=False
    )

    ctx.add_shutdown_callback(signal_manager.decrement_active_jobs)
    signal_manager.increment_active_jobs()

    print(f"Agent {agent_name} joining room {ctx.room.name}")

    stt_impl = get_stt_impl(agent_config)

    if not stt_impl.capabilities.streaming:
        # wrap with a stream adapter to use streaming semantics
        stt_impl = stt.StreamAdapter(
            stt=stt_impl,
            vad=silero.VAD.load(
                min_silence_duration=0.2,
            ),
        )

    async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stt_forwarder = transcription.STTSegmentsForwarder(
            room=ctx.room, participant=participant, track=track
        )

        print(
            f"Agent {agent_name} transcribing audio track {track.sid} from participant {participant.identity}"
        )

        stt_stream = stt_impl.stream()
        asyncio.create_task(_forward_transcription(stt_stream, stt_forwarder))

        async for ev in audio_stream:
            stt_stream.push_frame(ev.frame)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        # spin up a task to transcribe each track
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(transcribe_track(participant, track))

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)


async def request_fnc(req: JobRequest) -> None:
    agent_config, agent_name = get_config_loader().load_agent_config()

    signal_manager = get_signal_manager(
        agent_config, agent_name, register_signals=False
    )

    logging.info(f"Agent {agent_name} received job request {req.job.id}")

    if not signal_manager.can_accept_new_jobs():
        logging.warning(f"Agent {agent_name} cannot accept new job requests")
        await req.reject()
        return
    else:
        logging.info(f"Agent {agent_name} can accept job request {req.job.id}")
        await req.accept(
            # the agent's name (Participant.name), defaults to ""
            name=agent_name,
            # the agent's identity (Participant.identity), defaults to "agent-<jobid>"
            identity="agent-" + agent_name + "-" + req.job.id,
            # attributes to set on the agent participant upon join
            # attributes={"myagent": "rocks"},
        )


if __name__ == "__main__":
    agent_config, agent_name = get_config_loader().load_agent_config()

    os.environ["AGENT_MAIN_PID"] = str(os.getpid())
    os.environ["AGENT_UUID"] = uuid.uuid4().hex

    # Create a signal manager for the main process
    get_signal_manager(agent_config, agent_name, register_signals=True)

    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=request_fnc,
        api_key=agent_config["api_key"],
        api_secret=agent_config["api_secret"],
        ws_url=agent_config["ws_url"],
        max_retry=sys.maxsize,
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
            hidden=agent_config["speech_to_text"]["hidden"],
        ),
    )
    if agent_config["automatic_dispatch"] == True:
        worker_options.agent_name = agent_name

    logging.info(
        f"Starting agent {agent_name} with automatic dispatch configured to {agent_config['automatic_dispatch']}"
    )

    cli.run_app(worker_options)
