import asyncio
import logging
import os
import yaml
import logging
import re

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

from stt_impl import get_stt_impl

logger = logging.getLogger("agent")


def load_agent_config() -> tuple[object, str]:
    agent_config: object = None
    agent_name: str = None

    # Try to load configuration from env vars
    config_body = os.environ.get("AGENT_CONFIG_BODY")
    if config_body is not None:
        try:
            agent_config = yaml.safe_load(config_body)
        except yaml.YAMLError as exc:
            logger.error("Error loading configuration from AGENT_CONFIG_BODY env var")
            logger.error(exc)
            exit(1)
        try:
            agent_name = agent_config["agent_name"]
        except Exception as exc:
            logger.error(
                'Property "agent_name" is missing. It must be defined when providing agent configuration through env var AGENT_CONFIG_BODY'
            )
            exit(1)
        if agent_config["agent_name"] is None:
            logger.error(
                'Property "agent_name" is missing. It must be defined when providing agent configuration through env var AGENT_CONFIG_BODY'
            )
            exit(1)
        agent_name = agent_config["agent_name"]
    else:
        config_file = os.environ.get("AGENT_CONFIG_FILE")
        if config_file is not None:
            if not is_agent_config_file(
                os.path.dirname(config_file), os.path.basename(config_file)
            ):
                logger.error(
                    f"Env var AGENT_CONFIG_FILE set to {config_file}, but file is not a valid agent configuration file. It must exist and be named agent-AGENT_NAME.yml"
                )
                exit(1)
        else:
            # If env vars are not defined, try to find the config file in the current working directory

            # Possible paths for the config file are any file agent-AGENT_NAME.y(a)ml
            # in the current working directory or in the location of the python script,
            # being AGENT_NAME a unique string identifying the agent.
            cwd = os.getcwd()
            for f in os.listdir(cwd):
                if is_agent_config_file(cwd, f):
                    config_file = os.path.join(cwd, f)
                    break
            if config_file is None:
                filedir = os.path.dirname(os.path.realpath(__file__))
                for f in os.listdir(filedir):
                    if is_agent_config_file(filedir, f):
                        config_file = os.path.join(filedir, f)
                        break

        if config_file is None:
            logger.error(
                "\nAgent configuration not found. One of these must be defined:\n    - env var AGENT_CONFIG_FILE with the path to the YAML configuration file.\n    - env var AGENT_CONFIG_BODY with the configuration YAML as a string.\n    - A file agent-AGENT_NAME.yml in the current working directory"
            )
            exit(1)

        with open(config_file) as stream:
            try:
                agent_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.error(f"Error loading configuration file {config_file}")
                logger.error(exc)
                exit(1)

        # Load agent name from the file name
        agent_name = os.path.basename(config_file).split(".")[0].split("agent-")[1]
        # If property "agent_name" of the config file is defined check that it matches the value inside the file name
        agent_name_in_config_file = None
        try:
            agent_name_in_config_file = agent_config["agent_name"]
        except Exception as exc:
            # Do nothing
            pass
        if (
            agent_name_in_config_file is not None
            and agent_name_in_config_file != agent_name
        ):
            logger.error(
                f"Agent name is defined as \"{agent_config['agent_name']}\" inside configuration file {config_file} and it does not match the value of the file name \"{agent_name}\""
            )
            exit(1)

    if "api_key" not in agent_config:
        if not os.environ.get("LIVEKIT_API_KEY"):
            logger.error(
                "api_key not defined in agent configuration or LIVEKIT_API_KEY env var"
            )
            exit(1)
        agent_config["api_key"] = os.environ["LIVEKIT_API_KEY"]
    if "api_secret" not in agent_config:
        if not os.environ.get("LIVEKIT_API_SECRET"):
            logger.error(
                "api_secret not defined in agent configuration or LIVEKIT_API_SECRET env var"
            )
            exit(1)
        agent_config["api_secret"] = os.environ["LIVEKIT_API_SECRET"]
    if "ws_url" not in agent_config:
        if not os.environ.get("LIVEKIT_URL"):
            logger.error(
                "ws_url not defined in agent configuration or LIVEKIT_URL env var"
            )
            exit(1)
        agent_config["ws_url"] = os.environ["LIVEKIT_URL"]

    return agent_config, agent_name


def is_agent_config_file(file_folder: str, file_name: str) -> bool:
    return (
        os.path.isfile(os.path.join(file_folder, file_name))
        and re.match(r"agent-[a-zA-Z0-9-_]+\.ya?ml", file_name) is not None
    )


async def _forward_transcription(
    stt_stream: stt.SpeechStream, stt_forwarder: transcription.STTSegmentsForwarder
):
    """Forward the transcription to the client and log the transcript in the console"""
    async for ev in stt_stream:
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            # you may not want to log interim transcripts, they are not final and may be incorrect
            logger.info(
                f"{stt_forwarder._participant_identity} is saying -> {ev.alternatives[0].text}"
            )
        elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            logger.info(
                f"{stt_forwarder._participant_identity} said -> {ev.alternatives[0].text}"
            )
        elif ev.type == stt.SpeechEventType.RECOGNITION_USAGE:
            logger.debug(f"metrics: {ev.recognition_usage}")

        stt_forwarder.update(ev)


async def entrypoint(ctx: JobContext):
    agent_config, agent_name = load_agent_config()

    print("Agent", agent_name, "joining room", ctx.room.name)

    logger.info(f"starting transcriber (speech to text) example, room: {ctx.room.name}")
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
            "Agent",
            agent_name,
            "transcribing audio track",
            track.sid,
            "from participant",
            participant.identity,
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


async def request_fnc(req: JobRequest):
    agent_config, agent_name = load_agent_config()

    print("Agent", agent_name, "received job request", req.job.id)

    # accept the job request
    await req.accept(
        # the agent's name (Participant.name), defaults to ""
        name=agent_name,
        # the agent's identity (Participant.identity), defaults to "agent-<jobid>"
        identity="agent-" + req.job.id,
        # attributes to set on the agent participant upon join
        # attributes={"myagent": "rocks"},
    )
    # or reject it
    # await req.reject()


if __name__ == "__main__":
    agent_config, agent_name = load_agent_config()

    print("Starting agent", agent_name)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc,
            api_key=agent_config["api_key"],
            api_secret=agent_config["api_secret"],
            ws_url=agent_config["ws_url"],
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
                hidden=agent_config["hidden"],
            ),
        )
    )
