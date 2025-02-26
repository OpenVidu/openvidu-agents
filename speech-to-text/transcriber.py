import asyncio
import logging
import os
import yaml
import logging
from munch import Munch

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
)
from livekit.plugins import silero

from stt_impl import get_stt_impl

logger = logging.getLogger("agent")


def load_agent_config():
    config_file = os.environ.get("AGENT_CONFIG_FILE")
    if config_file is None:
        config_body = os.environ.get("AGENT_CONFIG_BODY")
        if config_body is None:
            possible_config_paths = [
                os.getcwd() + "/agent.yaml",
                os.getcwd() + "/agent.yml",
                os.path.dirname(os.path.realpath(__file__)) + "/agent.yaml",
                os.path.dirname(os.path.realpath(__file__)) + "/agent.yml",
            ]
            for path in possible_config_paths:
                if os.path.exists(path):
                    config_file = path
                    break
        if config_file is None:
            logger.error(
                "No agent configuration found. One of these must be defined:\n    - env var AGENT_CONFIG_FILE with the path to a YAML.\n    - env var AGENT_CONFIG_BODY with the configuration YAML as a string.\n    - A file named agent.yaml (or agent.yml) in the current directory."
            )
            exit(1)

    if config_file is not None:
        with open(config_file) as stream:
            try:
                agent_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.error(f"Error loading configuration file {config_file}")
                logger.error(exc)
                exit(1)
    elif config_body is not None:
        try:
            agent_config = yaml.safe_load(config_body)
        except yaml.YAMLError as exc:
            logger.error("Error loading configuration from AGENT_CONFIG_BODY env var")
            logger.error(exc)
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
    return agent_config


agent_config = load_agent_config()


async def _forward_transcription(
    stt_stream: stt.SpeechStream, stt_forwarder: transcription.STTSegmentsForwarder
):
    """Forward the transcription to the client and log the transcript in the console"""
    async for ev in stt_stream:
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            # you may not want to log interim transcripts, they are not final and may be incorrect
            pass
        elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            logger.info(" -> ", ev.alternatives[0].text)
        elif ev.type == stt.SpeechEventType.RECOGNITION_USAGE:
            logger.debug(f"metrics: {ev.recognition_usage}")

        stt_forwarder.update(ev)


async def entrypoint(ctx: JobContext):
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


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=agent_config["api_key"],
            api_secret=agent_config["api_secret"],
            ws_url=agent_config["ws_url"],
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
                hidden=agent_config["stt"]["hidden"],
            ),
        )
    )
