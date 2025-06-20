import logging
import sys

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    WorkerType,
    WorkerPermissions,
    JobRequest,
)

from openviduagentutils.openvidu_agent import OpenViduAgent


async def entrypoint(ctx: JobContext) -> None:
    openvidu_agent = OpenViduAgent.get_instance()
    openvidu_agent.new_active_job(ctx)

    agent_name = openvidu_agent.get_agent_name()

    # Use livekit sdks to interact with the room

    print(f"Agent {agent_name} joining room {ctx.room.name}")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)


async def request_fnc(req: JobRequest) -> None:
    openvidu_agent = OpenViduAgent.get_instance()
    agent_name = openvidu_agent.get_agent_name()

    logging.info(f"Agent {agent_name} received job request {req.job.id}")

    if not openvidu_agent.can_accept_new_jobs():
        logging.warning(f"Agent {agent_name} cannot accept new job requests")
        await req.reject()
        return
    else:
        logging.info(f"Agent {agent_name} can accept job request {req.job.id}")
        await req.accept(
            name=agent_name,
            identity="agent-" + agent_name + "-" + req.job.id,
        )


if __name__ == "__main__":

    openvidu_agent = OpenViduAgent.get_instance(True)
    agent_config = openvidu_agent.get_agent_config()
    agent_name = openvidu_agent.get_agent_name()

    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=request_fnc,
        api_key=agent_config["api_key"],
        api_secret=agent_config["api_secret"],
        ws_url=agent_config["ws_url"],
        max_retry=sys.maxsize,
        worker_type=WorkerType.ROOM,
        permissions=WorkerPermissions(
            can_publish=False,
            can_subscribe=True,
        ),
    )
    if agent_config["live_captions"]["processing"] == "manual":
        worker_options.agent_name = agent_name

    logging.info(
        f"Starting agent {agent_name} with processing configured to {agent_config['live_captions']['processing']}"
    )

    cli.run_app(worker_options)
