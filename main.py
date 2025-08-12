"""
LiveKit Intake Agent
Run with: python main.py console
buns ahh file likely shiuld combine with livekit_main.py later on but meh main system is there for testing
"""

from src.livekit_main import entrypoint_fnc, prewarm_fnc
from livekit.agents import WorkerType, WorkerOptions, cli

if __name__ == "__main__":
    print("Starting LiveKit intake agent")
    opts = WorkerOptions(
        prewarm_fnc=prewarm_fnc, 
        entrypoint_fnc=entrypoint_fnc,
        worker_type=WorkerType.ROOM, 
        agent_name="IntakeBot"
    )
    cli.run_app(opts=opts) 