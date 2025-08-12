# import sys
# import argparse

# def run_console_mode():
#     """Run LiveKit console mode."""
#     # Import and run LiveKit console mode
#     from src.livekit_main import entrypoint_fnc, prewarm_fnc
#     from livekit.agents import WorkerType, WorkerOptions, cli
    
#     print("Starting LiveKit agent in console mode")
#     opts = WorkerOptions(
#         prewarm_fnc=prewarm_fnc, 
#         entrypoint_fnc=entrypoint_fnc,
#         worker_type=WorkerType.ROOM, 
#         agent_name="botboy"
#     )
#     cli.run_app(opts=opts)

# # Check if console mode is requested
# # i dont like ts ima come back to it later
# if len(sys.argv) > 1 and sys.argv[1] == "console":
#     run_console_mode()
# else:
#     # Original FastAPI webhook mode
#     from fastapi import FastAPI, Request
#     from contextlib import asynccontextmanager
#     import uvicorn
#     import asyncio
#     from src.bot_telnyx import run_bot
#     from threading import Lock

#     # Thread-safe set for ongoing call IDs
#     call_queue = set()
#     call_queue_lock = Lock()

#     @asynccontextmanager
#     async def lifespan(app: FastAPI):
#         print("Application startup complete")
#         yield
#         print("Application shutdown complete")

#     app = FastAPI(lifespan=lifespan)

#     @app.get("/")
#     async def health_check():
#         return {"status": "ok"}

#     @app.get("/queue_status")
#     async def queue_status():
#         with call_queue_lock:
#             return {"ongoing_calls": list(call_queue), "count": len(call_queue)}

#     @app.post("/")
#     async def agent_request(request: Request):
#         body = await request.json()
#         print(body)
#         call_control_id = body.get("call_control_id")
#         if not call_control_id:
#             return {"error": "Missing call_control_id"}

#         # Add call to queue at the very start
#         print("ADDING TO QUEUE")
#         with call_queue_lock:
#             call_queue.add(call_control_id)

#         # Create an event to signal when bot is ready
#         bot_ready_event = asyncio.Event()
#         prompt = body["prompt"]
#         #sexy add here last name
#         last_name = "zoccoli"
#         # last_name = "smith"
#         prompt = prompt + "your last name is" + last_name
#         caller_number = body["caller_number"]
#         # Start bot in background task
#         bot_task = asyncio.create_task(
#             run_bot(body["room_url"], body["token"], body["sip_uri"], body["call_control_id"], body["firm_id"], caller_number, bot_ready_event, prompt)
#         )
#         # Wait for bot to be ready (with timeout)
#         try:
#             await asyncio.wait_for(bot_ready_event.wait(), timeout=30.0)
#             return {"message": "Bot started and ready"}
#         except asyncio.TimeoutError:
#             bot_task.cancel()
#             # Remove from queue on failure
#             with call_queue_lock:
#                 call_queue.discard(call_control_id)
#             return {"error": "Bot startup timeout"}

#     # Helper to remove a call from the queue (to be called after stallion post completes)
#     def remove_call_from_queue(call_control_id):
#         with call_queue_lock:
#             call_queue.discard(call_control_id)

#     # Run webhook server
#     if __name__ == "__main__":
#         print("Starting webhook server...")
#         uvicorn.run("main:app", host="0.0.0.0", port=4000, reload=False) 