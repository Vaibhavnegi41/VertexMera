import uuid
import asyncio
from backend.graph import chatbot


async def main():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"--- Running VertexMera Async Pipeline (Thread: {thread_id}) ---")

    results = await chatbot.ainvoke({
        "question": "what is linear regression?",
        "documents": [],
        "steps": [],
        "generation": "",
        "web_searched": False,
        "pii_map": {},
        "pii_redacted": False
    }, config=config)

    print("Generation:\n", results.get("generation"))
    print("\nSteps:\n", results.get("steps"))
    print("\nPII Redacted:", results.get("pii_redacted", False))


if __name__ == "__main__":
    asyncio.run(main())