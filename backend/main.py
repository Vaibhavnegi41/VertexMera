import uuid
from backend.graph import chatbot  


if __name__ == "__main__":
    # Current conversation session thread ID for PostgresSaver persistence
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"--- Running VertexMera Pipeline (Thread: {thread_id}) ---")
    results = chatbot.invoke({
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