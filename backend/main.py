from backend.graph import chatbot  


if __name__ == "__main__":
    results = chatbot.invoke({
        "question": "what is linear regression?",
        "documents": [],
        "steps": [],
        "generation": "",
        "web_searched": False,
        "pii_map": {},
        "pii_redacted": False
    })

    print("Generation:\n", results["generation"])
    print("\nSteps:\n", results["steps"])
    print("\nPII Redacted:", results.get("pii_redacted", False))