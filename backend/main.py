from backend.graph import chatbot  


if __name__ == "__main__":
    results = chatbot.invoke({
        "question": "what is linear regression?",
        "documents": [],
        "steps": [],
        "generation": "",
        "web_searched": False
    })

    print(results["generation"])
    print(results["steps"])