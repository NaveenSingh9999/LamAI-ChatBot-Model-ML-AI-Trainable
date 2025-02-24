import json

KNOWLEDGE_FILE = "knowledge.json"

# Load existing knowledge safely
try:
    with open(KNOWLEDGE_FILE, "r") as f:
        knowledge = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    knowledge = {}

# Train with new data
while True:
    key = input("Enter topic (or 'exit' to stop): ").strip()
    if key.lower() == "exit":
        break
    response = input("Enter response: ").strip()

    if key in knowledge:
        if response not in knowledge[key]:
            knowledge[key].append(response)
    else:
        knowledge[key] = [response]

    with open(KNOWLEDGE_FILE, "w") as f:
        json.dump(knowledge, f, indent=4)

    print(f"Added: '{key}' → '{response}'")
