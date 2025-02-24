import os
import json
import re
import PyPDF2

# Path to training data
TRAINING_DATA_FOLDER = "training_data"
KNOWLEDGE_FILE = "knowledge.json"

# Load or initialize knowledge
def load_knowledge():
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_knowledge(knowledge):
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=4, ensure_ascii=False)

# Normalize query (remove spaces)
def normalize_query(query):
    return re.sub(r"\s+", "", query.lower())

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

# Train from text, Python, C++, and PDF files
def train_from_files():
    knowledge = load_knowledge()

    for file in os.listdir(TRAINING_DATA_FOLDER):
        file_path = os.path.join(TRAINING_DATA_FOLDER, file)
        content = ""

        if file.endswith((".txt", ".py", ".cpp")):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        elif file.endswith(".pdf"):
            content = extract_text_from_pdf(file_path)
        else:
            continue

        lines = content.split("\n")
        for line in lines:
            match = re.match(r"(.+?)\s*=\s*(.+)", line)
            if match:
                query, answer = match.groups()
                query = normalize_query(query)
                answer = answer.strip()
                if query and answer:
                    knowledge[query] = answer  # Store as key-value pair

    save_knowledge(knowledge)
    print("Training complete!")

# Detect mathematical queries
def is_math_query(query):
    return re.fullmatch(r"[0-9\+\-\*/\(\)]+", query.replace(" ", "")) is not None

# Solve math expressions safely
def solve_math_expression(query):
    try:
        result = eval(query, {"__builtins__": None}, {})  # Secure eval with no built-in functions
        return str(result)
    except:
        return "I couldn't solve that equation."

# Find the best matching response
def find_best_match(query, knowledge):
    normalized_query = normalize_query(query)
    for stored_query in knowledge:
        if stored_query in normalized_query:
            return knowledge[stored_query]
    return None

# Handle user query
def process_query(query):
    knowledge = load_knowledge()
    
    if is_math_query(query):
        return solve_math_expression(query)

    response = find_best_match(query, knowledge)
    return response if response else "I don't know the answer. Can you teach me?"
