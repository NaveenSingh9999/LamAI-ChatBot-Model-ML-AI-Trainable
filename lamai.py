import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import re
import requests
import spacy
import PyPDF2
import wikipedia
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load SpaCy model
nlp = spacy.load("en_core_web_lg")  # Use "md" instead of "sm"

# Path to training data folder
TRAINING_DATA_FOLDER = "training_data"
KNOWLEDGE_FILE = "knowledge.json"
ARTICLES_FILE = "articles.json"
MODEL_FILE = "question_classifier.h5"
TOKENIZER_FILE = "tokenizer.json"

# Initialize Wikipedia API with user agent
wikipedia.set_lang("en")

# Global variable to store the context of the conversation
context = {
    "last_question": None,
    "last_answer": None
}

# Load or initialize knowledge
def load_knowledge():
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Load or initialize articles
def load_articles():
    if os.path.exists(ARTICLES_FILE):
        with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_knowledge(knowledge):
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=4, ensure_ascii=False)

def save_articles(articles):
    with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)

# Normalize query (preserve spaces)
def normalize_query(query):
    return re.sub(r"\s+", " ", query.lower().strip())

# Extract text from PDF files
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

# Extract text from web pages
def extract_text_from_web(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting text from {url}: {e}")
        return ""

# Train from text, Python, C++ and PDF files
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
        elif file == "train-v2.0.json":
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                for item in json_data["data"]:
                    for paragraph in item["paragraphs"]:
                        for qa in paragraph["qas"]:
                            question = qa["question"].strip()
                            if qa["answers"]:
                                answer = qa["answers"][0]["text"].strip()
                                knowledge[normalize_query(question)] = answer
            continue
        else:
            continue

        # Split content into lines and process each line
        lines = content.split("\n")
        for i in range(0, len(lines) - 1, 2):
            user_message = lines[i].replace("User: ", "").strip()
            bot_response = lines[i + 1].replace("Bot: ", "").strip()
            if user_message and bot_response:
                knowledge[normalize_query(user_message)] = bot_response

    save_knowledge(knowledge)
    print("Training complete!")

# Train from web pages
def train_from_web(url):
    articles = load_articles()
    text = extract_text_from_web(url)
    if text:
        title = url  # Use the URL as the title for simplicity
        articles[title] = {
            "title": title,
            "content": text
        }
    save_articles(articles)
    print("Training from web complete!")

# Detect mathematical expressions
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
    queries = list(knowledge.keys())
    responses = list(knowledge.values())

    if not queries:
        return None

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(queries)
    query_vec = vectorizer.transform([normalized_query])
    similarities = cosine_similarity(query_vec, vectors).flatten()
    best_match_index = similarities.argmax()

    if similarities[best_match_index] > 0.5:  # Threshold for similarity
        return responses[best_match_index]
    return None

def calculate_similarity(query1, query2):
    try:
        # Tokenize and remove stopwords using SpaCy
        doc1 = nlp(query1)
        doc2 = nlp(query2)
        tokens1 = set([token.text.lower() for token in doc1 if not token.is_stop and not token.is_punct])
        tokens2 = set([token.text.lower() for token in doc2 if not token.is_stop and not token.is_punct])

        # Calculate Jaccard similarity
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        similarity = len(intersection) / len(union) if union else 0
        # print(f"Tokens1: {tokens1}, Tokens2: {tokens2}, Intersection: {intersection}, Union: {union}, Similarity: {similarity}")
        return similarity
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0

# Handle different types of questions
def handle_which_question(query, knowledge):
    return find_best_match(query, knowledge)

def handle_who_question(query, knowledge):
    return find_best_match(query, knowledge)

def handle_what_question(query, knowledge):
    return find_best_match(query, knowledge)

def handle_why_question(query, knowledge):
    return find_best_match(query, knowledge)

def handle_when_question(query, knowledge):
    return find_best_match(query, knowledge)

# Wikipedia search
def search_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return "I couldn't find any information on Wikipedia."
    except Exception as e:
        return f"Error retrieving Wikipedia data: {e}"

# Web search (using DuckDuckGo)
def search_web(query):
    try:
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for link in soup.find_all('a', class_='result__a', limit=3):
                results.append(f"{link.text}\n{link['href']}")
            return '\n'.join(results) if results else "I couldn't find any information on the web."
        return "I couldn't find any information on the web."
    except Exception as e:
        return f"Error searching the web: {e}"

# Generate a general conversation response
def generate_general_response(query):
    doc = nlp(query)
    if doc:
        if any(token.lemma_ == "hello" for token in doc):
            return "Hi there! How can I help you today?"
        elif any(token.lemma_ == "bye" for token in doc):
            return "Goodbye! Have a great day!"
        elif any(token.lemma_ == "how" and token.nbor().lemma_ == "be" for token in doc):
            return "I'm just a bot, but I'm doing great! How about you?"
        elif any(token.lemma_ == "name" for token in doc):
            return "I'm LamAI, your friendly chatbot!"
        elif any(token.lemma_ == "joke" for token in doc):
            return "Why don't scientists trust atoms? Because they make up everything!"
    return "I'm not sure how to respond to that. Can you teach me?"

# Neural network model for question classification
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))  # 6 classes: which, who, what, why, when, other
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the neural network model
def train_model():
    knowledge = load_knowledge()
    queries = list(knowledge.keys())
    responses = list(knowledge.values())

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(queries)
    sequences = tokenizer.texts_to_sequences(queries)
    X = pad_sequences(sequences, maxlen=100)
    y = np.array([classify_question(q) for q in queries])

    model = create_model(X.shape[1])
    
    # Adjust validation_split based on the number of samples
    validation_split = 0.2 if len(X) >= 5 else 0
    
    model.fit(X, y, epochs=10, batch_size=32, validation_split=validation_split)
    model.save(MODEL_FILE)
    
    # Save the tokenizer
    tokenizer_json = tokenizer.to_json()
    with open(TOKENIZER_FILE, "w", encoding="utf-8") as f:
        f.write(tokenizer_json)
    
    return model, tokenizer

# Load the trained model and tokenizer
def load_model_and_tokenizer():
    model = load_model(MODEL_FILE)
    with open(TOKENIZER_FILE, "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    return model, tokenizer

# Classify question type
def classify_question(query):
    if query.lower().startswith("which"):
        return [1, 0, 0, 0, 0, 0]
    elif query.lower().startswith("who"):
        return [0, 1, 0, 0, 0, 0]
    elif query.lower().startswith("what"):
        return [0, 0, 1, 0, 0, 0]
    elif query.lower().startswith("why"):
        return [0, 0, 0, 1, 0, 0]
    elif query.lower().startswith("when"):
        return [0, 0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 0, 1]

# Load the trained model and tokenizer once
if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
    model, tokenizer = load_model_and_tokenizer()
else:
    model, tokenizer = train_model()

# Main chatbot logic
def respond_to_query(query):
    global context
    knowledge = load_knowledge()
    articles = load_articles()

    # Handle mathematical expressions dynamically
    if is_math_query(query):
        return solve_math_expression(query)

    # Tokenize and pad the query
    sequence = tokenizer.texts_to_sequences([query])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    # Predict the question type
    prediction = model.predict(padded_sequence)
    question_type = np.argmax(prediction)

    # Determine the type of question and handle accordingly
    if question_type == 0:
        return handle_which_question(query, knowledge)
    elif question_type == 1:
        return handle_who_question(query, knowledge)
    elif question_type == 2:
        return handle_what_question(query, knowledge)
    elif question_type == 3:
        return handle_why_question(query, knowledge)
    elif question_type == 4:
        return handle_when_question(query, knowledge)
    elif "wikipedia" in query.lower():
        return search_wikipedia(query.replace("wikipedia", "").strip())
    elif "web" in query.lower() or "search" in query.lower():
        return search_web(query.replace("web", "").replace("search", "").strip())
    else:
        # Check if the query is a follow-up question
        if context["last_answer"] and query.lower() in " ".join(context["last_answer"]).lower():
            return context["last_answer"]

        # Find best match in stored knowledge
        response = find_best_match(query, knowledge)
        if response:
            # Update context with the last question and answer
            context["last_question"] = query
            context["last_answer"] = response
            return response  # Return stored response

        # Find best match in stored articles
        response = find_best_match(query, articles)
        if response:
            # Update context with the last question and answer
            context["last_question"] = query
            context["last_answer"] = response
            return response  # Return stored response

        # Generate a general conversation response
        general_response = generate_general_response(query)
        if general_response:
            return general_response

        print("I don't know the answer. Can you teach me?")
        new_answer = input("You (provide answer): ").strip()
        if new_answer:
            knowledge[normalize_query(query)] = new_answer
            save_knowledge(knowledge)
            # Update context with the last question and answer
            context["last_question"] = query
            context["last_answer"] = new_answer
            return "Got it! I'll remember that for next time."
        return "Okay, no problem."

if __name__ == "__main__":
    train_from_files()
    print("LamAI: Hello! Type 'exit' to end the chat.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        response = respond_to_query(user_input)
        print(f"LamAI: {response}")