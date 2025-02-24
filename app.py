from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

app = Flask(__name__)

# Load FAISS DB
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("vector_db", embeddings)

# Load LLM (GPT-3.5 or Llama)
llm = OpenAI(temperature=0.5)

@app.route("/api", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    docs = db.similarity_search(user_input, k=2)
    context = "\n".join([doc.page_content for doc in docs])

    # Generate response
    prompt = f"Based on this context:\n{context}\nAnswer: {user_input}"
    response = llm.predict(prompt)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
