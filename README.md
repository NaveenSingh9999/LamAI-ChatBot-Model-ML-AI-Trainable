LamAI - A Smart Terminal Chatbot

LamAI is a terminal-based chatbot designed to interact with users using natural language processing. It is trained in advanced Python and C++ coding, provides witty and GenZ-style responses, and can learn from user interactions. It also features an API for integration with external applications.

Features

Conversational AI: Engages users with a fun, GenZ, and smart-ass tone.

Message Analysis: Reads up to 40 previous messages for context.

Code Assistance: Provides advanced Python and C++ coding help.

Training Capability: Can be trained using .txt, .py, .cpp, and .pdf files.

Knowledge Expansion: Learns from ongoing chats and stores knowledge.

Custom API: Provides chatbot functionalities through an API.

Multi-threaded API: Handles multiple requests efficiently.


Installation

Prerequisites

Python 3.x

Flask (pip install flask flask-cors)


Clone the Repository

```git clone https://github.com/NaveenSingh9999/LamAI.git && cd LamAI```

Usage

1. Run the Chatbot

```python lamai.py```

2. Train the Chatbot

```python lamai_train.py```

3. Teach New Knowledge

```python lamai_learn.py```

4. Start API Server

```python lamai_api.py```

API Usage

Endpoint: /chat

Method: POST

Request Format: { "message": "Your question here" }

Response: { "response": "Bot's reply" }


Example using curl:

```curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d '{"message": "What is Python?"}' ```

File Structure

```LamAI/
│── lamai.py          # Main chatbot script
│── lamai_train.py    # Training script
│── lamai_learn.py    # Learning script
│── lamai_api.py      # API server script
│── knowledge.json    # Knowledge database
│── chat_history.json # Chat history storage```

Contributing

Feel free to fork the repository and submit pull requests with improvements.

License

This project is licensed under the GNU General Public License.

Author

Developed by NaveenSingh9999.


---

Enjoy using LamAI! 🚀

