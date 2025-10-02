import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import re
import requests
import wikipedia
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from collections import deque
import logging
import warnings
warnings.filterwarnings('ignore')

# HuggingFace and ML libraries
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel, pipeline
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_TRANSFORMERS = True
    print("✅ HuggingFace Transformers available!")
except ImportError as e:
    print(f"⚠️ HuggingFace libraries not available: {e}")
    HAS_TRANSFORMERS = False

# Vector database
try:
    import faiss
    HAS_FAISS = True
    print("✅ FAISS vector database available!")
except ImportError:
    print("⚠️ FAISS not available, using basic similarity")
    HAS_FAISS = False

# Try to import optional advanced libraries
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except:
    print("SpaCy not available, using basic NLP")
    nlp = None
    HAS_SPACY = False

try:
    import PyPDF2
    HAS_PDF = True
except:
    print("PyPDF2 not available, PDF processing disabled")
    HAS_PDF = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
    from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    print("TensorFlow loaded successfully")
    HAS_TENSORFLOW = True
except Exception as e:
    print(f"TensorFlow not available, using enhanced rule-based system: {e}")
    HAS_TENSORFLOW = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except:
    print("TextBlob not available, using basic sentiment analysis")
    HAS_TEXTBLOB = False
    HAS_PDF = False

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    HAS_TENSORFLOW = True
except:
    print("TensorFlow not available, using enhanced rule-based system")
    HAS_TENSORFLOW = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except:
    print("TextBlob not available, basic sentiment analysis disabled")
    HAS_TEXTBLOB = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to training data folder
TRAINING_DATA_FOLDER = "training_data"
KNOWLEDGE_FILE = "knowledge.json"
ARTICLES_FILE = "articles.json"
MODEL_FILE = "question_classifier.h5"
TOKENIZER_FILE = "tokenizer.json"

# Initialize Wikipedia API with user agent
wikipedia.set_lang("en")

# Enhanced conversation context with memory
class AdvancedContext:
    def __init__(self):
        self.conversation_history = deque(maxlen=50)
        self.current_topic = None
        self.user_emotion = "neutral"
        self.personality_traits = {
            "humor": 0.8,
            "empathy": 0.9,
            "formality": 0.3,
            "creativity": 0.8
        }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    return "positive"
                elif polarity < -0.1:
                    return "negative"
                else:
                    return "neutral"
            except:
                pass
        
        # Fallback sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'happy', 'love']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'sad', 'disappointed']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def get_recent_context(self, limit=5):
        """Get recent conversation context"""
        recent = list(self.conversation_history)[-limit:]
        context_text = ""
        for entry in recent:
            if isinstance(entry, dict):
                context_text += f"User: {entry.get('user', '')}\nAI: {entry.get('ai', '')}\n"
            else:
                context_text += str(entry) + "\n"
        return context_text.strip()
    
    def add_interaction(self, user_input, ai_response):
        """Add a user-AI interaction to conversation history"""
        interaction = {
            'user': user_input,
            'ai': ai_response,
            'timestamp': datetime.now().isoformat() if 'datetime' in globals() else 'unknown'
        }
        self.conversation_history.append(interaction)
    
    def analyze_sentiment(self, text):
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'awesome', 'excellent', 'love', 'like', 'happy', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            self.user_emotion = "positive"
            return "positive"
        elif negative_count > positive_count:
            self.user_emotion = "negative"
            return "negative"
        else:
            self.user_emotion = "neutral"
            return "neutral"
    
    def get_recent_context(self, turns=3):
        """Get recent conversation context"""
        recent = list(self.conversation_history)[-turns:] if self.conversation_history else []
        context_text = ""
        for entry in recent:
            if isinstance(entry, dict):
                context_text += f"User: {entry.get('user', '')}\nAI: {entry.get('ai', '')}\n"
            else:
                context_text += str(entry) + "\n"
        return context_text.strip()

class VectorKnowledgeBase:
    """Vector-based knowledge storage and retrieval system"""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embeddings = []
        self.documents = []
        self.metadata = []
        self.index = None
        
        if HAS_TRANSFORMERS:
            try:
                self.sentence_model = SentenceTransformer(embedding_model)
                self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
                print(f"✅ Loaded embedding model: {embedding_model} (dim: {self.embedding_dim})")
            except Exception as e:
                print(f"❌ Error loading embedding model: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None
    
    def add_document(self, text, metadata=None):
        """Add a document to the knowledge base"""
        if not self.sentence_model:
            return False
        
        try:
            embedding = self.sentence_model.encode(text)
            self.embeddings.append(embedding)
            self.documents.append(text)
            self.metadata.append(metadata or {})
            return True
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def build_index(self):
        """Build FAISS index for fast similarity search"""
        if not self.embeddings or not HAS_FAISS:
            return False
        
        try:
            embeddings_array = np.array(self.embeddings).astype('float32')
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            print(f"✅ Built FAISS index with {len(self.embeddings)} documents")
            return True
        except Exception as e:
            logger.error(f"Error building index: {e}")
            return False
    
    def search(self, query, top_k=3):
        """Search for similar documents"""
        if not self.sentence_model:
            return []
        
        try:
            query_embedding = self.sentence_model.encode([query]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            if self.index and HAS_FAISS:
                scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.documents):
                        results.append({
                            'document': self.documents[idx],
                            'metadata': self.metadata[idx],
                            'score': float(score)
                        })
                return results
            else:
                return self._basic_search(query, top_k)
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
    
    def _basic_search(self, query, top_k):
        """Basic similarity search without FAISS"""
        if not self.embeddings:
            return []
        
        try:
            query_embedding = self.sentence_model.encode([query])
            similarities = []
            
            for i, doc_embedding in enumerate(self.embeddings):
                similarity = np.dot(query_embedding[0], doc_embedding) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(doc_embedding)
                )
                similarities.append((similarity, i))
            
            similarities.sort(reverse=True)
            results = []
            for score, idx in similarities[:top_k]:
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score)
                })
            return results
        except Exception as e:
            logger.error(f"Error in basic search: {e}")
            return []

# Advanced Reasoning Engine
class ReasoningEngine:
    def __init__(self):
        self.reasoning_types = {
            "mathematical": self.solve_math_reasoning,
            "logical": self.logical_reasoning,
            "causal": self.causal_reasoning,
            "comparison": self.comparison_reasoning
        }
    
    def analyze_query_type(self, query):
        query_lower = query.lower()
        if any(word in query_lower for word in ['calculate', 'solve', '+', '-', '*', '/', 'math']):
            return "mathematical"
        elif any(word in query_lower for word in ['if', 'then', 'because', 'therefore', 'logic']):
            return "logical"
        elif any(word in query_lower for word in ['why', 'cause', 'effect', 'reason', 'result']):
            return "causal"
        elif any(word in query_lower for word in ['compare', 'difference', 'better', 'versus']):
            return "comparison"
        else:
            return "general"
    
    def chain_of_thought(self, query, context=""):
        reasoning_type = self.analyze_query_type(query)
        
        steps = [
            f"🤔 Analyzing query: '{query}'",
            f"🧠 Identified reasoning type: {reasoning_type}",
            f"📋 Considering context: {context[:100]}..." if context else "📋 No additional context"
        ]
        
        if reasoning_type in self.reasoning_types:
            result = self.reasoning_types[reasoning_type](query, context)
            steps.extend(result.get('steps', []))
            conclusion = result.get('conclusion', 'Unable to reach conclusion')
        else:
            steps.append("💭 Using general reasoning approach")
            conclusion = "This requires general knowledge and understanding."
        
        return {
            'reasoning_type': reasoning_type,
            'steps': steps,
            'conclusion': conclusion
        }
    
    def solve_math_reasoning(self, query, context):
        steps = ["🔢 Breaking down mathematical problem"]
        
        # Extract mathematical expression
        math_matches = re.findall(r'\d+[\+\-\*/\d\(\)\s]*\d+', query)
        
        if math_matches:
            try:
                expression = math_matches[0].strip()
                steps.append(f"📐 Found expression: {expression}")
                
                # Safe evaluation
                result = eval(expression, {"__builtins__": {}})
                steps.append(f"✅ Calculated result: {result}")
                conclusion = f"The answer is {result}"
            except Exception as e:
                steps.append(f"❌ Calculation error: {str(e)}")
                conclusion = "Unable to solve this mathematical problem"
        else:
            conclusion = "No clear mathematical expression found"
        
        return {'steps': steps, 'conclusion': conclusion}
    
    def logical_reasoning(self, query, context):
        steps = ["🧩 Applying logical reasoning"]
        
        if "if" in query.lower() and "then" in query.lower():
            steps.append("📝 Detected conditional reasoning (if-then)")
            conclusion = "This follows a logical conditional pattern"
        elif "because" in query.lower() or "therefore" in query.lower():
            steps.append("🔗 Detected causal reasoning")
            conclusion = "This involves cause and effect relationships"
        else:
            steps.append("🤔 General logical analysis needed")
            conclusion = "Requires logical evaluation of the statement"
        
        return {'steps': steps, 'conclusion': conclusion}
    
    def causal_reasoning(self, query, context):
        steps = ["🔄 Analyzing causal relationships"]
        steps.append("🎯 Identifying potential causes and effects")
        conclusion = "Multiple factors may contribute to this outcome"
        return {'steps': steps, 'conclusion': conclusion}
    
    def comparison_reasoning(self, query, context):
        steps = ["⚖️ Looking for comparison patterns"]
        steps.append("📊 Comparing similar concepts or situations")
        conclusion = "Drawing parallels and identifying differences"
        return {'steps': steps, 'conclusion': conclusion}

# Global reasoning engine
reasoning_engine = ReasoningEngine()

# Global context manager
context_manager = AdvancedContext()

# Global vector knowledge base
vector_kb = VectorKnowledgeBase()

# Initialize knowledge base with training data
def initialize_knowledge_base():
    """Initialize the vector knowledge base with existing data"""
    global vector_kb
    
    # Load existing knowledge base if available
    if os.path.exists("vector_kb.json"):
        if vector_kb.load_kb("vector_kb.json"):
            print("✅ Loaded existing vector knowledge base")
            return
    
    print("🧠 Initializing vector knowledge base...")
    
    # Load conversation datasets if HuggingFace is available
    if HAS_TRANSFORMERS:
        try:
            # Load sample conversation data from HuggingFace
            datasets_to_load = ["daily_dialog", "empathetic_dialogues"]
            
            for dataset_name in datasets_to_load:
                try:
                    print(f"📥 Loading {dataset_name}...")
                    dataset = load_dataset(dataset_name, split="train[:100]")  # Small sample
                    
                    for sample in dataset:
                        if dataset_name == "daily_dialog":
                            dialog = sample.get('dialog', [])
                            if len(dialog) > 1:
                                conversation_text = " ".join(dialog)
                                vector_kb.add_document(conversation_text, {"source": dataset_name, "type": "conversation"})
                        elif dataset_name == "empathetic_dialogues":
                            context = sample.get('context', '')
                            response = sample.get('response', '')
                            if context and response:
                                conversation_text = f"{context} {response}"
                                vector_kb.add_document(conversation_text, {"source": dataset_name, "type": "empathetic"})
                
                except Exception as e:
                    print(f"⚠️ Error loading {dataset_name}: {e}")
                    continue
            
            # Build the search index
            if vector_kb.build_index():
                print("✅ Vector knowledge base index built successfully")
                # Save for future use
                vector_kb.save_kb("vector_kb.json")
            
        except Exception as e:
            print(f"⚠️ Error initializing knowledge base: {e}")
    
    # Add basic knowledge from existing files
    if os.path.exists(KNOWLEDGE_FILE):
        try:
            with open(KNOWLEDGE_FILE, 'r') as f:
                knowledge_data = json.load(f)
            
            for item in knowledge_data.get('knowledge', []):
                if isinstance(item, dict):
                    text = f"{item.get('question', '')} {item.get('answer', '')}"
                    vector_kb.add_document(text, {"source": "knowledge_file", "type": "qa"})
            
            print(f"✅ Added {len(knowledge_data.get('knowledge', []))} items from knowledge file")
        except Exception as e:
            print(f"⚠️ Error loading knowledge file: {e}")

# Initialize the knowledge base at startup
try:
    initialize_knowledge_base()
except Exception as e:
    print(f"⚠️ Knowledge base initialization failed: {e}")

# Enhanced tool functions
def enhanced_wikipedia_search(query):
    try:
        search_term = query.replace('wikipedia', '').replace('wiki', '').strip()
        summary = wikipedia.summary(search_term, sentences=3, auto_suggest=True)
        page = wikipedia.page(search_term, auto_suggest=True)
        
        return f"📖 Wikipedia: {page.title}\n\n{summary}\n\n🔗 Source: {page.url}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"🔍 Multiple Wikipedia results found for '{search_term}':\n{', '.join(e.options[:5])}\n\nPlease be more specific!"
    except wikipedia.exceptions.PageError:
        return f"❌ No Wikipedia page found for '{search_term}'. Try a different search term."
    except Exception as e:
        return f"⚠️ Wikipedia search error: {str(e)}"

def enhanced_web_search(query):
    try:
        search_term = re.sub(r'(web|search|google|internet|look up)', '', query, flags=re.IGNORECASE).strip()
        url = f"https://html.duckduckgo.com/html/?q={search_term}"
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; LamAI/2.0)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for link in soup.find_all('a', class_='result__a', limit=3):
                title = link.text.strip()
                href = link.get('href', '')
                if title and href:
                    results.append(f"• {title}")
            
            if results:
                return f"🌐 Web search results for '{search_term}':\n\n" + '\n'.join(results) + "\n\nThese results should help answer your question!"
            else:
                return f"🔍 No web results found for '{search_term}'. Try rephrasing your search."
        else:
            return "⚠️ Web search is temporarily unavailable. Try again later."
    except Exception as e:
        return f"❌ Web search error: {str(e)}"
    
    def add_interaction(self, user_input, ai_response):
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "ai": ai_response,
            "topic": self.current_topic
        }
        self.conversation_history.append(interaction)
    
    def get_recent_context(self, turns=3):
        recent = list(self.conversation_history)[-turns:]
        context_parts = []
        for interaction in recent:
            context_parts.append(f"User: {interaction['user']}")
            context_parts.append(f"AI: {interaction['ai']}")
        return "\n".join(context_parts)
    
    def analyze_sentiment(self, text):
        if HAS_TEXTBLOB:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.1:
                self.user_emotion = "happy"
            elif polarity < -0.1:
                self.user_emotion = "sad"
            else:
                self.user_emotion = "neutral"
        else:
            # Basic sentiment analysis
            positive_words = ["good", "great", "awesome", "happy", "love", "excellent"]
            negative_words = ["bad", "terrible", "hate", "sad", "awful", "horrible"]
            
            text_lower = text.lower()
            if any(word in text_lower for word in positive_words):
                self.user_emotion = "happy"
            elif any(word in text_lower for word in negative_words):
                self.user_emotion = "sad"
            else:
                self.user_emotion = "neutral"

# Global enhanced context
context = AdvancedContext()

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

# Advanced Reasoning Engine
class ReasoningEngine:
    def __init__(self):
        self.reasoning_types = {
            "mathematical": self.solve_math_step_by_step,
            "logical": self.logical_reasoning,
            "causal": self.causal_reasoning,
            "comparison": self.comparison_reasoning
        }
    
    def analyze_query_type(self, query):
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['calculate', 'solve', '+', '-', '*', '/', 'math', 'equals']):
            return "mathematical"
        elif any(word in query_lower for word in ['if', 'then', 'because', 'therefore', 'logic']):
            return "logical"
        elif any(word in query_lower for word in ['why', 'cause', 'effect', 'reason', 'result']):
            return "causal"
        elif any(word in query_lower for word in ['compare', 'difference', 'better', 'worse', 'versus']):
            return "comparison"
        else:
            return "general"
    
    def chain_of_thought(self, query, context_info=""):
        reasoning_type = self.analyze_query_type(query)
        
        steps = [
            f"🤔 Analyzing: '{query}'",
            f"🧠 Reasoning type: {reasoning_type}",
            f"📋 Context: {context_info[:50]}..." if context_info else "📋 No additional context"
        ]
        
        if reasoning_type in self.reasoning_types:
            result = self.reasoning_types[reasoning_type](query, context_info)
            steps.extend(result.get('steps', []))
            conclusion = result.get('conclusion', 'Let me think more about this...')
        else:
            steps.append("💭 Using general reasoning")
            conclusion = "This requires careful consideration of multiple factors."
        
        return {
            'reasoning_type': reasoning_type,
            'steps': steps,
            'conclusion': conclusion
        }
    
    def solve_math_step_by_step(self, query, context):
        steps = ["🔢 Breaking down mathematical problem"]
        
        # Extract mathematical expression
        math_matches = re.findall(r'\d+[\+\-\*/\d\(\)\s]*\d+', query)
        
        if math_matches:
            try:
                expression = math_matches[0].strip()
                steps.append(f"📐 Expression found: {expression}")
                
                # Safe evaluation
                result = eval(expression, {"__builtins__": {}})
                steps.append(f"✅ Result: {result}")
                conclusion = f"The answer is {result}"
            except Exception as e:
                steps.append(f"❌ Calculation error: {str(e)}")
                conclusion = "I had trouble with that calculation. Could you rephrase it?"
        else:
            conclusion = "I don't see a clear mathematical expression to solve."
        
        return {'steps': steps, 'conclusion': conclusion}
    
    def logical_reasoning(self, query, context):
        steps = ["🧩 Applying logical reasoning"]
        
        if "if" in query.lower() and "then" in query.lower():
            steps.append("📝 Conditional logic detected (if-then)")
            conclusion = "Based on the condition, the logical outcome follows."
        elif "because" in query.lower():
            steps.append("🔗 Causal logic detected")
            conclusion = "The conclusion follows from the given reason."
        else:
            steps.append("🤔 General logical analysis")
            conclusion = "Let me work through this logically."
        
        return {'steps': steps, 'conclusion': conclusion}
    
    def causal_reasoning(self, query, context):
        steps = ["🔄 Analyzing cause and effect"]
        steps.append("🎯 Looking for relationships")
        conclusion = "Multiple factors usually contribute to this outcome."
        return {'steps': steps, 'conclusion': conclusion}
    
    def comparison_reasoning(self, query, context):
        steps = ["⚖️ Making comparison"]
        steps.append("📊 Weighing different aspects")
        conclusion = "Each option has its own advantages and considerations."
        return {'steps': steps, 'conclusion': conclusion}

# Global reasoning engine
reasoning_engine = ReasoningEngine()

# Enhanced web search with better error handling
def enhanced_web_search(query):
    try:
        search_query = query.replace("search", "").replace("web", "").strip()
        url = f"https://html.duckduckgo.com/html/?q={search_query}"
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; LamAI/2.0)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for link in soup.find_all('a', class_='result__a', limit=3):
                title = link.text.strip()
                if title:
                    results.append(title)
            
            if results:
                return f"🔍 Web search results for '{search_query}':\n" + "\n".join(f"• {result}" for result in results)
            else:
                return "🔍 No specific results found, but you can try searching online for more information."
        else:
            return "🔍 Web search temporarily unavailable. Try asking me directly!"
    except Exception as e:
        return f"🔍 Search had an issue, but I can try to help based on my knowledge!"

# Enhanced Wikipedia search
def enhanced_wikipedia_search(query):
    try:
        search_term = query.replace("wikipedia", "").replace("wiki", "").strip()
        if not search_term:
            return "Please specify what you'd like me to search for on Wikipedia."
        
        summary = wikipedia.summary(search_term, sentences=3, auto_suggest=True)
        page = wikipedia.page(search_term, auto_suggest=True)
        
        return f"📚 Wikipedia: {page.title}\n\n{summary}\n\n🔗 Full article: {page.url}"
        
    except wikipedia.exceptions.DisambiguationError as e:
        options = e.options[:5]
        return f"📚 Multiple Wikipedia articles found for '{search_term}'. Did you mean: {', '.join(options)}?"
    except wikipedia.exceptions.PageError:
        return f"📚 No Wikipedia article found for '{search_term}'. Try a different search term."
    except Exception as e:
        return f"📚 Wikipedia search had an issue, but I can try to help with general knowledge!"

# Extract text from PDF files
# Extract text from PDF files
def extract_text_from_pdf(pdf_path):
    if not HAS_PDF:
        print(f"PDF processing not available for {pdf_path}")
        return ""
    
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
    if not HAS_TENSORFLOW:
        print("TensorFlow not available, skipping model loading")
        return None, None
    
    try:
        model = load_model(MODEL_FILE)
        with open(TOKENIZER_FILE, "r", encoding="utf-8") as f:
            tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading TensorFlow model: {e}")
        return None, None

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

# Load the trained model and tokenizer once (optional - only if TensorFlow is available)
# The enhanced system works without these models using TF-IDF and reasoning
# if HAS_TENSORFLOW and os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
#     try:
#         model, tokenizer = load_model_and_tokenizer()
#         print("TensorFlow models loaded successfully")
#     except Exception as e:
#         print(f"TensorFlow model loading failed: {e}")
#         model, tokenizer = None, None
# else:
#     print("Using enhanced rule-based system (TensorFlow models not available)")
#     model, tokenizer = None, None

# Enhanced Main chatbot logic with advanced reasoning
def respond_to_query(query):
    global context, vector_kb
    
    logger.info(f"Processing query: {query}")
    
    # Analyze user sentiment and update context
    context.analyze_sentiment(query)
    recent_context = context.get_recent_context(2)
    
    knowledge = load_knowledge()
    articles = load_articles()
    
    # Vector knowledge base search for context
    vector_results = []
    if vector_kb and vector_kb.sentence_model:
        try:
            vector_results = vector_kb.search(query, top_k=3)
            if vector_results:
                print(f"🔍 Found {len(vector_results)} relevant documents from vector KB")
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
    
    # Enhanced mathematical handling with reasoning
    if is_math_query(query) or any(word in query.lower() for word in ['calculate', 'solve', 'math']):
        reasoning_result = reasoning_engine.chain_of_thought(query, recent_context)
        if reasoning_result['reasoning_type'] == 'mathematical':
            response = f"Let me work through this step by step:\n\n"
            for step in reasoning_result['steps'][:3]:
                response += f"{step}\n"
            response += f"\n{reasoning_result['conclusion']}"
        else:
            response = solve_math_expression(query)
        
        context.add_interaction(query, response)
        return adapt_response_personality(response, context.user_emotion)
    
    # Enhanced tool usage detection
    if should_use_enhanced_tools(query):
        return handle_enhanced_tool_request(query)
    
    # Complex reasoning for Why/How/Explain questions with vector context
    if needs_reasoning(query):
        # Add vector search context to reasoning
        vector_context = ""
        if vector_results:
            vector_context = " ".join([result['document'][:200] for result in vector_results[:2]])
        
        combined_context = f"{recent_context}\n{vector_context}".strip()
        reasoning_result = reasoning_engine.chain_of_thought(query, combined_context)
        
        response = f"🤔 Let me think about this:\n\n"
        for step in reasoning_result['steps'][:4]:
            response += f"{step}\n"
        response += f"\n💡 {reasoning_result['conclusion']}"
        
        # Add relevant context from vector search if available
        if vector_results and vector_results[0]['score'] > 0.5:
            response += f"\n\n📚 Relevant context: {vector_results[0]['document'][:300]}..."
        
        context.add_interaction(query, response)
        return adapt_response_personality(response, context.user_emotion)
    
    # Try TensorFlow model if available
    if HAS_TENSORFLOW and os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
        try:
            # Load models
            model = load_model(MODEL_FILE)
            with open(TOKENIZER_FILE, "r", encoding="utf-8") as f:
                tokenizer_json = f.read()
            tokenizer = tokenizer_from_json(tokenizer_json)
            
            # Tokenize and predict
            sequence = tokenizer.texts_to_sequences([query])
            padded_sequence = pad_sequences(sequence, maxlen=100)
            prediction = model.predict(padded_sequence)
            question_type = np.argmax(prediction)
            
            # Handle based on question type
            if question_type == 0:
                response = handle_which_question(query, knowledge)
            elif question_type == 1:
                response = handle_who_question(query, knowledge)
            elif question_type == 2:
                response = handle_what_question(query, knowledge)
            elif question_type == 3:
                response = handle_why_question(query, knowledge)
            elif question_type == 4:
                response = handle_when_question(query, knowledge)
            else:
                response = None
                
            if response:
                context.add_interaction(query, response)
                return adapt_response_personality(response, context.user_emotion)
        except Exception as e:
            logger.warning(f"TensorFlow model failed: {e}, falling back to enhanced rules")
    
    # Enhanced knowledge search using TF-IDF + Vector search
    response = enhanced_knowledge_search(query, knowledge, articles, vector_results)
    if response:
        context.add_interaction(query, response)
        return adapt_response_personality(response, context.user_emotion)
    
    # Generate enhanced fallback response
    response = generate_enhanced_fallback_response(query, recent_context)
    context.add_interaction(query, response)
    return adapt_response_personality(response, context.user_emotion)

# Enhanced helper functions
def should_use_enhanced_tools(query):
    tool_indicators = [
        'search', 'wikipedia', 'wiki', 'web', 'look up', 'find information',
        'google', 'internet', 'research'
    ]
    return any(indicator in query.lower() for indicator in tool_indicators)

def handle_enhanced_tool_request(query):
    query_lower = query.lower()
    
    if 'wikipedia' in query_lower or 'wiki' in query_lower:
        return enhanced_wikipedia_search(query)
    elif any(term in query_lower for term in ['web', 'search', 'google', 'internet', 'look up']):
        return enhanced_web_search(query)
    else:
        return "I can help you search Wikipedia or the web! Just ask me to search for something."

def needs_reasoning(query):
    reasoning_indicators = [
        'why', 'how', 'explain', 'because', 'reason', 'cause', 'effect',
        'compare', 'difference', 'better', 'analyze', 'think', 'opinion'
    ]
    return any(indicator in query.lower() for indicator in reasoning_indicators)

def enhanced_knowledge_search(query, knowledge, articles, vector_results=None):
    # First check vector search results if available
    if vector_results:
        best_vector = vector_results[0] if vector_results else None
        if best_vector and best_vector['score'] > 0.7:
            return f"Based on my knowledge: {best_vector['document']}"
    
    # Then try exact matches
    normalized_query = normalize_query(query)
    if normalized_query in knowledge:
        response = knowledge[normalized_query]
        # Enhance with vector context if available
        if vector_results and vector_results[0]['score'] > 0.5:
            response += f"\n\nAdditional context: {vector_results[0]['document'][:200]}..."
        return response
    
    # Then try TF-IDF similarity search
    all_texts = list(knowledge.keys()) + list(articles.keys())
    all_responses = list(knowledge.values()) + [articles[k].get('content', '') for k in articles.keys()]
    
    if not all_texts:
        return None
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        vectors = vectorizer.fit_transform(all_texts)
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, vectors).flatten()
        
        best_match_index = similarities.argmax()
        if similarities[best_match_index] > 0.3:  # Lower threshold for better matches
            return all_responses[best_match_index]
    except Exception as e:
        logger.error(f"TF-IDF search failed: {e}")
    
    return None

def adapt_response_personality(response, user_emotion):
    # Add personality based on user emotion and AI traits
    if context.personality_traits['empathy'] > 0.8 and user_emotion == 'sad':
        empathy_prefixes = [
            "I understand that might be concerning. ",
            "That's a thoughtful question. ",
            "I can see why you'd want to know about that. "
        ]
        if np.random.random() < 0.4:
            response = np.random.choice(empathy_prefixes) + response
    
    if context.personality_traits['humor'] > 0.7 and user_emotion in ['happy', 'neutral']:
        if len(response.split()) > 10 and np.random.random() < 0.3:
            humor_additions = [" 😊", " Hope that helps! 🚀", " Pretty interesting, right?"]
            response += np.random.choice(humor_additions)
    
    return response

def generate_enhanced_fallback_response(query, recent_context):
    query_lower = query.lower()
    
    # Greeting responses
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
    if any(greeting in query_lower for greeting in greetings):
        return "Hello! I'm LamAI with enhanced reasoning, web search, and learning capabilities! How can I help you today? 🚀"
    
    # Farewell responses
    farewells = ['bye', 'goodbye', 'see you', 'farewell']
    if any(farewell in query_lower for farewell in farewells):
        return "Goodbye! It was great chatting with you. Feel free to ask me anything anytime! 👋"
    
    # Identity questions
    if any(phrase in query_lower for phrase in ['your name', 'who are you', 'what are you']):
        return ("I'm LamAI - an enhanced AI assistant! I can reason through complex problems, "
                "search Wikipedia and the web, solve math problems, and learn from our conversations. "
                "What would you like to explore together?")
    
    # Capability questions
    if any(phrase in query_lower for phrase in ['what can you do', 'capabilities', 'help']):
        return ("I have many enhanced capabilities! I can:\n"
                "🧠 Reason through complex problems step-by-step\n"
                "🔍 Search Wikipedia and the web for information\n"
                "🧮 Solve mathematical problems with explanations\n"
                "💭 Remember our conversation context\n"
                "😊 Adapt my personality to your preferences\n"
                "📚 Learn from our interactions\n\n"
                "Just ask me anything - math, science, research, or general questions!")
    
    # Jokes
    if 'joke' in query_lower:
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything! 😄",
            "I told my computer a joke about UDP... but I'm not sure if it got it! 💻",
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
            "What's the best thing about Switzerland? I don't know, but the flag is a big plus! ➕"
        ]
        return np.random.choice(jokes)
    
    # Context-aware responses
    if recent_context and len(recent_context) > 10:
        return ("That's an interesting follow-up to our conversation! I might need a bit more "
                "context to give you the best answer. Could you elaborate or ask me to search "
                "for specific information?")
    
    # Enhanced default responses
    suggestions = [
        ("I'm not certain about that specific topic, but I'd love to help! You could try:\n"
         "• Asking me to search Wikipedia for information\n"
         "• Having me solve a math problem\n"
         "• Asking me to explain or compare concepts\n"
         "• Requesting web search results"),
        
        ("That's a fascinating question! While I'm processing that, I can also:\n"
         "• Research topics on Wikipedia\n"
         "• Break down complex problems step-by-step\n"
         "• Search the web for current information\n"
         "• Help with mathematical calculations"),
        
        ("I want to give you the best answer possible! Try asking me to:\n"
         "• 'Search Wikipedia for [topic]'\n"
         "• 'Explain why [something happens]'\n"
         "• 'Calculate [math problem]'\n"
         "• 'Compare [A] and [B]'")
    ]
    
    return np.random.choice(suggestions)

if __name__ == "__main__":
    print("🚀 Enhanced LamAI Starting...")
    print("✅ Loading knowledge base...")
    
    # Train from files if available
    try:
        train_from_files()
        print("📚 Training data processed successfully!")
    except Exception as e:
        print(f"⚠️ Training skipped: {e}")
    
    print("🤖 Enhanced LamAI Ready!")
    print("💡 I now have advanced reasoning, web search, and learning capabilities!")
    print("🎯 Try: 'solve 15*24', 'search wikipedia for AI', or 'explain why we sleep'")
    print("Type 'exit' to end the chat.\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
            if user_input.lower() == "exit":
                print("👋 Goodbye! Thanks for chatting with Enhanced LamAI!")
                break
            elif not user_input:
                continue
                
            response = respond_to_query(user_input)
            print(f"🤖 LamAI: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Error in main loop: {e}")