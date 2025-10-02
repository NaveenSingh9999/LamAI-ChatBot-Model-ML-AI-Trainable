#!/usr/bin/env python3
"""
LamAI - Super Intelligent AI Assistant with Enhanced Vector Database
Completely rebuilt with quality knowledge for smart responses
"""

import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path

# HuggingFace and ML imports
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartVectorKnowledgeBase:
    """Enhanced vector database with comprehensive intelligence"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.index = None
        self.load_smart_knowledge()
    
    def load_smart_knowledge(self):
        """Load the enhanced smart vector database"""
        smart_db_file = Path("smart_vector_db.json")
        smart_index_file = Path("smart_vector_index.faiss")
        
        if smart_db_file.exists() and smart_index_file.exists():
            self.load_enhanced_knowledge(smart_db_file, smart_index_file)
        else:
            logger.warning("Smart vector database not found! Please run enhanced_trainer.py first.")
            self.create_basic_knowledge()
    
    def load_enhanced_knowledge(self, db_file, index_file):
        """Load the enhanced vector database"""
        try:
            with open(db_file, 'r') as f:
                data = json.load(f)
            
            self.documents = data.get('documents', [])
            self.metadata = data.get('metadata', [])
            embeddings_data = data.get('embeddings', [])
            
            if embeddings_data:
                self.embeddings = [np.array(emb) for emb in embeddings_data]
                self.index = faiss.read_index(str(index_file))
                logger.info(f"✅ Loaded {len(self.documents)} smart documents from enhanced vector database")
                
                # Show categories
                categories = list(set(meta.get('category', 'unknown') for meta in self.metadata))
                logger.info(f"📚 Knowledge categories: {', '.join(categories)}")
            else:
                self.create_basic_knowledge()
        except Exception as e:
            logger.error(f"Error loading smart knowledge: {e}")
            self.create_basic_knowledge()
    
    def create_basic_knowledge(self):
        """Create basic knowledge if smart database is not available"""
        basic_knowledge = [
            "I am LamAI, an intelligent AI assistant created to help you with various tasks and questions.",
            "I can help with programming, science, mathematics, technology, and general knowledge questions.",
            "I use advanced machine learning and natural language processing to provide helpful responses.",
            "I'm designed to be helpful, harmless, and honest in all my interactions.",
            "I can explain complex topics in simple terms and provide step-by-step guidance when needed."
        ]
        
        for text in basic_knowledge:
            self.add_document(text, {'source': 'basic', 'category': 'default'})
        
        self.build_index()
        logger.info("Created basic knowledge base - run enhanced_trainer.py for full intelligence")
    
    def add_document(self, text, metadata=None):
        """Add document to vector database"""
        embedding = self.model.encode(text)
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
    
    def build_index(self):
        """Build FAISS index for fast search"""
        if self.embeddings:
            embeddings_array = np.array(self.embeddings).astype('float32')
            self.index = faiss.IndexFlatIP(embeddings_array.shape[1])
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
    
    def search(self, query, k=3):
        """Search for relevant documents using vector similarity"""
        if not self.index or not self.documents:
            return []
        
        try:
            query_embedding = self.model.encode([query]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'text': self.documents[idx],
                        'score': float(score),
                        'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                        'category': self.metadata[idx].get('category', 'unknown') if idx < len(self.metadata) else 'unknown'
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

class LamAI:
    """Main LamAI class with enhanced intelligence and reasoning"""
    
    def __init__(self):
        self.name = "LamAI"
        self.vector_kb = SmartVectorKnowledgeBase()
        self.conversation_history = []
        
        # Enhanced conversation patterns
        self.greetings = [
            "Hello! I'm LamAI, your intelligent AI assistant. I'm here to help with questions, problems, and conversations. What can I do for you today?",
            "Hi there! I'm LamAI, equipped with comprehensive knowledge in science, technology, and more. What would you like to explore?",
            "Greetings! I'm LamAI, ready to assist you with thoughtful answers and helpful guidance. How can I help?"
        ]
        
        self.farewells = [
            "Goodbye! It was wonderful talking with you. Remember, I'm always here when you need assistance or just want to chat!",
            "Take care! I enjoyed our conversation. Feel free to return anytime with questions or for a friendly chat.",
            "See you later! I'm always ready to help, learn, and engage in meaningful conversations."
        ]
    
    def respond_to_query(self, user_input):
        """Generate intelligent response using enhanced vector search and reasoning"""
        user_input = user_input.strip()
        
        if not user_input:
            return "I'm here and ready to help! Please ask me anything - from science and technology to personal advice."
        
        # Handle basic greetings
        if self.is_greeting(user_input):
            return random.choice(self.greetings)
        
        # Handle farewells
        if self.is_farewell(user_input):
            return random.choice(self.farewells)
        
        # Search enhanced vector database
        relevant_docs = self.vector_kb.search(user_input, k=3)
        
        if relevant_docs and relevant_docs[0]['score'] > 0.2:
            # Use enhanced contextual response
            response = self.generate_enhanced_response(user_input, relevant_docs)
        else:
            # Generate intelligent general response
            response = self.generate_intelligent_response(user_input)
        
        # Add to conversation history
        self.conversation_history.append({
            'user': user_input,
            'assistant': response,
            'timestamp': datetime.now().isoformat(),
            'reasoning': f"Used {relevant_docs[0]['category'] if relevant_docs else 'general'} knowledge"
        })
        
        return response
    
    def generate_enhanced_response(self, query, relevant_docs):
        """Generate enhanced response based on smart vector search results"""
        best_match = relevant_docs[0]
        category = best_match['category']
        text = best_match['text']
        score = best_match['score']
        
        # Handle Q&A format responses
        if text.startswith('Q:') and 'A:' in text:
            # Extract the answer part
            answer_part = text.split('A:', 1)[1].strip()
            
            # Add conversational enhancement based on category
            if category == 'empathy':
                return answer_part + " Is there anything specific you'd like to discuss about this?"
            elif category == 'education':
                return answer_part + " Would you like me to explain any part of this in more detail?"
            elif category == 'technology':
                return answer_part + " Are you working on something related to this, or do you have follow-up questions?"
            elif category == 'guidance':
                return answer_part + " Would you like me to help you apply this to your specific situation?"
            else:
                return answer_part + " Feel free to ask if you need more information!"
        
        # Handle general knowledge responses
        return f"Based on my knowledge: {text} Would you like me to explore this topic further or answer any specific questions about it?"
    
    def generate_intelligent_response(self, query):
        """Generate intelligent response when no specific match is found"""
        query_lower = query.lower()
        
        # Check for emotional content
        if any(emotion in query_lower for emotion in ['sad', 'happy', 'angry', 'worried', 'excited', 'nervous', 'lonely']):
            return f"I understand you're sharing something personal with me. While I don't have specific guidance about '{query}' in my knowledge base right now, I want you to know that I'm here to listen and help however I can. Could you tell me more about what you're experiencing?"
        
        # Check for learning/help requests
        if any(keyword in query_lower for keyword in ['learn', 'teach', 'explain', 'help', 'understand']):
            return f"I'd love to help you learn about '{query}'! While I don't have that specific information readily available, I can help you break down the topic and explore it step by step. What aspect would you like to start with?"
        
        # Check for problem-solving
        if any(keyword in query_lower for keyword in ['problem', 'issue', 'stuck', 'difficult', 'challenge']):
            return f"It sounds like you're facing a challenge with '{query}'. I'm here to help you work through it! Could you describe the situation in more detail so I can provide the most useful guidance?"
        
        # Check for technical questions
        if any(keyword in query_lower for keyword in ['code', 'program', 'software', 'computer', 'algorithm']):
            return f"That's a great technical question about '{query}'! While I don't have specific information about this in my current knowledge base, I'd be happy to help you explore programming concepts. Could you provide more context about what you're trying to achieve?"
        
        # General intelligent response
        return f"That's an interesting question about '{query}'! I don't have specific information about this topic in my current knowledge base, but I'd love to help you explore it. Could you tell me more about what you're looking for, or would you like me to help you break down the question into parts we can tackle together?"
    
    def is_greeting(self, text):
        """Check if text is a greeting"""
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy', 'greetings']
        return any(greeting in text.lower() for greeting in greetings)
    
    def is_farewell(self, text):
        """Check if text is a farewell"""
        farewells = ['goodbye', 'bye', 'see you', 'farewell', 'take care', 'good night', 'later', 'adios']
        return any(farewell in text.lower() for farewell in farewells)
    
    def get_conversation_summary(self):
        """Get summary of recent conversation"""
        if not self.conversation_history:
            return "No conversation history yet."
        
        recent_topics = []
        for entry in self.conversation_history[-3:]:  # Last 3 exchanges
            reasoning = entry.get('reasoning', 'general knowledge')
            recent_topics.append(reasoning)
        
        return f"Recent conversation involved: {', '.join(set(recent_topics))}"
    
    def learn_from_interaction(self, user_input, feedback_positive=True):
        """Enhanced learning from user interactions"""
        interaction_data = {
            'input': user_input,
            'feedback': 'positive' if feedback_positive else 'negative',
            'timestamp': datetime.now().isoformat(),
            'conversation_context': self.get_conversation_summary()
        }
        
        # Log for future training
        logger.info(f"Learning: {interaction_data}")

def main():
    """Main function for testing enhanced LamAI"""
    print("🧠 LamAI - Super Intelligent AI Assistant")
    print("Enhanced with comprehensive knowledge and emotional intelligence")
    print("Type 'quit' to exit")
    print("-" * 60)
    
    ai = LamAI()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f"\n🧠 LamAI: {random.choice(ai.farewells)}")
                break
            
            if user_input:
                response = ai.respond_to_query(user_input)
                print(f"\n🧠 LamAI: {response}")
        
        except KeyboardInterrupt:
            print("\n\n🧠 LamAI: It was wonderful talking with you! Take care!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Error in main loop: {e}")

if __name__ == "__main__":
    main()