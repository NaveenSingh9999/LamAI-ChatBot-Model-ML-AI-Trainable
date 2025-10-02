#!/usr/bin/env python3
"""
LamAI Smart Training Script
Trains vector database with high-quality HuggingFace datasets
"""

import os
import json
import logging
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartTrainer:
    def __init__(self):
        print("🚀 Initializing Smart Trainer for LamAI...")
        
        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Storage
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.index = None
        
        print(f"✅ Embedding model loaded (dim: {self.embedding_dim})")
    
    def train_from_datasets(self):
        """Train with high-quality conversation datasets"""
        
        datasets_config = [
            {
                'name': 'daily_dialog',
                'config': None,
                'split': 'train[:2000]',
                'extractor': self.extract_daily_dialog,
                'description': 'Daily conversations'
            },
            {
                'name': 'empathetic_dialogues', 
                'config': None,
                'split': 'train[:1500]',
                'extractor': self.extract_empathetic,
                'description': 'Empathetic responses'
            },
            {
                'name': 'persona_chat',
                'config': None,
                'split': 'train[:1000]',
                'extractor': self.extract_persona_chat,
                'description': 'Personality-based dialogues'
            },
            {
                'name': 'blended_skill_talk',
                'config': None,
                'split': 'train[:1000]',
                'extractor': self.extract_blended_skill,
                'description': 'Multi-skill conversations'
            },
            {
                'name': 'wizard_of_wikipedia',
                'config': None,
                'split': 'train[:800]',
                'extractor': self.extract_wizard_wikipedia,
                'description': 'Knowledge-grounded dialogues'
            }
        ]
        
        total_added = 0
        
        for config in datasets_config:
            try:
                print(f"📥 Loading {config['name']} - {config['description']}...")
                
                if config['config']:
                    dataset = load_dataset(config['name'], config['config'], split=config['split'])
                else:
                    dataset = load_dataset(config['name'], split=config['split'])
                
                added = 0
                for sample in dataset:
                    try:
                        texts = config['extractor'](sample)
                        for text in texts:
                            if self.add_document(text, {
                                'source': config['name'],
                                'type': config['description'],
                                'timestamp': datetime.now().isoformat()
                            }):
                                added += 1
                    except Exception as e:
                        continue
                
                print(f"✅ Added {added} documents from {config['name']}")
                total_added += added
                
            except Exception as e:
                print(f"❌ Error loading {config['name']}: {e}")
                continue
        
        # Add scientific knowledge
        self.add_scientific_knowledge()
        
        # Add programming knowledge
        self.add_programming_knowledge()
        
        print(f"🎯 Total documents added: {total_added + 100}")  # +100 from manual knowledge
        
        # Build index
        if self.build_index():
            self.save_vector_db()
        
        return total_added
    
    def extract_daily_dialog(self, sample):
        """Extract from daily_dialog dataset"""
        dialog = sample.get('dialog', [])
        if len(dialog) >= 2:
            # Create Q&A pairs
            texts = []
            for i in range(0, len(dialog)-1, 2):
                if i+1 < len(dialog):
                    question = dialog[i]
                    answer = dialog[i+1]
                    texts.append(f"Q: {question} A: {answer}")
            return texts
        return []
    
    def extract_empathetic(self, sample):
        """Extract from empathetic_dialogues"""
        context = sample.get('context', '')
        response = sample.get('response', '')
        emotion = sample.get('emotion', '')
        
        if context and response:
            return [f"Context: {context} [Emotion: {emotion}] Response: {response}"]
        return []
    
    def extract_persona_chat(self, sample):
        """Extract from persona_chat"""
        history = sample.get('history', [])
        personality = sample.get('personality', [])
        
        if history and len(history) >= 2:
            persona_text = " ".join(personality[:2])  # First 2 personality traits
            conversation = " ".join(history[-4:])  # Last 4 turns
            return [f"Personality: {persona_text} Conversation: {conversation}"]
        return []
    
    def extract_blended_skill(self, sample):
        """Extract from blended_skill_talk"""
        dialog = sample.get('dialog', [])
        if len(dialog) >= 2:
            # Focus on knowledge-empathy-personality blend
            return [" ".join(dialog[-3:])]  # Last 3 turns
        return []
    
    def extract_wizard_wikipedia(self, sample):
        """Extract from wizard_of_wikipedia"""
        history = sample.get('history', [])
        knowledge = sample.get('knowledge', '')
        
        if history and knowledge:
            conversation = " ".join(history[-2:])  # Last 2 turns
            return [f"Knowledge: {knowledge[:200]} Conversation: {conversation}"]
        return []
    
    def add_scientific_knowledge(self):
        """Add scientific and educational knowledge"""
        scientific_knowledge = [
            "Photosynthesis is the process by which plants convert light energy into chemical energy, producing glucose and oxygen from carbon dioxide and water.",
            "DNA (Deoxyribonucleic acid) contains the genetic instructions for the development and function of living organisms, structured as a double helix.",
            "The theory of relativity, developed by Einstein, describes the relationship between space, time, and gravity in the universe.",
            "Quantum mechanics is the branch of physics that deals with the behavior of matter and energy at the atomic and subatomic level.",
            "The water cycle involves evaporation, condensation, precipitation, and collection, continuously moving water through Earth's systems.",
            "Artificial intelligence refers to computer systems that can perform tasks typically requiring human intelligence, such as learning and decision-making.",
            "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities since the Industrial Revolution.",
            "The human brain contains approximately 86 billion neurons that communicate through electrical and chemical signals.",
            "Evolution is the process by which species change over time through natural selection, genetic drift, and other mechanisms.",
            "The periodic table organizes chemical elements by their atomic number, revealing patterns in their properties and behaviors."
        ]
        
        for knowledge in scientific_knowledge:
            self.add_document(knowledge, {
                'source': 'manual',
                'type': 'scientific_knowledge',
                'category': 'education'
            })
        
        print("✅ Added scientific knowledge")
    
    def add_programming_knowledge(self):
        """Add programming and technology knowledge"""
        programming_knowledge = [
            "Python is a high-level programming language known for its simplicity and readability, widely used in data science, web development, and AI.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without explicit programming.",
            "APIs (Application Programming Interfaces) allow different software applications to communicate and share data with each other.",
            "Version control systems like Git help developers track changes in code and collaborate on software projects effectively.",
            "Databases store and organize data in structured ways, with SQL being a common language for querying relational databases.",
            "Web development involves creating websites and web applications using languages like HTML, CSS, JavaScript, and various frameworks.",
            "Cybersecurity focuses on protecting digital systems, networks, and data from unauthorized access, attacks, and damage.",
            "Cloud computing provides on-demand access to computing resources and services over the internet, enabling scalable applications.",
            "Data structures like arrays, lists, trees, and graphs organize data efficiently for different computational tasks and algorithms.",
            "Software engineering involves applying engineering principles to design, develop, test, and maintain large-scale software systems."
        ]
        
        for knowledge in programming_knowledge:
            self.add_document(knowledge, {
                'source': 'manual',
                'type': 'programming_knowledge',
                'category': 'technology'
            })
        
        print("✅ Added programming knowledge")
    
    def add_document(self, text, metadata=None):
        """Add document to vector database"""
        try:
            if len(text.strip()) < 10:  # Skip very short texts
                return False
            
            embedding = self.model.encode(text)
            self.embeddings.append(embedding)
            self.documents.append(text)
            self.metadata.append(metadata or {})
            return True
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def build_index(self):
        """Build FAISS index for fast search"""
        try:
            if not self.embeddings:
                return False
            
            embeddings_array = np.array(self.embeddings).astype('float32')
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            
            print(f"✅ Built FAISS index with {len(self.embeddings)} documents")
            return True
        except Exception as e:
            logger.error(f"Error building index: {e}")
            return False
    
    def save_vector_db(self):
        """Save the trained vector database"""
        try:
            # Save vector database
            vector_data = {
                'documents': self.documents,
                'metadata': self.metadata,
                'embeddings': [emb.tolist() for emb in self.embeddings],
                'model_name': 'all-MiniLM-L6-v2',
                'created_at': datetime.now().isoformat(),
                'total_documents': len(self.documents)
            }
            
            with open('smart_vector_db.json', 'w') as f:
                json.dump(vector_data, f)
            
            print("✅ Saved smart vector database to smart_vector_db.json")
            
            # Save FAISS index
            if self.index:
                faiss.write_index(self.index, "smart_vector_index.faiss")
                print("✅ Saved FAISS index to smart_vector_index.faiss")
            
            return True
        except Exception as e:
            logger.error(f"Error saving vector DB: {e}")
            return False
    
    def test_search(self):
        """Test the trained vector database"""
        test_queries = [
            "What is artificial intelligence?",
            "How do I learn programming?",
            "Tell me about photosynthesis",
            "I'm feeling sad today",
            "What's the weather like?"
        ]
        
        print("\n🧪 Testing vector search...")
        
        for query in test_queries:
            try:
                query_embedding = self.model.encode([query]).astype('float32')
                faiss.normalize_L2(query_embedding)
                
                if self.index:
                    scores, indices = self.index.search(query_embedding, 3)
                    
                    print(f"\nQuery: {query}")
                    for score, idx in zip(scores[0], indices[0]):
                        if idx < len(self.documents):
                            print(f"  Score: {score:.3f} - {self.documents[idx][:100]}...")
            except Exception as e:
                print(f"Error testing query '{query}': {e}")

def main():
    """Main training function"""
    print("🎯 Starting Smart Training for LamAI...")
    print("=" * 60)
    
    trainer = SmartTrainer()
    
    # Train with datasets
    total_docs = trainer.train_from_datasets()
    
    if total_docs > 0:
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Total documents trained: {total_docs}")
        
        # Test the system
        trainer.test_search()
        
        print(f"\n✅ Smart vector database ready!")
        print(f"📁 Files created:")
        print(f"   - smart_vector_db.json")
        print(f"   - smart_vector_index.faiss")
    else:
        print("❌ Training failed - no documents added")

if __name__ == "__main__":
    main()