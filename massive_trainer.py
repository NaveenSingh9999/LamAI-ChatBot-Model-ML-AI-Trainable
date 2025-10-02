#!/usr/bin/env python3
"""
MASSIVE LamAI Training Script
Train with 10+ HuggingFace datasets to make LamAI truly intelligent
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

class MassiveTrainer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.training_data = []
        self.embeddings = []
        
    def train_with_massive_datasets(self):
        """Train with 10+ diverse HuggingFace datasets"""
        
        datasets_config = [
            # Conversation datasets
            ("daily_dialog", "daily_dialog", self.process_daily_dialog, 2000),
            ("empathetic_dialogues", "empathetic_dialogues", self.process_empathetic, 1500),
            ("persona_chat", "persona_chat", self.process_persona_chat, 1000),
            ("blended_skill_talk", "blended_skill_talk", self.process_blended_skill, 1000),
            
            # Question Answering datasets
            ("squad", "squad", self.process_squad, 3000),
            ("natural_questions", "natural_questions", self.process_natural_questions, 2000),
            ("ms_marco", "ms_marco", self.process_ms_marco, 1500),
            
            # Knowledge datasets
            ("wiki_qa", "wiki_qa", self.process_wiki_qa, 1000),
            ("searchqa", "searchqa", self.process_search_qa, 1000),
            
            # Common sense and reasoning
            ("commonsense_qa", "commonsense_qa", self.process_commonsense, 1500),
            ("hellaswag", "hellaswag", self.process_hellaswag, 1000),
            
            # Math and logic
            ("math_qa", "math_qa", self.process_math_qa, 1000),
            ("gsm8k", "gsm8k", self.process_gsm8k, 800),
            
            # General knowledge
            ("trivia_qa", "trivia_qa", self.process_trivia_qa, 1500),
        ]
        
        print("🚀 Starting MASSIVE LamAI Training with 10+ datasets...")
        print("=" * 70)
        
        total_trained = 0
        
        for dataset_name, config, processor, max_samples in datasets_config:
            try:
                print(f"\n📥 Loading {dataset_name}...")
                
                # Load dataset
                if config == dataset_name:
                    dataset = load_dataset(dataset_name, split=f"train[:{max_samples}]")
                else:
                    dataset = load_dataset(dataset_name, config, split=f"train[:{max_samples}]")
                
                # Process with specific processor
                processed_count = processor(dataset, dataset_name)
                total_trained += processed_count
                
                print(f"✅ Processed {processed_count} samples from {dataset_name}")
                
            except Exception as e:
                print(f"⚠️ Error with {dataset_name}: {e}")
                # Try alternative datasets
                self.add_fallback_data(dataset_name)
        
        print(f"\n🎯 TOTAL TRAINING DATA: {total_trained} high-quality samples")
        print(f"📊 Categories: Conversations, Q&A, Knowledge, Math, Logic, Common Sense")
        
        # Generate embeddings
        self.generate_embeddings()
        
        # Save the massive knowledge base
        self.save_massive_kb()
        
        return total_trained
    
    def process_daily_dialog(self, dataset, source):
        """Process daily conversation data"""
        count = 0
        for sample in dataset:
            try:
                dialog = sample.get('dialog', [])
                if len(dialog) >= 2:
                    for i in range(0, len(dialog)-1, 2):
                        if i+1 < len(dialog):
                            user_msg = dialog[i]
                            ai_response = dialog[i+1]
                            
                            self.training_data.append({
                                'user': user_msg,
                                'response': ai_response,
                                'category': 'daily_conversation',
                                'source': source,
                                'type': 'dialog'
                            })
                            count += 1
            except Exception as e:
                continue
        return count
    
    def process_empathetic(self, dataset, source):
        """Process empathetic dialogue data"""
        count = 0
        for sample in dataset:
            try:
                context = sample.get('context', '')
                response = sample.get('response', '')
                emotion = sample.get('emotion', 'neutral')
                
                if context and response:
                    self.training_data.append({
                        'user': context,
                        'response': response,
                        'category': 'empathy',
                        'emotion': emotion,
                        'source': source,
                        'type': 'empathetic_response'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def process_persona_chat(self, dataset, source):
        """Process personality-based conversations"""
        count = 0
        for sample in dataset:
            try:
                history = sample.get('history', [])
                personality = sample.get('personality', [])
                
                if len(history) >= 2:
                    user_msg = history[-2] if len(history) > 1 else ""
                    ai_response = history[-1]
                    
                    self.training_data.append({
                        'user': user_msg,
                        'response': ai_response,
                        'category': 'personality',
                        'personality': personality,
                        'source': source,
                        'type': 'persona_dialog'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def process_blended_skill(self, dataset, source):
        """Process multi-skill conversations"""
        count = 0
        for sample in dataset:
            try:
                dialog = sample.get('dialog', [])
                if len(dialog) >= 2:
                    for i in range(0, len(dialog)-1, 2):
                        if i+1 < len(dialog):
                            user_msg = dialog[i]
                            ai_response = dialog[i+1]
                            
                            self.training_data.append({
                                'user': user_msg,
                                'response': ai_response,
                                'category': 'multi_skill',
                                'source': source,
                                'type': 'blended_conversation'
                            })
                            count += 1
            except Exception as e:
                continue
        return count
    
    def process_squad(self, dataset, source):
        """Process SQuAD question-answering data"""
        count = 0
        for sample in dataset:
            try:
                question = sample.get('question', '')
                answers = sample.get('answers', {})
                context = sample.get('context', '')
                
                if question and answers.get('text'):
                    answer = answers['text'][0] if answers['text'] else ""
                    
                    self.training_data.append({
                        'user': question,
                        'response': f"{answer}\n\nContext: {context[:200]}...",
                        'category': 'question_answering',
                        'source': source,
                        'type': 'factual_qa'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def process_natural_questions(self, dataset, source):
        """Process Natural Questions data"""
        count = 0
        for sample in dataset:
            try:
                question = sample.get('question', {}).get('text', '')
                annotations = sample.get('annotations', [])
                
                if question and annotations:
                    for annotation in annotations[:1]:  # Take first annotation
                        if annotation.get('short_answers'):
                            short_answer = annotation['short_answers'][0].get('text', '')
                            if short_answer:
                                self.training_data.append({
                                    'user': question,
                                    'response': short_answer,
                                    'category': 'natural_qa',
                                    'source': source,
                                    'type': 'short_answer'
                                })
                                count += 1
            except Exception as e:
                continue
        return count
    
    def process_ms_marco(self, dataset, source):
        """Process MS MARCO data"""
        count = 0
        for sample in dataset:
            try:
                query = sample.get('query', '')
                passages = sample.get('passages', [])
                
                if query and passages:
                    # Use the first relevant passage
                    passage = passages[0].get('passage_text', '') if passages else ""
                    if passage:
                        self.training_data.append({
                            'user': query,
                            'response': passage[:500] + "..." if len(passage) > 500 else passage,
                            'category': 'search_qa',
                            'source': source,
                            'type': 'passage_retrieval'
                        })
                        count += 1
            except Exception as e:
                continue
        return count
    
    def process_wiki_qa(self, dataset, source):
        """Process WikiQA data"""
        count = 0
        for sample in dataset:
            try:
                question = sample.get('question', '')
                answer = sample.get('answer', '')
                label = sample.get('label', 0)
                
                if question and answer and label == 1:  # Only correct answers
                    self.training_data.append({
                        'user': question,
                        'response': answer,
                        'category': 'wiki_knowledge',
                        'source': source,
                        'type': 'wiki_qa'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def process_search_qa(self, dataset, source):
        """Process SearchQA data"""
        count = 0
        for sample in dataset:
            try:
                question = sample.get('question', '')
                answer = sample.get('answer', '')
                
                if question and answer:
                    self.training_data.append({
                        'user': question,
                        'response': answer,
                        'category': 'search_knowledge',
                        'source': source,
                        'type': 'search_qa'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def process_commonsense(self, dataset, source):
        """Process CommonsenseQA data"""
        count = 0
        for sample in dataset:
            try:
                question = sample.get('question', '')
                choices = sample.get('choices', {})
                answer_key = sample.get('answerKey', '')
                
                if question and choices.get('text') and answer_key:
                    labels = choices.get('label', [])
                    texts = choices.get('text', [])
                    
                    # Find the correct answer
                    correct_answer = ""
                    for i, label in enumerate(labels):
                        if label == answer_key and i < len(texts):
                            correct_answer = texts[i]
                            break
                    
                    if correct_answer:
                        self.training_data.append({
                            'user': question,
                            'response': f"The answer is: {correct_answer}. This involves common sense reasoning about everyday situations and logical thinking.",
                            'category': 'common_sense',
                            'source': source,
                            'type': 'reasoning'
                        })
                        count += 1
            except Exception as e:
                continue
        return count
    
    def process_hellaswag(self, dataset, source):
        """Process HellaSwag data"""
        count = 0
        for sample in dataset:
            try:
                context = sample.get('ctx', '')
                endings = sample.get('endings', [])
                label = sample.get('label', -1)
                
                if context and endings and 0 <= label < len(endings):
                    correct_ending = endings[label]
                    
                    self.training_data.append({
                        'user': f"Complete this scenario: {context}",
                        'response': f"The most logical continuation is: {correct_ending}",
                        'category': 'scenario_completion',
                        'source': source,
                        'type': 'logical_reasoning'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def process_math_qa(self, dataset, source):
        """Process MathQA data"""
        count = 0
        for sample in dataset:
            try:
                problem = sample.get('Problem', '')
                rationale = sample.get('Rationale', '')
                correct_answer = sample.get('correct', '')
                
                if problem and rationale:
                    self.training_data.append({
                        'user': problem,
                        'response': f"Let me solve this step by step:\n{rationale}\n\nAnswer: {correct_answer}",
                        'category': 'mathematics',
                        'source': source,
                        'type': 'math_problem'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def process_gsm8k(self, dataset, source):
        """Process GSM8K math word problems"""
        count = 0
        for sample in dataset:
            try:
                question = sample.get('question', '')
                answer = sample.get('answer', '')
                
                if question and answer:
                    self.training_data.append({
                        'user': question,
                        'response': f"Here's how to solve this math problem:\n{answer}",
                        'category': 'math_word_problems',
                        'source': source,
                        'type': 'word_problem'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def process_trivia_qa(self, dataset, source):
        """Process TriviaQA data"""
        count = 0
        for sample in dataset:
            try:
                question = sample.get('question', '')
                answer = sample.get('answer', {})
                
                if question and answer.get('value'):
                    answer_text = answer['value']
                    
                    self.training_data.append({
                        'user': question,
                        'response': answer_text,
                        'category': 'trivia_knowledge',
                        'source': source,
                        'type': 'trivia'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def add_fallback_data(self, failed_dataset):
        """Add high-quality fallback data when a dataset fails"""
        fallback_data = [
            {
                'user': 'What is artificial intelligence?',
                'response': 'Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think, learn, and make decisions. It encompasses machine learning, natural language processing, computer vision, and robotics.',
                'category': 'technology_knowledge',
                'source': 'fallback',
                'type': 'definition'
            },
            {
                'user': 'How do I solve math problems?',
                'response': 'To solve math problems effectively: 1) Read the problem carefully, 2) Identify what you know and what you need to find, 3) Choose the right mathematical approach, 4) Work through the solution step by step, 5) Check your answer.',
                'category': 'mathematics',
                'source': 'fallback',
                'type': 'methodology'
            },
            {
                'user': 'Tell me about empathy',
                'response': 'Empathy is the ability to understand and share the feelings of others. It involves putting yourself in someone else\'s shoes and responding with compassion and understanding. Empathy builds stronger relationships and better communication.',
                'category': 'empathy',
                'source': 'fallback',
                'type': 'emotional_intelligence'
            }
        ]
        
        self.training_data.extend(fallback_data)
        print(f"✅ Added {len(fallback_data)} fallback samples for {failed_dataset}")
    
    def generate_embeddings(self):
        """Generate embeddings for all training data"""
        print("\n🧠 Generating embeddings for massive dataset...")
        
        # Combine user input and response for better context understanding
        texts_for_embedding = []
        for item in self.training_data:
            combined_text = f"User: {item['user']} AI: {item['response']}"
            texts_for_embedding.append(combined_text)
        
        # Generate embeddings in batches
        batch_size = 100
        for i in range(0, len(texts_for_embedding), batch_size):
            batch = texts_for_embedding[i:i+batch_size]
            batch_embeddings = self.model.encode(batch)
            self.embeddings.extend(batch_embeddings.tolist())
            
            if i % 500 == 0:
                print(f"  🔄 Processed {i + len(batch)}/{len(texts_for_embedding)} embeddings...")
        
        print(f"✅ Generated {len(self.embeddings)} embeddings")
    
    def save_massive_kb(self):
        """Save the massive knowledge base"""
        kb_data = {
            'training_data': self.training_data,
            'embeddings': self.embeddings,
            'metadata': {
                'total_samples': len(self.training_data),
                'categories': list(set(item['category'] for item in self.training_data)),
                'sources': list(set(item['source'] for item in self.training_data)),
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        }
        
        with open('massive_smart_kb.json', 'w') as f:
            json.dump(kb_data, f, indent=2)
        
        print(f"\n💾 Saved massive knowledge base: {len(self.training_data)} samples")
        print(f"📊 Categories: {', '.join(kb_data['metadata']['categories'])}")
        print(f"🗃️ File: massive_smart_kb.json")

if __name__ == "__main__":
    trainer = MassiveTrainer()
    total_trained = trainer.train_with_massive_datasets()
    print(f"\n🎉 MASSIVE TRAINING COMPLETE!")
    print(f"🚀 LamAI is now trained with {total_trained} high-quality samples!")
    print(f"💡 Ready to be truly intelligent!")