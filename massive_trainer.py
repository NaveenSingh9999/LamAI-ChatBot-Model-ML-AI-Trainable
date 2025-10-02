#!/usr/bin/env python3
"""
MASSIVE LamAI Training Script
Train with high-quality HuggingFace datasets using efficient vector storage
Manages cache automatically to prevent storage bloat
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import warnings
import shutil
warnings.filterwarnings('ignore')

class MassiveTrainer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.training_data = []
        self.embeddings = []
        self.cache_dir = os.path.expanduser('~/.cache/huggingface')
        self.kb_file = 'massive_smart_kb.json'
        
    def load_existing_knowledge(self):
        """Load existing knowledge base to preserve previous training"""
        if os.path.exists(self.kb_file):
            try:
                with open(self.kb_file, 'r', encoding='utf-8') as f:
                    existing_kb = json.load(f)
                
                if 'training_data' in existing_kb:
                    self.training_data = existing_kb['training_data']
                    print(f"📚 Loaded {len(self.training_data)} existing knowledge samples")
                    
                    # Load existing embeddings if available
                    if 'embeddings' in existing_kb:
                        self.embeddings = existing_kb['embeddings']
                        print(f"🧠 Loaded {len(self.embeddings)} existing embeddings")
                    else:
                        # Generate embeddings for existing data
                        print("🔄 Generating embeddings for existing data...")
                        self.generate_embeddings()
                else:
                    print("📝 No existing training data found")
            except Exception as e:
                print(f"⚠️ Error loading existing knowledge: {e}")
                print("🆕 Starting fresh...")
        else:
            print("🆕 No existing knowledge base found, starting fresh")
        
    def manage_cache(self):
        """Manage HuggingFace cache to prevent storage bloat"""
        try:
            if os.path.exists(self.cache_dir):
                cache_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(self.cache_dir)
                    for filename in filenames
                ) / (1024 * 1024 * 1024)  # Convert to GB
                
                print(f"📊 Current cache size: {cache_size:.2f} GB")
                
                # If cache is over 2GB, clean it
                if cache_size > 2.0:
                    print("🧹 Cache is large, cleaning old files...")
                    shutil.rmtree(self.cache_dir, ignore_errors=True)
                    print("✅ Cache cleaned!")
        except Exception as e:
            print(f"⚠️ Cache management warning: {e}")
    
    def train_incremental_batch(self, batch_size=2):
        """Train with 2 datasets at a time to prevent memory overload and preserve knowledge"""
        
        # Load existing knowledge first
        self.load_existing_knowledge()
        
        # ALL VAST DATASETS FOR SUPER INTELLIGENCE - MASSIVE EXPANSION
        all_datasets = [
            # EXISTING WORKING DATASETS
            ("squad", "squad", self.process_squad, 1000, "Q&A"),
            ("commonsense_qa", "commonsense_qa", self.process_commonsense, 500, "Reasoning"),
            ("coqa", "coqa", self.process_coqa, 300, "Conversational QA"),
            ("drop", "drop", self.process_drop, 300, "Reading Comprehension"),
            
            # VAST ADDITIONAL DATASETS FOR SUPER INTELLIGENCE
            ("boolq", "boolq", self.process_boolq, 500, "Boolean Questions"),
            ("multirc", "multirc", self.process_multirc, 400, "Multi-Choice Reading"),
            ("wic", "wic", self.process_wic, 300, "Word in Context"),
            ("copa", "copa", self.process_copa, 200, "Causal Reasoning"),
            ("rte", "rte", self.process_rte, 300, "Textual Entailment"),
            ("cb", "cb", self.process_cb, 200, "Commitment Bank"),
            ("wsc", "wsc", self.process_wsc, 200, "Winograd Schema"),
            ("record", "record", self.process_record, 400, "Reading Comprehension"),
            ("openbookqa", "openbookqa", self.process_openbookqa, 300, "Open Book QA"),
            ("social_i_qa", "social_i_qa", self.process_social_iqa, 400, "Social Intelligence"),
            ("winogrande", "winogrande", self.process_winogrande, 500, "Commonsense Reasoning"),
            ("sciq", "sciq", self.process_sciq, 400, "Science Questions"),
            ("cosmos_qa", "cosmos_qa", self.process_cosmos_qa, 300, "Commonsense QA"),
            ("dream", "dream", self.process_dream, 300, "Dialogue Reasoning"),
        ]
        
        # Track which datasets we've already processed
        processed_sources = set()
        if self.training_data:
            processed_sources = set([str(item.get('source', '')) for item in self.training_data])
            print(f"📋 Already processed: {', '.join(processed_sources)}")
        
        # Filter out already processed datasets
        remaining_datasets = [d for d in all_datasets if d[0] not in processed_sources]
        
        if not remaining_datasets:
            print("✅ All datasets already processed!")
            return len(self.training_data)
        
        print(f"🎯 Processing {len(remaining_datasets)} remaining datasets in batches of {batch_size}")
        
        # Process in batches
        total_new_samples = 0
        batch_count = 0
        
        for i in range(0, len(remaining_datasets), batch_size):
            batch = remaining_datasets[i:i+batch_size]
            batch_count += 1
            
            print(f"\n🔄 BATCH {batch_count}: Processing {len(batch)} datasets...")
            print("=" * 50)
            
            batch_start_count = len(self.training_data)
            
            for dataset_name, dataset_config, processor_func, max_samples, description in batch:
                try:
                    print(f"\n📚 Loading {dataset_name} ({description})...")
                    
                    # Load dataset with streaming and proper config handling
                    if "/" in dataset_config:
                        parts = dataset_config.split("/")
                        ds_name = parts[0]
                        config = parts[1] if len(parts) > 1 else None
                        dataset = load_dataset(ds_name, config, split='train', streaming=True)
                    else:
                        dataset = load_dataset(dataset_config, split='train', streaming=True)
                    
                    # Process samples
                    processed_count = processor_func(dataset, max_samples)
                    total_new_samples += processed_count
                    
                    print(f"✅ Added {processed_count} samples from {dataset_name}")
                    
                except Exception as e:
                    print(f"⚠️ Error with {dataset_name}: {e}")
                    print(f"🔄 Adding fallback data...")
                    self.add_fallback_data(dataset_name)
            
            # Save progress after each batch
            batch_new_samples = len(self.training_data) - batch_start_count
            print(f"\n📊 Batch {batch_count} Summary:")
            print(f"   - New samples: {batch_new_samples}")
            print(f"   - Total samples: {len(self.training_data)}")
            
            # Generate embeddings for new data
            print("🧠 Updating embeddings...")
            self.generate_embeddings()
            
            # Save incremental progress
            self.save_massive_kb()
            print(f"💾 Progress saved! Knowledge preserved.")
            
            # Clean cache between batches
            if batch_count % 2 == 0:
                self.manage_cache()
        
        return len(self.training_data)
    
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
    
    def process_daily_dialog(self, dataset, max_samples):
        """Process daily conversation data from HuggingFace"""
        count = 0
        for sample in dataset:
            if count >= max_samples:
                break
            try:
                dialog = sample.get('dialog', [])
                if len(dialog) >= 2:
                    for i in range(0, len(dialog)-1, 2):
                        if i+1 < len(dialog) and count < max_samples:
                            user_msg = dialog[i]
                            ai_response = dialog[i+1]
                            
                            self.training_data.append({
                                'user': user_msg,
                                'response': ai_response,
                                'category': 'daily_conversation',
                                'source': 'daily_dialog',
                                'type': 'dialog'
                            })
                            count += 1
            except Exception as e:
                continue
        return count
    
    def process_empathetic(self, dataset, max_samples):
        """Process empathetic dialogue data from HuggingFace"""
        count = 0
        for sample in dataset:
            if count >= max_samples:
                break
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
                        'source': 'empathetic_dialogues',
                        'type': 'empathetic_response'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def process_squad(self, dataset, max_samples):
        """Process SQuAD question-answering data"""
        count = 0
        for sample in dataset:
            if count >= max_samples:
                break
            try:
                question = sample.get('question', '')
                answers = sample.get('answers', {})
                context = sample.get('context', '')
                
                if question and answers.get('text') and answers['text']:
                    answer = answers['text'][0]
                    
                    self.training_data.append({
                        'user': question,
                        'response': answer,
                        'category': 'qa',
                        'source': 'squad',
                        'type': 'reading_comprehension',
                        'context': context[:200]
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def process_natural_questions(self, dataset, max_samples):
        """Process Natural Questions data"""
        count = 0
        for sample in dataset:
            if count >= max_samples:
                break
            try:
                question = sample.get('question', '')
                answer = sample.get('answer', [''])[0] if sample.get('answer') else ''
                
                if question and answer:
                    self.training_data.append({
                        'user': question,
                        'response': answer,
                        'category': 'knowledge',
                        'source': 'natural_questions',
                        'type': 'factual_qa'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def process_commonsense(self, dataset, max_samples):
        """Process common sense reasoning data"""
        count = 0
        for sample in dataset:
            if count >= max_samples:
                break
            try:
                question = sample.get('question', '')
                choices = sample.get('choices', {})
                answer_key = sample.get('answerKey', '')
                
                if question and choices.get('text') and answer_key:
                    # Find the correct answer
                    choice_labels = choices.get('label', [])
                    choice_texts = choices.get('text', [])
                    
                    if answer_key in choice_labels:
                        answer_idx = choice_labels.index(answer_key)
                        if answer_idx < len(choice_texts):
                            answer = choice_texts[answer_idx]
                            
                            self.training_data.append({
                                'user': question,
                                'response': answer,
                                'category': 'reasoning',
                                'source': 'commonsense_qa',
                                'type': 'multiple_choice'
                            })
                            count += 1
            except Exception as e:
                continue
        return count
    
    def process_math_qa(self, dataset, max_samples):
        """Process math question-answering data"""
        count = 0
        for sample in dataset:
            if count >= max_samples:
                break
            try:
                problem = sample.get('Problem', '')
                answer = sample.get('correct', '')
                
                if problem and answer:
                    self.training_data.append({
                        'user': problem,
                        'response': f"The answer is {answer}",
                        'category': 'math',
                        'source': 'math_qa',
                        'type': 'math_problem'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def process_gsm8k(self, dataset, max_samples):
        """Process GSM8K math word problems"""
        count = 0
        for sample in dataset:
            if count >= max_samples:
                break
            try:
                question = sample.get('question', '')
                answer = sample.get('answer', '')
                
                if question and answer:
                    # Extract just the final answer number
                    import re
                    final_answer = re.findall(r'####\s*(.+)', answer)
                    if final_answer:
                        clean_answer = final_answer[0].strip()
                    else:
                        clean_answer = answer
                    
                    self.training_data.append({
                        'user': question,
                        'response': f"The answer is {clean_answer}",
                        'category': 'math_word_problems',
                        'source': 'gsm8k',
                        'type': 'word_problem'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
    
    def process_trivia_qa(self, dataset, max_samples):
        """Process trivia questions"""
        count = 0
        for sample in dataset:
            if count >= max_samples:
                break
            try:
                question = sample.get('question', '')
                answer = sample.get('answer', {})
                
                if question and answer.get('value'):
                    answer_text = answer['value']
                    
                    self.training_data.append({
                        'user': question,
                        'response': answer_text,
                        'category': 'trivia_knowledge',
                        'source': 'trivia_qa',
                        'type': 'trivia'
                    })
                    count += 1
            except Exception as e:
                continue
        return count
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
        """Generate embeddings for new training data (incremental)"""
        # Only generate embeddings for new data
        current_embedding_count = len(self.embeddings)
        new_data_count = len(self.training_data)
        
        if current_embedding_count < new_data_count:
            print(f"\n🧠 Generating embeddings for {new_data_count - current_embedding_count} new samples...")
            
            # Get texts for new samples only
            new_texts = []
            for i in range(current_embedding_count, new_data_count):
                item = self.training_data[i]
                combined_text = f"User: {item['user']} AI: {item['response']}"
                new_texts.append(combined_text)
            
            if new_texts:
                # Generate embeddings for new data in batches
                batch_size = 100
                new_embeddings = []
                
                for i in range(0, len(new_texts), batch_size):
                    batch = new_texts[i:i+batch_size]
                    batch_embeddings = self.model.encode(batch)
                    new_embeddings.extend(batch_embeddings.tolist())
                    
                    if len(new_embeddings) % 200 == 0:
                        print(f"  🔄 Processed {len(new_embeddings)}/{len(new_texts)} new embeddings...")
                
                # Append to existing embeddings
                self.embeddings.extend(new_embeddings)
                
                print(f"✅ Generated {len(new_embeddings)} new embeddings")
                print(f"📊 Total embeddings: {len(self.embeddings)}")
            
        else:
            print(f"✅ Embeddings up to date: {len(self.embeddings)} total")
    
    # NEW DATASET PROCESSORS FOR 10 ADDITIONAL QUALITY DATASETS
    
    def process_natural_questions(self, dataset, source):
        """Process Natural Questions dataset (real Google search questions)"""
        count = 0
        for sample in dataset:
            try:
                question = sample.get('question', {}).get('text', '') if isinstance(sample.get('question'), dict) else sample.get('question', '')
                annotations = sample.get('annotations', [])
                
                if question and annotations:
                    # Use first annotation that has an answer
                    for ann in annotations:
                        if ann.get('short_answers') or ann.get('long_answer', {}).get('start_byte', -1) >= 0:
                            answer = ann.get('short_answers', [{}])[0].get('text', '') if ann.get('short_answers') else "Yes, based on the context provided."
                            
                            self.training_data.append({
                                'user': question,
                                'response': answer or "Based on reliable sources, this requires more specific context to answer accurately.",
                                'category': 'qa',
                                'source': source,
                                'type': 'natural_question'
                            })
                            count += 1
                            break
                    
                if count >= 1000:
                    break
            except Exception as e:
                continue
        return count

    def process_eli5(self, dataset, source):
        """Process ELI5 (Explain Like I'm 5) dataset"""
        count = 0
        for sample in dataset:
            try:
                title = sample.get('title', '')
                selftext = sample.get('selftext', '')
                answers = sample.get('answers', {})
                
                question = title
                if selftext and selftext.strip():
                    question += f" {selftext.strip()}"
                
                if question and answers:
                    answer_texts = answers.get('text', [])
                    if answer_texts and isinstance(answer_texts, list):
                        best_answer = answer_texts[0] if answer_texts[0] else "Let me explain this in simple terms..."
                        
                        self.training_data.append({
                            'user': question,
                            'response': best_answer,
                            'category': 'qa',
                            'source': source,
                            'type': 'explanation'
                        })
                        count += 1
                        
                if count >= 800:
                    break
            except Exception as e:
                continue
        return count

    def process_hotpot_qa(self, dataset, source):
        """Process HotpotQA dataset (multi-hop reasoning)"""
        count = 0
        for sample in dataset:
            try:
                question = sample.get('question', '')
                answer = sample.get('answer', '')
                
                if question and answer:
                    self.training_data.append({
                        'user': question,
                        'response': answer,
                        'category': 'reasoning',
                        'source': source,
                        'type': 'multi_hop_reasoning'
                    })
                    count += 1
                    
                if count >= 600:
                    break
            except Exception as e:
                continue
        return count

    def process_ms_marco(self, dataset, source):
        """Process MS MARCO dataset (reading comprehension)"""
        count = 0
        for sample in dataset:
            try:
                query = sample.get('query', '')
                answers = sample.get('answers', [])
                
                if query and answers:
                    answer = answers[0] if isinstance(answers, list) and answers else str(answers)
                    
                    self.training_data.append({
                        'user': query,
                        'response': answer,
                        'category': 'qa',
                        'source': source,
                        'type': 'reading_comprehension'
                    })
                    count += 1
                    
                if count >= 700:
                    break
            except Exception as e:
                continue
        return count

    def process_coqa(self, dataset, source):
        """Process CoQA dataset (conversational Q&A)"""
        count = 0
        for sample in dataset:
            try:
                questions = sample.get('questions', [])
                answers = sample.get('answers', [])
                
                if questions and answers:
                    for i, (q, a) in enumerate(zip(questions, answers)):
                        question_text = q.get('input_text', '') if isinstance(q, dict) else str(q)
                        answer_text = a.get('input_text', '') if isinstance(a, dict) else str(a)
                        
                        if question_text and answer_text:
                            self.training_data.append({
                                'user': question_text,
                                'response': answer_text,
                                'category': 'qa',
                                'source': source,
                                'type': 'conversational_qa'
                            })
                            count += 1
                            
                        if count >= 500:
                            break
                            
                if count >= 500:
                    break
            except Exception as e:
                continue
        return count

    def process_quac(self, dataset, source):
        """Process QuAC dataset (Question Answering in Context)"""
        count = 0
        for sample in dataset:
            try:
                questions = sample.get('questions', [])
                answers = sample.get('answers', [])
                
                if questions and answers:
                    for q, a in zip(questions, answers):
                        question_text = q if isinstance(q, str) else q.get('text', '')
                        answer_text = a if isinstance(a, str) else a.get('text', '')
                        
                        if question_text and answer_text:
                            self.training_data.append({
                                'user': question_text,
                                'response': answer_text,
                                'category': 'qa',
                                'source': source,
                                'type': 'contextual_qa'
                            })
                            count += 1
                            
                        if count >= 400:
                            break
                            
                if count >= 400:
                    break
            except Exception as e:
                continue
        return count

    def process_drop(self, dataset, source):
        """Process DROP dataset (reading comprehension with operations)"""
        count = 0
        for sample in dataset:
            try:
                question = sample.get('question', '')
                answers_spans = sample.get('answers_spans', {})
                
                if question and answers_spans:
                    spans = answers_spans.get('spans', [])
                    if spans:
                        answer = spans[0] if isinstance(spans, list) else str(spans)
                        
                        self.training_data.append({
                            'user': question,
                            'response': answer,
                            'category': 'reasoning',
                            'source': source,
                            'type': 'reading_comprehension_ops'
                        })
                        count += 1
                        
                if count >= 500:
                    break
            except Exception as e:
                continue
        return count

    def process_narrativeqa(self, dataset, source):
        """Process NarrativeQA dataset (story understanding)"""
        count = 0
        for sample in dataset:
            try:
                question = sample.get('question', {}).get('text', '') if isinstance(sample.get('question'), dict) else sample.get('question', '')
                answers = sample.get('answers', [])
                
                if question and answers:
                    answer = answers[0].get('text', '') if isinstance(answers[0], dict) else str(answers[0])
                    
                    self.training_data.append({
                        'user': question,
                        'response': answer,
                        'category': 'qa',
                        'source': source,
                        'type': 'narrative_understanding'
                    })
                    count += 1
                    
                if count >= 300:
                    break
            except Exception as e:
                continue
        return count

    def process_race(self, dataset, source):
        """Process RACE dataset (reading comprehension)"""
        count = 0
        for sample in dataset:
            try:
                article = sample.get('article', '')
                question = sample.get('question', '')
                options = sample.get('options', [])
                answer = sample.get('answer', '')
                
                if question and options and answer:
                    # Find the correct answer from options
                    answer_idx = ord(answer.upper()) - ord('A') if len(answer) == 1 and answer.upper() in 'ABCD' else 0
                    correct_answer = options[answer_idx] if 0 <= answer_idx < len(options) else options[0]
                    
                    full_question = f"{article}\n\nQuestion: {question}\nOptions: {', '.join(options)}"
                    
                    self.training_data.append({
                        'user': full_question,
                        'response': f"The correct answer is: {correct_answer}",
                        'category': 'qa',
                        'source': source,
                        'type': 'reading_comprehension'
                    })
                    count += 1
                    
                if count >= 600:
                    break
            except Exception as e:
                continue
        return count

    def process_arc(self, dataset, source):
        """Process ARC dataset (science questions)"""
        count = 0
        for sample in dataset:
            try:
                question = sample.get('question', '')
                choices = sample.get('choices', {})
                answerKey = sample.get('answerKey', '')
                
                if question and choices and answerKey:
                    labels = choices.get('label', [])
                    texts = choices.get('text', [])
                    
                    if labels and texts:
                        # Find correct answer
                        correct_answer = ""
                        for label, text in zip(labels, texts):
                            if label == answerKey:
                                correct_answer = text
                                break
                        
                        if correct_answer:
                            self.training_data.append({
                                'user': f"{question} Options: {', '.join(texts)}",
                                'response': f"The correct answer is: {correct_answer}",
                                'category': 'qa',
                                'source': source,
                                'type': 'science_question'
                            })
                            count += 1
                            
                if count >= 400:
                    break
            except Exception as e:
                continue
        return count

    def process_piqa(self, dataset, source):
        """Process PIQA dataset (physical reasoning)"""
        count = 0
        for sample in dataset:
            try:
                goal = sample.get('goal', '')
                sol1 = sample.get('sol1', '')
                sol2 = sample.get('sol2', '')
                label = sample.get('label', 0)
                
                if goal and sol1 and sol2:
                    correct_solution = sol1 if label == 0 else sol2
                    
                    self.training_data.append({
                        'user': f"How to: {goal}\nOption A: {sol1}\nOption B: {sol2}",
                        'response': f"The correct approach is: {correct_solution}",
                        'category': 'reasoning',
                        'source': source,
                        'type': 'physical_reasoning'
                    })
                    count += 1
                    
                if count >= 300:
                    break
            except Exception as e:
                continue
        return count

    # VAST DATASET PROCESSORS FOR SUPER INTELLIGENCE
    
    def process_boolq(self, dataset, source):
        """Process BoolQ dataset (boolean questions)"""
        count = 0
        for sample in dataset:
            try:
                passage = sample.get('passage', '')
                question = sample.get('question', '')
                answer = sample.get('answer', False)
                
                if passage and question:
                    answer_text = "Yes" if answer else "No"
                    full_question = f"Based on this passage: {passage}\n\nQuestion: {question}"
                    
                    self.training_data.append({
                        'user': full_question,
                        'response': f"{answer_text}. {passage[:100]}..." if len(passage) > 100 else f"{answer_text}. {passage}",
                        'category': 'qa',
                        'source': source,
                        'type': 'boolean_qa'
                    })
                    count += 1
                    
                if count >= 500:
                    break
            except Exception as e:
                continue
        return count

    def process_multirc(self, dataset, source):
        """Process MultiRC dataset (multi-choice reading)"""
        count = 0
        for sample in dataset:
            try:
                paragraph = sample.get('paragraph', '')
                question = sample.get('question', '')
                answer = sample.get('answer', '')
                
                if paragraph and question and answer:
                    self.training_data.append({
                        'user': f"Passage: {paragraph}\n\nQuestion: {question}",
                        'response': answer,
                        'category': 'reasoning',
                        'source': source,
                        'type': 'multi_choice_reading'
                    })
                    count += 1
                    
                if count >= 400:
                    break
            except Exception as e:
                continue
        return count

    def process_wic(self, dataset, source):
        """Process WiC dataset (word in context)"""
        count = 0
        for sample in dataset:
            try:
                word = sample.get('word', '')
                sentence1 = sample.get('sentence1', '')
                sentence2 = sample.get('sentence2', '')
                label = sample.get('label', False)
                
                if word and sentence1 and sentence2:
                    question = f"Does the word '{word}' have the same meaning in these two sentences?\n1: {sentence1}\n2: {sentence2}"
                    answer = "Yes, the word has the same meaning in both sentences." if label else "No, the word has different meanings in these sentences."
                    
                    self.training_data.append({
                        'user': question,
                        'response': answer,
                        'category': 'reasoning',
                        'source': source,
                        'type': 'word_meaning'
                    })
                    count += 1
                    
                if count >= 300:
                    break
            except Exception as e:
                continue
        return count

    def process_copa(self, dataset, source):
        """Process COPA dataset (causal reasoning)"""
        count = 0
        for sample in dataset:
            try:
                premise = sample.get('premise', '')
                choice1 = sample.get('choice1', '')
                choice2 = sample.get('choice2', '')
                question = sample.get('question', '')
                label = sample.get('label', 0)
                
                if premise and choice1 and choice2:
                    correct_choice = choice1 if label == 0 else choice2
                    question_type = "What was the CAUSE of this?" if question == "cause" else "What happened as a RESULT of this?"
                    
                    self.training_data.append({
                        'user': f"{premise}\n\n{question_type}\nA: {choice1}\nB: {choice2}",
                        'response': f"The correct answer is: {correct_choice}",
                        'category': 'reasoning',
                        'source': source,
                        'type': 'causal_reasoning'
                    })
                    count += 1
                    
                if count >= 200:
                    break
            except Exception as e:
                continue
        return count

    def process_rte(self, dataset, source):
        """Process RTE dataset (textual entailment)"""
        count = 0
        for sample in dataset:
            try:
                sentence1 = sample.get('sentence1', '')
                sentence2 = sample.get('sentence2', '')
                label = sample.get('label', 0)
                
                if sentence1 and sentence2:
                    entailment = "entailment" if label == 0 else "not entailment"
                    
                    self.training_data.append({
                        'user': f"Premise: {sentence1}\nHypothesis: {sentence2}\n\nDoes the premise entail the hypothesis?",
                        'response': f"The relationship is: {entailment}. This means the hypothesis {'can be inferred from' if label == 0 else 'cannot be inferred from'} the premise.",
                        'category': 'reasoning',
                        'source': source,
                        'type': 'textual_entailment'
                    })
                    count += 1
                    
                if count >= 300:
                    break
            except Exception as e:
                continue
        return count

    def process_cb(self, dataset, source):
        """Process CB dataset (commitment bank)"""
        count = 0
        for sample in dataset:
            try:
                premise = sample.get('premise', '')
                hypothesis = sample.get('hypothesis', '')
                label = sample.get('label', 0)
                
                if premise and hypothesis:
                    labels = ['entailment', 'contradiction', 'neutral']
                    label_text = labels[label] if label < len(labels) else 'neutral'
                    
                    self.training_data.append({
                        'user': f"Premise: {premise}\nHypothesis: {hypothesis}\n\nWhat is the logical relationship?",
                        'response': f"The relationship is: {label_text}. The hypothesis {'follows from' if label_text == 'entailment' else 'contradicts' if label_text == 'contradiction' else 'is neutral with respect to'} the premise.",
                        'category': 'reasoning',
                        'source': source,
                        'type': 'logical_reasoning'
                    })
                    count += 1
                    
                if count >= 200:
                    break
            except Exception as e:
                continue
        return count

    def process_wsc(self, dataset, source):
        """Process WSC dataset (Winograd schema)"""
        count = 0
        for sample in dataset:
            try:
                text = sample.get('text', '')
                target = sample.get('target', {})
                label = sample.get('label', False)
                
                if text and target:
                    span1_text = target.get('span1_text', '')
                    span2_text = target.get('span2_text', '')
                    
                    if span1_text and span2_text:
                        answer = f"Yes, '{span1_text}' refers to '{span2_text}'" if label else f"No, '{span1_text}' does not refer to '{span2_text}'"
                        
                        self.training_data.append({
                            'user': f"In this sentence: {text}\n\nDoes '{span1_text}' refer to '{span2_text}'?",
                            'response': answer,
                            'category': 'reasoning',
                            'source': source,
                            'type': 'coreference_resolution'
                        })
                        count += 1
                        
                if count >= 200:
                    break
            except Exception as e:
                continue
        return count

    def process_record(self, dataset, source):
        """Process ReCoRD dataset (reading comprehension)"""
        count = 0
        for sample in dataset:
            try:
                passage = sample.get('passage', '')
                query = sample.get('query', '')
                answers = sample.get('answers', [])
                
                if passage and query and answers:
                    answer = answers[0] if isinstance(answers, list) and answers else str(answers)
                    
                    self.training_data.append({
                        'user': f"Passage: {passage}\n\nQuestion: {query}",
                        'response': answer,
                        'category': 'qa',
                        'source': source,
                        'type': 'reading_comprehension'
                    })
                    count += 1
                    
                if count >= 400:
                    break
            except Exception as e:
                continue
        return count

    def process_openbookqa(self, dataset, source):
        """Process OpenBookQA dataset"""
        count = 0
        for sample in dataset:
            try:
                question_stem = sample.get('question_stem', '')
                choices = sample.get('choices', {})
                answerKey = sample.get('answerKey', '')
                
                if question_stem and choices and answerKey:
                    labels = choices.get('label', [])
                    texts = choices.get('text', [])
                    
                    if labels and texts:
                        correct_answer = ""
                        for label, text in zip(labels, texts):
                            if label == answerKey:
                                correct_answer = text
                                break
                        
                        if correct_answer:
                            self.training_data.append({
                                'user': f"{question_stem}\nOptions: {', '.join(texts)}",
                                'response': f"The correct answer is: {correct_answer}",
                                'category': 'qa',
                                'source': source,
                                'type': 'open_book_qa'
                            })
                            count += 1
                            
                if count >= 300:
                    break
            except Exception as e:
                continue
        return count

    def process_social_iqa(self, dataset, source):
        """Process Social IQA dataset"""
        count = 0
        for sample in dataset:
            try:
                context = sample.get('context', '')
                question = sample.get('question', '')
                answerA = sample.get('answerA', '')
                answerB = sample.get('answerB', '')
                answerC = sample.get('answerC', '')
                label = sample.get('label', '1')
                
                if context and question and answerA and answerB and answerC:
                    answers = [answerA, answerB, answerC]
                    correct_answer = answers[int(label) - 1] if label.isdigit() and 1 <= int(label) <= 3 else answerA
                    
                    self.training_data.append({
                        'user': f"Context: {context}\nQuestion: {question}\nA: {answerA}\nB: {answerB}\nC: {answerC}",
                        'response': f"The correct answer is: {correct_answer}",
                        'category': 'reasoning',
                        'source': source,
                        'type': 'social_intelligence'
                    })
                    count += 1
                    
                if count >= 400:
                    break
            except Exception as e:
                continue
        return count

    def process_winogrande(self, dataset, source):
        """Process WinoGrande dataset"""
        count = 0
        for sample in dataset:
            try:
                sentence = sample.get('sentence', '')
                option1 = sample.get('option1', '')
                option2 = sample.get('option2', '')
                answer = sample.get('answer', '1')
                
                if sentence and option1 and option2:
                    correct_option = option1 if answer == '1' else option2
                    
                    self.training_data.append({
                        'user': f"Complete the sentence: {sentence}\nOption 1: {option1}\nOption 2: {option2}",
                        'response': f"The correct completion is: {correct_option}",
                        'category': 'reasoning',
                        'source': source,
                        'type': 'commonsense_reasoning'
                    })
                    count += 1
                    
                if count >= 500:
                    break
            except Exception as e:
                continue
        return count

    def process_sciq(self, dataset, source):
        """Process SciQ dataset (science questions)"""
        count = 0
        for sample in dataset:
            try:
                question = sample.get('question', '')
                correct_answer = sample.get('correct_answer', '')
                distractor1 = sample.get('distractor1', '')
                distractor2 = sample.get('distractor2', '')
                distractor3 = sample.get('distractor3', '')
                support = sample.get('support', '')
                
                if question and correct_answer:
                    options = [correct_answer, distractor1, distractor2, distractor3]
                    options = [opt for opt in options if opt]  # Remove empty options
                    
                    question_text = f"{question}\nOptions: {', '.join(options)}"
                    if support:
                        question_text = f"Context: {support}\n\n{question_text}"
                    
                    self.training_data.append({
                        'user': question_text,
                        'response': f"The correct answer is: {correct_answer}",
                        'category': 'qa',
                        'source': source,
                        'type': 'science_question'
                    })
                    count += 1
                    
                if count >= 400:
                    break
            except Exception as e:
                continue
        return count

    def process_cosmos_qa(self, dataset, source):
        """Process Cosmos QA dataset"""
        count = 0
        for sample in dataset:
            try:
                context = sample.get('context', '')
                question = sample.get('question', '')
                answer0 = sample.get('answer0', '')
                answer1 = sample.get('answer1', '')
                answer2 = sample.get('answer2', '')
                answer3 = sample.get('answer3', '')
                label = sample.get('label', 0)
                
                if context and question:
                    answers = [answer0, answer1, answer2, answer3]
                    correct_answer = answers[label] if label < len(answers) else answer0
                    
                    self.training_data.append({
                        'user': f"Context: {context}\nQuestion: {question}\nA: {answer0}\nB: {answer1}\nC: {answer2}\nD: {answer3}",
                        'response': f"The correct answer is: {correct_answer}",
                        'category': 'reasoning',
                        'source': source,
                        'type': 'commonsense_qa'
                    })
                    count += 1
                    
                if count >= 300:
                    break
            except Exception as e:
                continue
        return count

    def process_dream(self, dataset, source):
        """Process DREAM dataset (dialogue reasoning)"""
        count = 0
        for sample in dataset:
            try:
                dialogue = sample.get('dialogue', [])
                question = sample.get('question', '')
                choice = sample.get('choice', [])
                answer = sample.get('answer', '')
                
                if dialogue and question and choice and answer:
                    dialogue_text = '\n'.join(dialogue) if isinstance(dialogue, list) else str(dialogue)
                    choices_text = '\n'.join([f"{chr(65+i)}: {choice[i]}" for i in range(len(choice))]) if isinstance(choice, list) else str(choice)
                    
                    self.training_data.append({
                        'user': f"Dialogue:\n{dialogue_text}\n\nQuestion: {question}\n{choices_text}",
                        'response': f"The correct answer is: {answer}",
                        'category': 'reasoning',
                        'source': source,
                        'type': 'dialogue_reasoning'
                    })
                    count += 1
                    
                if count >= 300:
                    break
            except Exception as e:
                continue
        return count

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
    print("🧠 LamAI - Incremental Training with Quality HuggingFace Datasets")
    print("🔄 Training 2 datasets at a time to preserve knowledge and prevent memory overload")
    print("=" * 80)
    
    # Train incrementally (2 datasets at a time)
    total_samples = trainer.train_incremental_batch(batch_size=2)
    
    print(f"\n🎉 Incremental training completed successfully!")
    print(f"📊 Total samples in knowledge base: {total_samples}")
    print(f"💾 Knowledge preserved with vector embeddings")
    print(f"🧠 LamAI is progressively getting smarter!")
    print(f"🔗 Test with: python lamai.py")
    print(f"🔄 Run again to continue training with remaining datasets")