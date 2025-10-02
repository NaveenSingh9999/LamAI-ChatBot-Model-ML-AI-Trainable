#!/usr/bin/env python3
"""
Debug script to check LamAI's search functionality
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def debug_search():
    """Debug the search functionality"""
    
    # Load the massive knowledge base
    print("🔍 Loading massive knowledge base for debugging...")
    with open('massive_smart_kb.json', 'r') as f:
        data = json.load(f)
    
    training_data = data.get('training_data', [])
    embeddings_data = data.get('embeddings', [])
    
    print(f"📊 Training data entries: {len(training_data)}")
    print(f"📊 Embeddings count: {len(embeddings_data)}")
    
    if len(training_data) != len(embeddings_data):
        print("❌ Mismatch between training data and embeddings!")
        return
    
    # Check some sample data
    print("\n📋 Sample training data:")
    for i in range(min(5, len(training_data))):
        item = training_data[i]
        print(f"  {i}: Q: {item.get('user', '')[:50]}...")
        print(f"     A: {item.get('response', '')[:50]}...")
        print(f"     Category: {item.get('category', 'unknown')}")
        print()
    
    # Test search functionality
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = np.array(embeddings_data)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    
    # Test queries
    test_queries = [
        "what is a computer",
        "5+5",
        "who are you",
        "artificial intelligence"
    ]
    
    print("🔍 Testing search functionality:")
    for query in test_queries:
        print(f"\n🔎 Query: {query}")
        
        query_embedding = model.encode([query]).astype('float32')
        scores, indices = index.search(query_embedding, 3)
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(training_data):
                item = training_data[idx]
                print(f"  {i+1}. Score: {score:.3f}")
                print(f"     Q: {item.get('user', '')}")
                print(f"     A: {item.get('response', '')}")
                print(f"     Category: {item.get('category', 'unknown')}")
                print()

if __name__ == "__main__":
    debug_search()