#!/usr/bin/env python3
"""
Test data generator for RAGLite SQLite backend testing.
Creates a small SQLite database with sample documents for testing embeddings and semantic search.
"""

import sqlite3
import json
from pathlib import Path
import sys
import os

# Add src to path for raglite imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def create_test_database(db_path: str = "tests/test_raglite.db"):
    """Create a test SQLite database with sample documents."""

    # Sample documents for testing
    test_documents = [
        {
            "id": "doc1",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions. Supervised learning requires labeled training data, while unsupervised learning finds hidden patterns in unlabeled data.",
            "category": "AI"
        },
        {
            "id": "doc2",
            "title": "Natural Language Processing Basics",
            "content": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. Key techniques include tokenization, part-of-speech tagging, named entity recognition, and sentiment analysis. Modern NLP relies heavily on transformer architectures.",
            "category": "NLP"
        },
        {
            "id": "doc3",
            "title": "Vector Databases Explained",
            "content": "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They use similarity metrics like cosine similarity or Euclidean distance to find nearest neighbors. Popular vector databases include Pinecone, Weaviate, and Chroma, with SQLite-vec providing similar functionality for SQLite.",
            "category": "Databases"
        },
        {
            "id": "doc4",
            "title": "Retrieval-Augmented Generation",
            "content": "Retrieval-Augmented Generation (RAG) combines the strengths of retrieval systems and generative models. It first retrieves relevant documents from a knowledge base, then uses this information to generate more accurate and contextually relevant responses. This approach is particularly effective for question-answering and knowledge-intensive tasks.",
            "category": "AI"
        },
        {
            "id": "doc5",
            "title": "SQLite for Modern Applications",
            "content": "SQLite is a self-contained, serverless SQL database engine that's widely used in applications ranging from mobile apps to web browsers. Its key advantages include zero-configuration setup, cross-platform compatibility, and ACID compliance. Recent extensions like sqlite-vec enable vector search capabilities within SQLite.",
            "category": "Databases"
        }
    ]

    # Create database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables matching RAGLite schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            content TEXT,
            chunk_index INTEGER,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    ''')

    # Insert test documents
    for doc in test_documents:
        cursor.execute('''
            INSERT OR REPLACE INTO documents (id, title, content, category)
            VALUES (?, ?, ?, ?)
        ''', (doc['id'], doc['title'], doc['content'], doc['category']))

        # Create chunks (split content into sentences for testing)
        sentences = doc['content'].split('. ')
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                chunk_id = f"{doc['id']}_chunk_{i}"
                cursor.execute('''
                    INSERT OR REPLACE INTO chunks (id, document_id, content, chunk_index)
                    VALUES (?, ?, ?, ?)
                ''', (chunk_id, doc['id'], sentence.strip() + '.', i))

    conn.commit()
    conn.close()

    print(f"✅ Test database created at {db_path}")
    print(f"📊 Added {len(test_documents)} documents with chunks")

    return db_path

def verify_database(db_path: str):
    """Verify the test database contents."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check documents
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]

    # Check chunks
    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]

    print(f"📋 Database verification:")
    print(f"   - Documents: {doc_count}")
    print(f"   - Chunks: {chunk_count}")

    # Show sample data
    cursor.execute("SELECT id, title, category FROM documents LIMIT 3")
    sample_docs = cursor.fetchall()

    print("📖 Sample documents:")
    for doc in sample_docs:
        print(f"   - {doc[0]}: {doc[1]} ({doc[2]})")

    conn.close()

if __name__ == "__main__":
    db_path = "tests/test_raglite.db"
    create_test_database(db_path)
    verify_database(db_path)
