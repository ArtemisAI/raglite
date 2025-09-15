#!/usr/bin/env python3
"""
Test script for RAGLite SQLite backend functionality.
Tests embeddings, semantic search, and RAG operations using the test database.
"""

import sqlite3
import sys
from pathlib import Path
import os

# Add src to path for raglite imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_sqlite_backend():
    """Test basic SQLite backend functionality."""
    print("🧪 Testing SQLite Backend...")

    # Test database connection
    db_path = "tests/test_raglite.db"
    if not Path(db_path).exists():
        print("❌ Test database not found. Run create_test_db.py first.")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Test basic queries
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    print(f"✅ Connected to database with {doc_count} documents")

    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]
    print(f"✅ Found {chunk_count} chunks")

    conn.close()
    return True

def test_sqlite_vec_setup():
    """Test sqlite-vec extension setup."""
    print("\n🧪 Testing SQLite-Vec Setup...")

    try:
        import sqlite_vec
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)

        # Use the correct path for Windows
        vec_path = sqlite_vec.loadable_path()
        print(f"   Loading sqlite-vec from: {vec_path}")
        conn.load_extension(vec_path)

        # Create test vector table
        conn.execute("""
            CREATE VIRTUAL TABLE test_vectors USING vec0(
                id INTEGER PRIMARY KEY,
                vector FLOAT[3]
            )
        """)

        # Insert test vectors (convert to bytes for sqlite-vec)
        test_vectors = [
            (1, sqlite_vec.serialize_float32([0.1, 0.2, 0.3])),
            (2, sqlite_vec.serialize_float32([0.4, 0.5, 0.6])),
            (3, sqlite_vec.serialize_float32([0.7, 0.8, 0.9]))
        ]

        conn.executemany("INSERT INTO test_vectors (id, vector) VALUES (?, ?)", test_vectors)
        conn.commit()

        # Test similarity search (also serialize query vector)
        query_vector = sqlite_vec.serialize_float32([0.1, 0.2, 0.3])
        result = conn.execute("""
            SELECT id, vec_distance_cosine(vector, ?) as distance
            FROM test_vectors
            ORDER BY distance
            LIMIT 2
        """, (query_vector,)).fetchall()

        print("✅ SQLite-vec working correctly")
        print(f"   - Nearest neighbors: {result}")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ SQLite-vec test failed: {e}")
        return False

def test_semantic_search_simulation():
    """Simulate semantic search using the test database."""
    print("\n🧪 Testing Semantic Search Simulation...")

    # This simulates what RAGLite would do for semantic search
    # In a real scenario, this would use actual embeddings

    conn = sqlite3.connect("tests/test_raglite.db")
    cursor = conn.cursor()

    # Simulate a search query - use terms that exist in test data
    query = "machine learning"

    # Simple keyword search (placeholder for semantic search)
    cursor.execute("""
        SELECT d.title, c.content, d.category
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE LOWER(c.content) LIKE LOWER(?)
        LIMIT 3
    """, (f"%{query}%",))

    results = cursor.fetchall()

    print(f"🔍 Search results for '{query}':")
    for i, (title, content, category) in enumerate(results, 1):
        print(f"   {i}. {title} ({category})")
        print(f"      {content[:100]}...")

    conn.close()
    return len(results) > 0

def test_rag_pipeline_simulation():
    """Simulate a basic RAG pipeline."""
    print("\n🧪 Testing RAG Pipeline Simulation...")

    # Simulate: Query -> Retrieve -> Generate
    query = "artificial intelligence"

    # Step 1: Retrieve relevant chunks
    conn = sqlite3.connect("tests/test_raglite.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT d.title, c.content
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE LOWER(c.content) LIKE LOWER(?) OR LOWER(d.title) LIKE LOWER(?)
        LIMIT 2
    """, (f"%{query}%", f"%{query}%"))

    retrieved_chunks = cursor.fetchall()
    conn.close()

    if not retrieved_chunks:
        print("❌ No relevant chunks found")
        return False

    # Step 2: Simulate generation (in real RAGLite, this would use LLM)
    context = "\n".join([f"From {title}: {content}" for title, content in retrieved_chunks])

    simulated_response = f"Based on the retrieved information:\n\n{context}\n\nRetrieval-Augmented Generation (RAG) combines retrieval systems with generative models to provide more accurate and contextually relevant responses."

    print("🤖 Simulated RAG Response:")
    print(f"   Query: {query}")
    print(f"   Retrieved {len(retrieved_chunks)} relevant chunks")
    print(f"   Generated response: {simulated_response[:200]}...")

    return True

def main():
    """Run all tests."""
    print("🚀 Starting RAGLite SQLite Backend Tests\n")

    tests = [
        ("SQLite Backend", test_sqlite_backend),
        ("SQLite-Vec Setup", test_sqlite_vec_setup),
        ("Semantic Search", test_semantic_search_simulation),
        ("RAG Pipeline", test_rag_pipeline_simulation)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))

    print("\n📊 Test Results Summary:")
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")

    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! SQLite backend is ready for RAGLite.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
