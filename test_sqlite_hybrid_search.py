#!/usr/bin/env python3
"""Test SQLite hybrid search without embedder dependencies."""

import sys
from pathlib import Path
import os
import tempfile
import numpy as np

# Add src to path for raglite imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_hybrid_search_with_mock_embedder():
    """Test hybrid search by mocking the embedding process."""
    print("🧪 Testing SQLite Hybrid Search with Mock Embedder...")
    
    # Set environment to disable external dependencies
    os.environ["RAGLITE_DISABLE_GPU"] = "1"
    os.environ["OPENAI_API_KEY"] = "mock-key-for-testing"
    
    try:
        import unittest.mock
        
        # Mock all the heavy dependencies
        with unittest.mock.patch('raglite._litellm.get_embedding_dim', return_value=1536), \
             unittest.mock.patch('raglite._embed.embed_strings') as mock_embed:
            
            # Mock the embedding function to return fake embeddings
            def fake_embed(strings, config=None):
                """Return fake embeddings for testing."""
                return np.random.rand(len(strings), 1536).astype(np.float32)
            
            mock_embed.side_effect = fake_embed
            
            from raglite._config import RAGLiteConfig
            from raglite._database import create_database_engine
            from raglite._search import hybrid_search, keyword_search, vector_search
            from sqlmodel import Session, text
            
            # Create a temporary database
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                db_path = tmp_db.name
            
            try:
                # Configure for SQLite
                config = RAGLiteConfig(
                    db_url=f'sqlite:///{db_path}',
                    embedder='openai/text-embedding-ada-002'
                )
                
                # Create engine and tables
                engine = create_database_engine(config)
                print(f"✅ SQLite engine created")
                
                # Create test data
                with Session(engine) as session:
                    # Insert test document
                    session.execute(text("""
                        INSERT INTO document (id, filename, url, metadata)
                        VALUES ('test-doc-1', 'test.txt', NULL, '{}')
                    """))
                    
                    # Insert test chunk
                    session.execute(text("""
                        INSERT INTO chunk (id, document_id, "index", headings, body, metadata)
                        VALUES ('test-chunk-1', 'test-doc-1', 0, 'Test Heading', 
                               'This is a test chunk about machine learning and artificial intelligence.', '{}')
                    """))
                    
                    # Insert fake embedding
                    fake_embedding = np.random.rand(1536).astype(np.float32)
                    embedding_bytes = fake_embedding.tobytes()
                    session.execute(text("""
                        INSERT INTO chunk_embedding (chunk_id, embedding)
                        VALUES ('test-chunk-1', ?)
                    """), [embedding_bytes])
                    
                    # Insert into FTS5
                    session.execute(text("""
                        INSERT INTO chunk_fts (chunk_id, body)
                        VALUES ('test-chunk-1', 'This is a test chunk about machine learning and artificial intelligence.')
                    """))
                    
                    session.commit()
                    print("✅ Test data created with embeddings")
                
                # Test individual searches first
                print("\n🔍 Testing keyword search...")
                ks_chunk_ids, ks_scores = keyword_search("machine learning", num_results=5, config=config)
                print(f"   Keyword search found {len(ks_chunk_ids)} results")
                
                print("\n🔍 Testing vector search...")
                vs_chunk_ids, vs_scores = vector_search("machine learning", num_results=5, config=config) 
                print(f"   Vector search found {len(vs_chunk_ids)} results")
                
                # Test hybrid search
                print("\n🔍 Testing hybrid search...")
                chunk_ids, scores = hybrid_search("machine learning", num_results=5, config=config)
                print(f"✅ Hybrid search executed successfully")
                print(f"   Found {len(chunk_ids)} results")
                if chunk_ids:
                    print(f"   First result: {chunk_ids[0]} (score: {scores[0]})")
                
                return len(chunk_ids) > 0
                
            finally:
                # Cleanup
                try:
                    Path(db_path).unlink()
                except Exception:
                    pass
                    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the hybrid search test."""
    print("🚀 Testing SQLite Hybrid Search Implementation\n")
    
    success = test_hybrid_search_with_mock_embedder()
    
    if success:
        print("\n🎉 SQLite hybrid search test passed!")
    else:
        print("\n⚠️  SQLite hybrid search test failed")

if __name__ == "__main__":
    main()