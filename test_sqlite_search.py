#!/usr/bin/env python3
"""Test SQLite search functionality implementation."""

import sys
from pathlib import Path
import numpy as np
import tempfile
import os

# Add src to path for raglite imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_sqlite_search_functions():
    """Test the SQLite search functions with a minimal setup."""
    print("🧪 Testing SQLite Search Functions...")
    
    try:
        # Set environment to avoid GPU dependencies
        os.environ["RAGLITE_DISABLE_GPU"] = "1"
        
        from raglite._config import RAGLiteConfig
        from raglite._database import create_database_engine, SQLModel, Chunk, ChunkEmbedding
        from raglite._search import vector_search, keyword_search, hybrid_search
        from sqlmodel import Session
        
        # Create a temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            # Configure for SQLite
            config = RAGLiteConfig(
                db_url=f'sqlite:///{db_path}',
                embedder='openai/text-embedding-ada-002'  # Mock embedder that should work
            )
            
            # Create engine and tables
            engine = create_database_engine(config)
            print(f"✅ SQLite engine created, sqlite-vec available: {getattr(engine, 'sqlite_vec_available', False)}")
            
            # Create test data
            with Session(engine) as session:
                # Insert a test document and chunk
                from raglite._database import Document
                
                doc = Document(
                    id="test-doc-1",
                    filename="test.txt",
                    metadata_={"test": True}
                )
                session.add(doc)
                
                chunk = Chunk(
                    id="test-chunk-1",
                    document_id="test-doc-1",
                    index=0,
                    headings="Test Heading",
                    body="This is a test chunk about machine learning and artificial intelligence.",
                    metadata_={"test": True}
                )
                session.add(chunk)
                
                # Add a test embedding (fake embedding for testing)
                test_embedding = np.random.rand(1536).astype(np.float32)  # OpenAI ada-002 dimension
                chunk_embedding = ChunkEmbedding(
                    chunk_id="test-chunk-1",
                    embedding=test_embedding
                )
                session.add(chunk_embedding)
                
                session.commit()
                print("✅ Test data inserted")
            
            # Test keyword search
            print("\n🔍 Testing keyword search...")
            try:
                chunk_ids, scores = keyword_search("machine learning", num_results=5, config=config)
                if chunk_ids:
                    print(f"✅ Keyword search found {len(chunk_ids)} results")
                    print(f"   First result: {chunk_ids[0]} (score: {scores[0]})")
                else:
                    print("⚠️  Keyword search returned no results")
            except Exception as e:
                print(f"❌ Keyword search failed: {e}")
            
            # Test vector search
            print("\n🔍 Testing vector search...")
            try:
                # Use the same test embedding as query
                chunk_ids, scores = vector_search(test_embedding, num_results=5, config=config)
                if chunk_ids:
                    print(f"✅ Vector search found {len(chunk_ids)} results")
                    print(f"   First result: {chunk_ids[0]} (score: {scores[0]})")
                else:
                    print("⚠️  Vector search returned no results")
            except Exception as e:
                print(f"❌ Vector search failed: {e}")
            
            # Test hybrid search
            print("\n🔍 Testing hybrid search...")
            try:
                chunk_ids, scores = hybrid_search("machine learning", num_results=5, config=config)
                if chunk_ids:
                    print(f"✅ Hybrid search found {len(chunk_ids)} results")
                    print(f"   First result: {chunk_ids[0]} (score: {scores[0]})")
                else:
                    print("⚠️  Hybrid search returned no results")
            except Exception as e:
                print(f"❌ Hybrid search failed: {e}")
                
        finally:
            # Cleanup
            try:
                Path(db_path).unlink()
            except Exception:
                pass
        
        return True
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

def main():
    """Run the SQLite search tests."""
    print("🚀 Testing SQLite Search Implementation\n")
    
    success = test_sqlite_search_functions()
    
    if success:
        print("\n🎉 SQLite search tests completed!")
    else:
        print("\n⚠️  SQLite search tests encountered issues")

if __name__ == "__main__":
    main()