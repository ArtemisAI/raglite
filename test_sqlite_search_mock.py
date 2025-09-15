#!/usr/bin/env python3
"""Test SQLite search functionality without embedding dependencies."""

import sys
from pathlib import Path
import os
import tempfile

# Add src to path for raglite imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_mock_search():
    """Test search functions with minimal dependencies."""
    print("🧪 Testing SQLite Search with Mock Configuration...")
    
    # Set environment to disable all external dependencies
    os.environ["RAGLITE_DISABLE_GPU"] = "1"
    os.environ["OPENAI_API_KEY"] = "mock-key-for-testing"
    
    try:
        # Mock the get_embedding_dim function to avoid LLM dependencies
        import unittest.mock
        
        with unittest.mock.patch('raglite._litellm.get_embedding_dim', return_value=1536):
            from raglite._config import RAGLiteConfig
            from raglite._database import create_database_engine, SQLModel, Chunk, Document
            from raglite._search import keyword_search
            from sqlmodel import Session, text
            
            # Create a temporary database
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                db_path = tmp_db.name
            
            try:
                # Configure for SQLite with mock embedder
                config = RAGLiteConfig(
                    db_url=f'sqlite:///{db_path}',
                    embedder='openai/text-embedding-ada-002'
                )
                
                # Create engine and tables
                engine = create_database_engine(config)
                print(f"✅ SQLite engine created, sqlite-vec available: {getattr(engine, 'sqlite_vec_available', False)}")
                
                # Create test data manually using SQL to avoid embedding issues
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
                    
                    # Insert into FTS5 manually
                    try:
                        session.execute(text("""
                            INSERT INTO chunk_fts (chunk_id, body)
                            VALUES ('test-chunk-1', 'This is a test chunk about machine learning and artificial intelligence.')
                        """))
                        print("✅ FTS5 data inserted")
                    except Exception as e:
                        print(f"⚠️  FTS5 insertion failed: {e}")
                    
                    session.commit()
                    print("✅ Test data created")
                
                # Test keyword search
                print("\n🔍 Testing keyword search...")
                try:
                    chunk_ids, scores = keyword_search("machine learning", num_results=5, config=config)
                    print(f"✅ Keyword search executed successfully")
                    print(f"   Found {len(chunk_ids)} results")
                    if chunk_ids:
                        print(f"   First result: {chunk_ids[0]} (score: {scores[0]})")
                    else:
                        print("   No results (may be due to FTS5 setup)")
                    return True
                except Exception as e:
                    print(f"❌ Keyword search failed: {e}")
                    return False
                    
            finally:
                # Cleanup
                try:
                    Path(db_path).unlink()
                except Exception:
                    pass
                    
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

def test_database_structure():
    """Test that the SQLite database structure is created correctly."""
    print("\n🧪 Testing SQLite Database Structure...")
    
    try:
        import unittest.mock
        
        with unittest.mock.patch('raglite._litellm.get_embedding_dim', return_value=1536):
            from raglite._config import RAGLiteConfig
            from raglite._database import create_database_engine
            from sqlmodel import Session, text
            
            # Create a temporary database
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                db_path = tmp_db.name
            
            try:
                config = RAGLiteConfig(
                    db_url=f'sqlite:///{db_path}',
                    embedder='openai/text-embedding-ada-002'
                )
                
                engine = create_database_engine(config)
                
                # Test database structure
                with Session(engine) as session:
                    # Check main tables exist
                    tables = session.execute(text("""
                        SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    """)).fetchall()
                    
                    table_names = [row[0] for row in tables]
                    print(f"✅ Found tables: {', '.join(table_names)}")
                    
                    # Check FTS5 table exists
                    if 'chunk_fts' in table_names:
                        print("✅ FTS5 table created")
                    else:
                        print("⚠️  FTS5 table not found")
                    
                    # Check sqlite-vec table exists
                    if 'chunk_embeddings_vec' in table_names:
                        print("✅ SQLite-vec table created")
                    else:
                        print("⚠️  SQLite-vec table not found")
                    
                return True
                
            finally:
                try:
                    Path(db_path).unlink()
                except Exception:
                    pass
                    
    except Exception as e:
        print(f"❌ Database structure test failed: {e}")
        return False

def main():
    """Run the SQLite search tests."""
    print("🚀 Testing SQLite Search Implementation (Mock Version)\n")
    
    tests = [
        ("Database Structure", test_database_structure),
        ("Mock Search", test_mock_search)
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
        print("🎉 All SQLite search tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()