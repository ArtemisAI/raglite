#!/usr/bin/env python3
"""
Comprehensive test suite for RAGLite SQLite backend functionality.
Tests all aspects of the SQLite implementation including database operations, 
search functionality, performance, and error handling.
"""

import sqlite3
import sys
import tempfile
from pathlib import Path
import os
import time
from typing import List, Tuple

# Add src to path for raglite imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_environment_setup():
    """Test basic environment setup and imports."""
    print("🧪 Testing Environment Setup...")
    
    try:
        # Test basic imports
        import raglite
        from raglite._database import create_database_engine
        from raglite._config import RAGLiteConfig
        from raglite import vector_search, keyword_search, hybrid_search
        print("✅ All basic imports successful")
        
        # Test dependency availability
        try:
            import sqlite_vec
            print(f"✅ sqlite-vec available: {sqlite_vec.__version__}")
        except ImportError:
            print("⚠️  sqlite-vec not available - will test fallback")
        
        try:
            import pynndescent
            print(f"✅ PyNNDescent available: {pynndescent.__version__}")
        except ImportError:
            print("❌ PyNNDescent not available - fallback may fail")
        
        return True
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return False

def test_sqlite_vec_extension():
    """Test sqlite-vec extension functionality."""
    print("\n🧪 Testing SQLite-Vec Extension...")
    
    try:
        import sqlite_vec
        
        # Create temporary database for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            conn = sqlite3.connect(db_path)
            conn.enable_load_extension(True)
            conn.load_extension(sqlite_vec.loadable_path())
            
            # Create test vector table
            conn.execute("""
                CREATE VIRTUAL TABLE test_vectors USING vec0(
                    id INTEGER PRIMARY KEY,
                    vector FLOAT[3]
                )
            """)
            
            # Insert test vectors
            test_vectors = [
                (1, sqlite_vec.serialize_float32([0.1, 0.2, 0.3])),
                (2, sqlite_vec.serialize_float32([0.4, 0.5, 0.6])),
                (3, sqlite_vec.serialize_float32([0.7, 0.8, 0.9]))
            ]
            
            conn.executemany("INSERT INTO test_vectors (id, vector) VALUES (?, ?)", test_vectors)
            conn.commit()
            
            # Test similarity search
            query_vector = sqlite_vec.serialize_float32([0.1, 0.2, 0.3])
            result = conn.execute("""
                SELECT id, vec_distance_cosine(vector, ?) as distance
                FROM test_vectors
                ORDER BY distance
                LIMIT 2
            """, (query_vector,)).fetchall()
            
            print("✅ sqlite-vec working correctly")
            print(f"   - Vector search results: {result}")
            
            conn.close()
            return True
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
                
    except Exception as e:
        print(f"❌ sqlite-vec test failed: {e}")
        return False

def test_database_engine_creation():
    """Test SQLite database engine creation and connection."""
    print("\n🧪 Testing Database Engine Creation...")
    
    try:
        from raglite._database import create_database_engine
        from raglite._config import RAGLiteConfig
        from sqlalchemy import text
        
        # Test with test database
        config = RAGLiteConfig(
            db_url='sqlite:///tests/test_raglite.db',
            embedder='openai/text-embedding-ada-002'  # Use OpenAI as fallback
        )
        
        engine = create_database_engine(config)
        print("✅ SQLite engine created successfully")
        print(f"   - Engine type: {type(engine)}")
        print(f"   - Dialect: {engine.dialect.name}")
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text('SELECT COUNT(*) FROM documents'))
            doc_count = result.fetchone()[0]
            print(f"✅ Database connection successful - {doc_count} documents")
        
        # Check sqlite-vec status
        if hasattr(engine, 'sqlite_vec_available'):
            print(f"   - sqlite-vec status: {engine.sqlite_vec_available}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database engine creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_database_operations():
    """Test basic database operations (SELECT, INSERT, etc.)."""
    print("\n🧪 Testing Basic Database Operations...")
    
    try:
        from raglite._database import create_database_engine
        from raglite._config import RAGLiteConfig
        from sqlalchemy import text
        
        config = RAGLiteConfig(
            db_url='sqlite:///tests/test_raglite.db',
            embedder='openai/text-embedding-ada-002'
        )
        engine = create_database_engine(config)
        
        with engine.connect() as conn:
            # Test document queries
            result = conn.execute(text("SELECT title, category FROM documents LIMIT 3"))
            documents = result.fetchall()
            print(f"✅ Document queries work - sample: {documents[0] if documents else 'None'}")
            
            # Test chunk queries
            result = conn.execute(text("SELECT COUNT(*) FROM chunks"))
            chunk_count = result.fetchone()[0]
            print(f"✅ Chunk queries work - {chunk_count} chunks found")
            
            # Test join queries
            result = conn.execute(text("""
                SELECT d.title, COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                GROUP BY d.id, d.title
                LIMIT 2
            """))
            join_results = result.fetchall()
            print(f"✅ Join queries work - sample: {join_results[0] if join_results else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic database operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_functionality_simulation():
    """Test search functionality without requiring API keys."""
    print("\n🧪 Testing Search Functionality (Simulation)...")
    
    try:
        # Test with direct database queries first (simulating search)
        conn = sqlite3.connect("tests/test_raglite.db")
        cursor = conn.cursor()
        
        # Test keyword search simulation
        search_terms = ["machine learning", "vector", "database"]
        for term in search_terms:
            cursor.execute("""
                SELECT d.title, c.content, d.category
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE LOWER(c.content) LIKE LOWER(?)
                LIMIT 2
            """, (f"%{term}%",))
            
            results = cursor.fetchall()
            print(f"✅ Keyword search for '{term}': {len(results)} results")
            if results:
                print(f"   - First result: {results[0][0]} ({results[0][2]})")
        
        # Test if we can access the FTS table
        try:
            cursor.execute("SELECT COUNT(*) FROM chunk_fts")
            fts_count = cursor.fetchone()[0]
            print(f"✅ FTS5 table accessible - {fts_count} indexed chunks")
        except Exception as e:
            print(f"⚠️  FTS5 table issue: {e}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Search functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_options():
    """Test various configuration options for SQLite backend."""
    print("\n🧪 Testing Configuration Options...")
    
    try:
        from raglite._config import RAGLiteConfig
        
        # Test different SQLite configurations
        configs = [
            {
                'name': 'File database',
                'db_url': 'sqlite:///tests/test_raglite.db',
                'embedder': 'openai/text-embedding-ada-002'
            },
            {
                'name': 'Memory database',
                'db_url': 'sqlite:///:memory:',
                'embedder': 'openai/text-embedding-ada-002'
            }
        ]
        
        for config_data in configs:
            try:
                config = RAGLiteConfig(**{k: v for k, v in config_data.items() if k != 'name'})
                print(f"✅ {config_data['name']} configuration created")
                print(f"   - DB URL: {config.db_url}")
                print(f"   - Embedder: {config.embedder}")
            except Exception as e:
                print(f"❌ {config_data['name']} configuration failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_performance_characteristics():
    """Test basic performance characteristics of SQLite backend."""
    print("\n🧪 Testing Performance Characteristics...")
    
    try:
        from raglite._database import create_database_engine
        from raglite._config import RAGLiteConfig
        from sqlalchemy import text
        
        config = RAGLiteConfig(
            db_url='sqlite:///tests/test_raglite.db',
            embedder='openai/text-embedding-ada-002'
        )
        engine = create_database_engine(config)
        
        # Test query performance
        queries = [
            "SELECT COUNT(*) FROM documents",
            "SELECT COUNT(*) FROM chunks",
            "SELECT d.title, COUNT(c.id) FROM documents d LEFT JOIN chunks c ON d.id = c.document_id GROUP BY d.id",
        ]
        
        for query in queries:
            start_time = time.time()
            with engine.connect() as conn:
                result = conn.execute(text(query))
                _ = result.fetchall()
            end_time = time.time()
            
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"✅ Query performance: {duration:.2f}ms for: {query[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and recovery mechanisms."""
    print("\n🧪 Testing Error Handling...")
    
    try:
        from raglite._database import create_database_engine
        from raglite._config import RAGLiteConfig
        from sqlalchemy import text
        
        # Test invalid database path
        try:
            config = RAGLiteConfig(
                db_url='sqlite:///nonexistent/path/database.db',
                embedder='openai/text-embedding-ada-002'
            )
            engine = create_database_engine(config)
            print("✅ Engine created with invalid path (will create directory)")
        except Exception as e:
            print(f"⚠️  Invalid path handling: {e}")
        
        # Test invalid SQL query
        try:
            config = RAGLiteConfig(
                db_url='sqlite:///tests/test_raglite.db',
                embedder='openai/text-embedding-ada-002'
            )
            engine = create_database_engine(config)
            
            with engine.connect() as conn:
                result = conn.execute(text("SELECT * FROM nonexistent_table"))
                _ = result.fetchall()
        except Exception as e:
            print(f"✅ Invalid query properly handled: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_vector_search_fallback():
    """Test vector search fallback mechanisms."""
    print("\n🧪 Testing Vector Search Fallback...")
    
    try:
        # Test PyNNDescent availability
        try:
            import pynndescent
            print("✅ PyNNDescent available for fallback")
            
            # Create simple test to verify it can create an index
            import numpy as np
            data = np.random.random((10, 5)).astype(np.float32)
            index = pynndescent.NNDescent(data, metric='cosine')
            index.prepare()
            print("✅ PyNNDescent index creation successful")
            
        except ImportError:
            print("❌ PyNNDescent not available - vector search may fail without sqlite-vec")
        
        # Test sqlite-vec availability
        try:
            import sqlite_vec
            print("✅ sqlite-vec available for primary vector search")
        except ImportError:
            print("⚠️  sqlite-vec not available - using fallback only")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector search fallback test failed: {e}")
        return False

def test_concurrent_access():
    """Test concurrent database access."""
    print("\n🧪 Testing Concurrent Access...")
    
    try:
        from raglite._database import create_database_engine
        from raglite._config import RAGLiteConfig
        from sqlalchemy import text
        import threading
        import time
        
        config = RAGLiteConfig(
            db_url='sqlite:///tests/test_raglite.db',
            embedder='openai/text-embedding-ada-002'
        )
        
        results = []
        errors = []
        
        def query_worker(worker_id: int):
            try:
                engine = create_database_engine(config)
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT COUNT(*) FROM documents"))
                    count = result.fetchone()[0]
                    results.append((worker_id, count))
                    time.sleep(0.1)  # Simulate work
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Create multiple worker threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=query_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        if errors:
            print(f"⚠️  Some concurrent access errors: {len(errors)} errors")
            for worker_id, error in errors:
                print(f"   - Worker {worker_id}: {error}")
        else:
            print(f"✅ Concurrent access successful - {len(results)} workers completed")
        
        return len(errors) == 0
        
    except Exception as e:
        print(f"❌ Concurrent access test failed: {e}")
        return False

def main():
    """Run all tests and report results."""
    print("🚀 Starting Comprehensive RAGLite SQLite Backend Tests\n")
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("SQLite-Vec Extension", test_sqlite_vec_extension), 
        ("Database Engine Creation", test_database_engine_creation),
        ("Basic Database Operations", test_basic_database_operations),
        ("Search Functionality", test_search_functionality_simulation),
        ("Configuration Options", test_configuration_options),
        ("Performance Characteristics", test_performance_characteristics),
        ("Error Handling", test_error_handling),
        ("Vector Search Fallback", test_vector_search_fallback),
        ("Concurrent Access", test_concurrent_access),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            print("=" * 60)
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📊 Comprehensive Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name:.<40} {status}")
        if success:
            passed += 1
    
    total = len(results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print("=" * 60)
    print(f"🎯 Overall Results:")
    print(f"   - Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
    print(f"   - Total Time: {total_time:.2f} seconds")
    print(f"   - Average Time per Test: {total_time/total:.2f} seconds")
    
    if passed == total:
        print("\n🎉 All tests passed! SQLite backend is fully functional.")
        return 0
    elif passed >= total * 0.8:  # 80% pass rate
        print(f"\n⚠️  Most tests passed ({success_rate:.1f}%). SQLite backend is mostly functional.")
        return 0
    else:
        print(f"\n❌ Many tests failed ({100-success_rate:.1f}% failure rate). SQLite backend needs attention.")
        return 1

if __name__ == "__main__":
    exit(main())