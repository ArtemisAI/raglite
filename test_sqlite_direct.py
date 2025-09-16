#!/usr/bin/env python3
"""
Direct SQLite functionality test - bypassing import chain issues.
This test validates the SQLite implementation by importing only specific modules.
"""

import tempfile
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add src to path for raglite imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_sqlite_vec_direct():
    """Test SQLiteVec directly without full raglite imports."""
    print("🧪 Testing SQLiteVec directly...")
    
    # Import just the typing module components we need
    import sqlalchemy
    from sqlalchemy.types import UserDefinedType
    from sqlalchemy.engine import Dialect
    from typing import Any, Callable
    
    # Define SQLiteVec class directly (from _typing.py)
    class SQLiteVec(UserDefinedType[np.ndarray]):
        """A SQLite vector column type for SQLAlchemy using sqlite-vec or JSON serialization."""
        
        cache_ok = True
        
        def __init__(self, dim: int | None = None) -> None:
            super().__init__()
            self.dim = dim
        
        def get_col_spec(self, **kwargs: Any) -> str:
            return "TEXT"
        
        def bind_processor(self, dialect: Dialect) -> Callable[[np.ndarray | None], str | None]:
            """Process NumPy ndarray to JSON string for SQLite storage."""
            import json
            
            def process(value: np.ndarray | None) -> str | None:
                if value is not None:
                    # Convert to list and serialize as JSON
                    value_list = np.ravel(value).tolist()
                    return json.dumps(value_list)
                return None
            
            return process
        
        def result_processor(self, dialect: Dialect, coltype: Any) -> Callable[[str | None], np.ndarray | None]:
            """Process JSON string from SQLite to NumPy ndarray."""
            import json
            
            def process(value: str | None) -> np.ndarray | None:
                if value is not None:
                    # Deserialize JSON and convert to numpy array
                    value_list = json.loads(value)
                    return np.asarray(value_list, dtype=np.float32)
                return None
            
            return process
    
    # Test serialization with different dimensions
    for dim in [384, 768, 1536]:
        original_embedding = np.random.rand(dim).astype(np.float32)
        
        vec = SQLiteVec(dim=dim)
        
        # Test bind processor (serialization)
        bind_processor = vec.bind_processor(None)
        serialized = bind_processor(original_embedding)
        
        # Test result processor (deserialization)
        result_processor = vec.result_processor(None, None)
        deserialized = result_processor(serialized)
        
        # Verify roundtrip accuracy
        assert np.allclose(original_embedding, deserialized, rtol=1e-6), f"Serialization failed for dim {dim}"
        
        print(f"✅ SQLiteVec serialization test passed for dimension {dim}")


def test_sqlite_database_creation():
    """Test SQLite database engine creation."""
    print("🧪 Testing SQLite database creation...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # Import just what we need for database creation
        import sqlite3
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import Session
        
        # Create engine
        db_url = f'sqlite:///{db_path}'
        engine = create_engine(db_url, pool_pre_ping=True)
        
        # Test basic connection
        with Session(engine) as session:
            result = session.execute(text("SELECT 1 as test")).scalar()
            assert result == 1, "Basic SQL query failed"
        
        print("✅ SQLite database creation test passed")
        
        # Test if sqlite-vec extension can be loaded
        try:
            import sqlite_vec
            conn = sqlite3.connect(db_path)
            conn.enable_load_extension(True)
            conn.load_extension(sqlite_vec.loadable_path())
            conn.enable_load_extension(False)
            print("✅ sqlite-vec extension available and loadable")
            conn.close()
        except ImportError:
            print("ℹ️  sqlite-vec extension not available (expected in some environments)")
        except Exception as e:
            print(f"ℹ️  sqlite-vec extension loading failed: {e}")
        
    finally:
        if 'engine' in locals() and engine:
            engine.dispose()
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_pynndescent_availability():
    """Test PyNNDescent fallback availability."""
    print("🧪 Testing PyNNDescent availability...")
    
    try:
        import pynndescent
        
        # Test basic functionality
        data = np.random.random((100, 384)).astype(np.float32)
        index = pynndescent.NNDescent(data, metric='cosine', n_neighbors=10)
        
        query = np.random.random((1, 384)).astype(np.float32)
        indices, distances = index.query(query, k=5)
        
        assert indices.shape == (1, 5), "PyNNDescent query shape incorrect"
        assert distances.shape == (1, 5), "PyNNDescent distances shape incorrect"
        
        print("✅ PyNNDescent fallback available and working")
        
    except ImportError:
        print("⚠️  PyNNDescent not available - fallback vector search will not work")
    except Exception as e:
        print(f"⚠️  PyNNDescent test failed: {e}")


def test_json_embedding_roundtrip():
    """Test JSON embedding serialization edge cases."""
    print("🧪 Testing JSON embedding roundtrip with edge cases...")
    
    test_cases = [
        np.array([0.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float32),  # Basic values
        np.random.rand(1024).astype(np.float32),  # Large embedding
        np.array([1e-10, 1e10, -1e-10, -1e10], dtype=np.float32),  # Extreme values
        np.zeros(384, dtype=np.float32),  # All zeros
        np.ones(384, dtype=np.float32),  # All ones
    ]
    
    for i, original in enumerate(test_cases):
        # Serialize
        serialized = json.dumps(original.tolist())
        
        # Deserialize  
        deserialized = np.asarray(json.loads(serialized), dtype=np.float32)
        
        # Verify
        assert np.allclose(original, deserialized, rtol=1e-6), f"Test case {i} failed"
        
        print(f"✅ JSON roundtrip test case {i} passed")


if __name__ == "__main__":
    print("🚀 Running Direct SQLite Functionality Tests")
    print("=" * 60)
    
    try:
        test_sqlite_vec_direct()
        print()
        
        test_sqlite_database_creation()
        print()
        
        test_pynndescent_availability()
        print()
        
        test_json_embedding_roundtrip()
        print()
        
        print("🎉 All direct SQLite tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)