#!/usr/bin/env python3
"""Simple test of SQLite embedding fix."""

import json
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_basic_serialization():
    """Test basic JSON serialization of embeddings."""
    print("🧪 Testing basic embedding serialization...")
    
    # Create test embedding
    embedding = np.random.rand(10).astype(np.float32)
    print(f"Original embedding: {embedding[:3]}...")
    
    # Serialize to JSON
    serialized = json.dumps(embedding.tolist())
    print(f"Serialized to JSON: {len(serialized)} chars")
    
    # Deserialize
    deserialized = np.asarray(json.loads(serialized), dtype=np.float32)
    print(f"Deserialized: {deserialized[:3]}...")
    
    # Check equality
    if np.allclose(embedding, deserialized):
        print("✅ Basic serialization test passed")
        return True
    else:
        print("❌ Basic serialization test failed")
        return False

if __name__ == "__main__":
    test_basic_serialization()
