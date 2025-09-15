#!/usr/bin/env python3
"""Simple SQLite verification without embeddings."""

import os
os.environ["RAGLITE_DISABLE_GPU"] = "1"  # Disable GPU to avoid llama-cpp-python

# Mock the embedder config to use a simple mock
from raglite._config import RAGLiteConfig
config = RAGLiteConfig(
    db_url='sqlite:///tests/test_raglite.db',
    embedder='mock/test'  # Use mock embedder to avoid dependencies
)

try:
    from raglite._database import create_database_engine
    engine = create_database_engine(config)
    print('✅ SQLite engine created successfully')
    print(f'sqlite-vec available: {getattr(engine, "sqlite_vec_available", False)}')
    print('✅ Basic SQLite functionality working')
except Exception as e:
    print(f'❌ Error: {e}')
    print('Phase 1 foundation may not be complete')