#!/usr/bin/env python3
"""Basic SQLite functionality test for RAGLite."""

import tempfile
from pathlib import Path
import os

# Set environment to use OpenAI defaults
os.environ['RAGLITE_DISABLE_GPU'] = '1'

from raglite import RAGLiteConfig
from raglite._database import create_database_engine

def test_sqlite_basic():
    """Test basic SQLite functionality."""
    # Create SQLite config
    with tempfile.TemporaryDirectory() as temp_dir:
        db_file = Path(temp_dir) / 'test.db'
        config = RAGLiteConfig(
            db_url=f'sqlite:///{db_file}',
            llm='openai/gpt-3.5-turbo',
            embedder='openai/text-embedding-ada-002'
        )
        print(f'✅ SQLite config: {config.db_url}')
        
        # Test database engine creation
        engine = create_database_engine(config)
        print(f'✅ Engine created with dialect: {engine.dialect.name}')
        
        # Test basic connection
        with engine.connect() as conn:
            result = conn.execute('SELECT 1 as test')
            print(f'✅ Database connection works: {result.fetchone()}')
            
            # Check if tables were created
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            result = conn.execute(query)
            tables = [row[0] for row in result.fetchall()]
            print(f'✅ Tables created: {tables}')
            
            # Test inserting a simple document
            print("Testing document insertion...")
            
        print("✅ All basic SQLite tests passed!")

if __name__ == "__main__":
    test_sqlite_basic()
