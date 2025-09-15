#!/usr/bin/env python3
"""
Test SQLite embedding storage fix - validates JSON serialization solution.
This test addresses the critical issue identified in the GitHub Copilot session.
"""

import json
import tempfile
import os
from pathlib import Path
import numpy as np
import sys

# Add src to path for raglite imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from raglite._config import RAGLiteConfig
from raglite._database import create_database_engine, ChunkEmbedding, Document, Chunk
from sqlmodel import Session, text


def test_embedding_serialization():
    """Test that embeddings can be serialized/deserialized for SQLite."""
    print("🧪 Testing embedding serialization...")
    
    # Test with numpy array
    original_embedding = np.random.rand(384).astype(np.float32)
    
    # Serialize to JSON (what our new SQLiteVec does)
    serialized = json.dumps(original_embedding.tolist())
    
    # Deserialize from JSON
    deserialized = np.asarray(json.loads(serialized), dtype=np.float32)
    
    # Verify they match
    assert np.allclose(original_embedding, deserialized, rtol=1e-6), "Serialization roundtrip failed"
    
    print("✅ Embedding serialization test passed")


def test_sqlite_embedding_insertion():
    """Test that embeddings can be inserted into SQLite database using new JSON approach."""
    print("🧪 Testing SQLite embedding insertion...")
    
    # Create temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        config = RAGLiteConfig(db_url=f'sqlite:///{db_path}')
        engine = create_database_engine(config)
        
        # Create test document and chunk first
        test_doc = Document(
            id="test-doc-1",
            filename="test.txt",
            content="This is a test document about machine learning.",
            metadata_={}
        )
        
        test_chunk = Chunk(
            id="test-chunk-1",
            document_id="test-doc-1", 
            index=0,
            headings="",
            body="This is a test document about machine learning.",
            metadata_={}
        )
        
        # Create test embedding
        test_embedding = np.random.rand(384).astype(np.float32)
        
        # Test insertion using the updated SQLiteVec type
        with Session(engine) as session:
            # Insert document and chunk first
            session.add(test_doc)
            session.add(test_chunk)
            session.commit()
            
            # Now insert embedding - this should work with our JSON serialization fix
            chunk_embedding = ChunkEmbedding(
                chunk_id="test-chunk-1",
                embedding=test_embedding
            )
            session.add(chunk_embedding)
            session.commit()
            
            # Verify insertion by retrieving the embedding
            result = session.execute(
                text("SELECT chunk_id, embedding FROM chunk_embedding WHERE chunk_id = ?"),
                ("test-chunk-1",)
            ).fetchone()
            
            assert result is not None, "Embedding insertion failed"
            assert result[0] == "test-chunk-1", "Chunk ID mismatch"
            
            # Verify the embedding was stored as JSON and can be deserialized
            stored_embedding_json = result[1]
            assert isinstance(stored_embedding_json, str), "Embedding should be stored as JSON string"
            
            # Deserialize and verify
            retrieved_embedding = np.asarray(json.loads(stored_embedding_json), dtype=np.float32)
            assert np.allclose(test_embedding, retrieved_embedding, rtol=1e-6), "Embedding roundtrip failed"
            
        print("✅ SQLite embedding insertion test passed")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_chunk_embedding_orm():
    """Test that ChunkEmbedding ORM operations work with the JSON serialization."""
    print("🧪 Testing ChunkEmbedding ORM operations...")
    
    # Create temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        config = RAGLiteConfig(db_url=f'sqlite:///{db_path}')
        engine = create_database_engine(config)
        
        # Create test document and chunk
        test_doc = Document(
            id="test-doc-2",
            filename="test2.txt",
            content="Another test document.",
            metadata_={}
        )
        
        test_chunk = Chunk(
            id="test-chunk-2",
            document_id="test-doc-2",
            index=0,
            headings="",
            body="Another test document.",
            metadata_={}
        )
        
        test_embedding = np.random.rand(256).astype(np.float32)
        
        with Session(engine) as session:
            # Insert document and chunk
            session.add(test_doc)
            session.add(test_chunk)
            session.commit()
            
            # Insert embedding using ORM
            chunk_embedding = ChunkEmbedding(
                chunk_id="test-chunk-2",
                embedding=test_embedding
            )
            session.add(chunk_embedding)
            session.commit()
            
            # Retrieve using ORM
            retrieved = session.query(ChunkEmbedding).filter_by(chunk_id="test-chunk-2").first()
            assert retrieved is not None, "ORM retrieval failed"
            
            # Verify the embedding data matches
            assert np.allclose(test_embedding, retrieved.embedding, rtol=1e-6), "ORM embedding roundtrip failed"
            
        print("✅ ChunkEmbedding ORM test passed")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_multiple_embeddings():
    """Test bulk insertion of multiple embeddings."""
    print("🧪 Testing multiple embedding insertions...")
    
    # Create temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        config = RAGLiteConfig(db_url=f'sqlite:///{db_path}')
        engine = create_database_engine(config)
        
        # Create test document
        test_doc = Document(
            id="test-doc-multi",
            filename="multi_test.txt",
            content="Multi-chunk test document.",
            metadata_={}
        )
        
        # Create multiple chunks and embeddings
        num_chunks = 5
        chunks = []
        embeddings = []
        
        for i in range(num_chunks):
            chunk = Chunk(
                id=f"test-chunk-multi-{i}",
                document_id="test-doc-multi",
                index=i,
                headings="",
                body=f"This is chunk {i} of the multi-chunk document.",
                metadata_={}
            )
            chunks.append(chunk)
            
            embedding = np.random.rand(128).astype(np.float32)
            embeddings.append(embedding)
        
        with Session(engine) as session:
            # Insert document and chunks
            session.add(test_doc)
            for chunk in chunks:
                session.add(chunk)
            session.commit()
            
            # Insert multiple embeddings
            for i, embedding in enumerate(embeddings):
                chunk_embedding = ChunkEmbedding(
                    chunk_id=f"test-chunk-multi-{i}",
                    embedding=embedding
                )
                session.add(chunk_embedding)
            session.commit()
            
            # Verify all embeddings were inserted
            count = session.execute(
                text("SELECT COUNT(*) FROM chunk_embedding WHERE chunk_id LIKE 'test-chunk-multi-%'")
            ).scalar()
            
            assert count == num_chunks, f"Expected {num_chunks} embeddings, got {count}"
            
            # Verify each embedding can be retrieved correctly
            for i, original_embedding in enumerate(embeddings):
                retrieved = session.query(ChunkEmbedding).filter_by(chunk_id=f"test-chunk-multi-{i}").first()
                assert retrieved is not None, f"Failed to retrieve embedding {i}"
                assert np.allclose(original_embedding, retrieved.embedding, rtol=1e-6), f"Embedding {i} roundtrip failed"
        
        print("✅ Multiple embeddings test passed")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def main():
    """Run all SQLite embedding fix tests."""
    print("🚀 SQLite Embedding Fix Validation\n")
    print("This test suite validates the fix for the GitHub Copilot session failure:")
    print("  Issue: sqlite3.ProgrammingError: Error binding parameter 2: type 'list' is not supported")
    print("  Solution: JSON serialization of embeddings for SQLite storage\n")
    
    try:
        # Test 1: Basic serialization
        test_embedding_serialization()
        
        # Test 2: Database insertion
        test_sqlite_embedding_insertion()
        
        # Test 3: ORM operations
        test_chunk_embedding_orm()
        
        # Test 4: Multiple embeddings
        test_multiple_embeddings()
        
        print("\n🎉 All SQLite embedding tests passed!")
        print("✅ The GitHub Copilot session blocking issue has been resolved")
        print("✅ SQLite backend can now store and retrieve embeddings using JSON serialization")
        print("✅ Ready to proceed with Phase 2 implementation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
