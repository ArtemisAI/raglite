import numpy as np
from sqlmodel import Session, text
from raglite._config import RAGLiteConfig
from raglite._database import create_database_engine, Chunk, ChunkEmbedding, Document

# 1. Setup
config = RAGLiteConfig(db_url='sqlite:///tests/test_serialization.db')
engine = create_database_engine(config)

# Clean up previous test data
with Session(engine) as session:
    session.execute(text("DELETE FROM chunk_embedding"))
    session.execute(text("DELETE FROM chunk"))
    session.execute(text("DELETE FROM document"))
    session.commit()

# 2. Create a sample document and chunk
with Session(engine) as session:
    doc = Document.from_text("This is a test document.", id="doc1")
    chunk = Chunk.from_body(doc, 0, "This is a test chunk.")
    session.add(doc)
    session.add(chunk)
    session.commit()
    session.refresh(chunk)

    # 3. Create and insert a sample embedding
    original_embedding = np.random.rand(1, 1536).astype(np.float32)
    embedding_record = ChunkEmbedding(chunk_id=chunk.id, embedding=original_embedding.flatten())
    session.add(embedding_record)
    session.commit()
    session.refresh(embedding_record)


    # 4. Retrieve the embedding
    retrieved_embedding_record = session.get(ChunkEmbedding, embedding_record.id)
    retrieved_embedding = retrieved_embedding_record.embedding

    # 5. Compare the embeddings
    assert np.allclose(original_embedding.flatten(), retrieved_embedding, atol=1e-6), "Embedding serialization failed!"
    print("✅ Embedding serialization test passed!")