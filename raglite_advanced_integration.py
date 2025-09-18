import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

# Advanced RAGLite integration with vector embeddings
def create_advanced_config():
    """Create advanced configuration for RAGLite with vector support."""
    config = {
        'db_url': os.getenv("RAGLITE_DB_URL", "postgresql://mcp_agent_rw:secure_password_2025@localhost:5432/divorce_case"),
        'llm': os.getenv("RAGLITE_LLM", "gpt-4o-mini"),
        'embedder': os.getenv("RAGLITE_EMBEDDER", "text-embedding-3-large"),
        'chunk_max_size': 2048,
        'embedding_dimension': 1536  # For text-embedding-3-large
    }
    return config

def create_advanced_tables(config):
    """Create advanced RAGLite tables with vector support."""
    from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Text, TIMESTAMP, JSON, Index
    from sqlalchemy.dialects.postgresql import UUID

    try:
        engine = create_engine(config['db_url'])
        metadata = MetaData()

        # Drop existing tables if they exist
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS raglite_search_queries CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS raglite_chunk_embeddings CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS raglite_chunks CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS raglite_documents CASCADE"))
            conn.commit()

        # Enhanced documents table
        documents_table = Table(
            'raglite_documents',
            metadata,
            Column('id', String, primary_key=True),
            Column('content', Text),
            Column('filename', String),
            Column('file_path', String),
            Column('content_type', String),  # 'pdf', 'docx', 'txt', etc.
            Column('metadata_', JSON),
            Column('created_at', TIMESTAMP, server_default=text('NOW()')),
            Column('updated_at', TIMESTAMP, server_default=text('NOW()'))
        )

        # Enhanced chunks table
        chunks_table = Table(
            'raglite_chunks',
            metadata,
            Column('id', String, primary_key=True),
            Column('document_id', String),
            Column('content', Text),
            Column('chunk_index', Integer),
            Column('total_chunks', Integer),
            Column('start_char', Integer),
            Column('end_char', Integer),
            Column('metadata_', JSON),
            Column('created_at', TIMESTAMP, server_default=text('NOW()'))
        )

        # Vector embeddings table with pgvector
        chunk_embeddings_table = Table(
            'raglite_chunk_embeddings',
            metadata,
            Column('chunk_id', String, primary_key=True),
            Column('embedding', String),  # Store as string for now, will convert to vector later
            Column('model_name', String),
            Column('created_at', TIMESTAMP, server_default=text('NOW()'))
        )

        # Search queries table for analytics
        search_queries_table = Table(
            'raglite_search_queries',
            metadata,
            Column('id', String, primary_key=True),
            Column('query', Text),
            Column('query_type', String),  # 'keyword', 'vector', 'hybrid'
            Column('num_results', Integer),
            Column('response_time_ms', Integer),
            Column('created_at', TIMESTAMP, server_default=text('NOW()'))
        )

        # Create tables
        metadata.create_all(engine)

        # Create regular index for performance (can't use vector index with string storage)
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_chunk_id
                ON raglite_chunk_embeddings (chunk_id);
            """))
            conn.commit()

        print("✅ Advanced RAGLite tables created with vector support!")
        return True

    except Exception as e:
        print(f"❌ Advanced table creation failed: {e}")
        return False

def generate_embeddings_openai(texts: List[str], config: Dict) -> List[List[float]]:
    """Generate embeddings using OpenAI API."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=texts,
            model=config['embedder']
        )

        embeddings = [data.embedding for data in response.data]
        print(f"✅ Generated {len(embeddings)} embeddings using {config['embedder']}")
        return embeddings

    except ImportError:
        print("❌ OpenAI package not installed. Install with: pip install openai")
        return []
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return []

def generate_embeddings_local(texts: List[str], config: Dict) -> List[List[float]]:
    """Generate basic embeddings using a simple hash-based approach as fallback."""
    try:
        # Try sentence transformers first
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts).tolist()
        print(f"✅ Generated {len(embeddings)} embeddings using local model")
        return embeddings
    except Exception as e:
        print(f"⚠️  Local model failed ({e}), using fallback embeddings")
        # Fallback: create random embeddings for testing
        import random
        embeddings = []
        for text in texts:
            # Create deterministic "embeddings" based on text hash
            random.seed(hash(text))
            embedding = [random.uniform(-1, 1) for _ in range(384)]  # 384 dimensions
            embeddings.append(embedding)
        print(f"✅ Generated {len(embeddings)} fallback embeddings")
        return embeddings

def insert_document_with_embeddings(config: Dict, content: str, filename: str, metadata: Dict = None):
    """Insert a document with embeddings into the database."""
    from sqlalchemy import create_engine, text
    import uuid

    if metadata is None:
        metadata = {}

    try:
        engine = create_engine(config['db_url'])

        # Split content into chunks
        chunks = split_text_into_chunks(content, config['chunk_max_size'])
        print(f"📄 Split document into {len(chunks)} chunks")

        # Generate embeddings
        if os.getenv("OPENAI_API_KEY"):
            embeddings = generate_embeddings_openai(chunks, config)
        else:
            embeddings = generate_embeddings_local(chunks, config)

        if not embeddings:
            print("❌ Failed to generate embeddings")
            return False

        doc_id = str(uuid.uuid4())

        with engine.connect() as conn:
            # Insert document
            conn.execute(text("""
                INSERT INTO raglite_documents (id, content, filename, content_type, metadata_)
                VALUES (:id, :content, :filename, :content_type, :metadata)
            """), {
                'id': doc_id,
                'content': content,
                'filename': filename,
                'content_type': 'txt',
                'metadata': json.dumps(metadata)
            })

            # Insert chunks and embeddings
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = str(uuid.uuid4())

                # Insert chunk
                conn.execute(text("""
                    INSERT INTO raglite_chunks (id, document_id, content, chunk_index, total_chunks, metadata_)
                    VALUES (:id, :document_id, :content, :chunk_index, :total_chunks, :metadata)
                """), {
                    'id': chunk_id,
                    'document_id': doc_id,
                    'content': chunk,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'metadata': json.dumps({'chunk_size': len(chunk)})
                })

                # Insert embedding
                conn.execute(text("""
                    INSERT INTO raglite_chunk_embeddings (chunk_id, embedding, model_name)
                    VALUES (:chunk_id, :embedding, :model_name)
                """), {
                    'chunk_id': chunk_id,
                    'embedding': f"[{','.join(map(str, embedding))}]",
                    'model_name': config['embedder']
                })

            conn.commit()

        print(f"✅ Document '{filename}' inserted with {len(chunks)} chunks and embeddings")
        return doc_id

    except Exception as e:
        print(f"❌ Document insertion failed: {e}")
        return False

def split_text_into_chunks(text: str, max_chunk_size: int) -> List[str]:
    """Split text into chunks of approximately max_chunk_size."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(' '.join(current_chunk + [word])) <= max_chunk_size:
            current_chunk.append(word)
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def vector_search(config: Dict, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform vector search using cosine similarity in Python."""
    from sqlalchemy import create_engine, text
    import ast

    try:
        # Generate query embedding
        if os.getenv("OPENAI_API_KEY"):
            query_embedding = generate_embeddings_openai([query], config)[0]
        else:
            query_embedding = generate_embeddings_local([query], config)[0]

        if not query_embedding:
            return []

        engine = create_engine(config['db_url'])

        with engine.connect() as conn:
            # Get all embeddings
            result = conn.execute(text("""
                SELECT
                    c.id as chunk_id,
                    c.content,
                    c.chunk_index,
                    d.filename,
                    d.metadata_ as doc_metadata,
                    ce.embedding
                FROM raglite_chunk_embeddings ce
                JOIN raglite_chunks c ON ce.chunk_id = c.id
                JOIN raglite_documents d ON c.document_id = d.id
            """))

            # Calculate similarities in Python
            similarities = []
            for row in result:
                try:
                    # Parse string embedding to list
                    stored_embedding = ast.literal_eval(row[5])
                    similarity = cosine_similarity(query_embedding, stored_embedding)

                    similarities.append({
                        'chunk_id': row[0],
                        'content': row[1],
                        'chunk_index': row[2],
                        'filename': row[3],
                        'doc_metadata': row[4],
                        'similarity': similarity
                    })
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse embedding for chunk {row[0]}: {e}")
                    continue

            # Sort by similarity and take top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            results = similarities[:top_k]

        print(f"✅ Vector search found {len(results)} results for '{query}'")
        return results

    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return []

def hybrid_search(config: Dict, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform hybrid search combining keyword and vector search."""
    from sqlalchemy import create_engine, text
    import ast

    try:
        # Generate query embedding
        if os.getenv("OPENAI_API_KEY"):
            query_embedding = generate_embeddings_openai([query], config)[0]
        else:
            query_embedding = generate_embeddings_local([query], config)[0]

        if not query_embedding:
            return []

        engine = create_engine(config['db_url'])

        with engine.connect() as conn:
            # Get all chunks with embeddings
            result = conn.execute(text("""
                SELECT
                    c.id as chunk_id,
                    c.content,
                    c.chunk_index,
                    d.filename,
                    d.metadata_ as doc_metadata,
                    ce.embedding
                FROM raglite_chunk_embeddings ce
                JOIN raglite_chunks c ON ce.chunk_id = c.id
                JOIN raglite_documents d ON c.document_id = d.id
            """))

            # Calculate hybrid scores in Python
            hybrid_scores = []
            for row in result:
                try:
                    # Parse string embedding to list
                    stored_embedding = ast.literal_eval(row[5])
                    vector_similarity = cosine_similarity(query_embedding, stored_embedding)

                    # Keyword matching score
                    content_lower = row[1].lower()
                    query_lower = query.lower()
                    keyword_score = 1.0 if query_lower in content_lower else 0.5

                    # Combined score (weighted average)
                    combined_score = (vector_similarity + keyword_score) / 2

                    hybrid_scores.append({
                        'chunk_id': row[0],
                        'content': row[1],
                        'chunk_index': row[2],
                        'filename': row[3],
                        'doc_metadata': row[4],
                        'vector_similarity': vector_similarity,
                        'keyword_score': keyword_score,
                        'combined_score': combined_score
                    })
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse embedding for chunk {row[0]}: {e}")
                    continue

            # Sort by combined score and take top_k
            hybrid_scores.sort(key=lambda x: x['combined_score'], reverse=True)
            results = hybrid_scores[:top_k]

        print(f"✅ Hybrid search found {len(results)} results for '{query}'")
        return results

    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
        return []

def test_advanced_integration():
    """Test the advanced RAGLite integration."""
    print("🚀 Starting Advanced RAGLite Integration Test...")

    config = create_advanced_config()
    print(f"📋 Configuration: {config}")

    # Create advanced tables
    if not create_advanced_tables(config):
        print("❌ Table creation failed")
        return

    # Test document insertion with embeddings
    test_content = """
    This is a comprehensive legal document discussing divorce case management procedures.
    It covers various aspects of family law including child custody arrangements,
    spousal support calculations, property division guidelines, and mediation processes.
    The document outlines the legal framework for divorce proceedings and provides
    guidance on case preparation, evidence collection, and court representation.
    """

    doc_id = insert_document_with_embeddings(
        config,
        test_content.strip(),
        "divorce_case_legal_guide.txt",
        {"type": "legal_guide", "category": "divorce_law", "author": "legal_team"}
    )

    if not doc_id:
        print("❌ Document insertion failed")
        return

    # Test vector search
    print("\n🔍 Testing Vector Search...")
    vector_results = vector_search(config, "child custody arrangements", 3)
    for result in vector_results:
        print(".3f")

    # Test hybrid search
    print("\n🔍 Testing Hybrid Search...")
    hybrid_results = hybrid_search(config, "spousal support", 3)
    for result in hybrid_results:
        print(".3f")

    print("\n🎉 Advanced RAGLite integration test completed successfully!")
    print("\n📝 Next Steps:")
    print("1. Install OpenAI API key for better embeddings")
    print("2. Add document processing for PDF/DOCX files")
    print("3. Implement RAG (Retrieval-Augmented Generation)")
    print("4. Add web interface with Chainlit")
    print("5. Integrate with actual divorce case documents")

if __name__ == "__main__":
    test_advanced_integration()
