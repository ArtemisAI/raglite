import os
import logging
from pathlib import Path

# Create a minimal RAGLite integration that avoids problematic dependencies
def create_minimal_config():
    """Create a minimal configuration for RAGLite without rerankers."""
    from sqlalchemy.engine import URL

    # Basic configuration
    config = {
        'db_url': os.getenv("RAGLITE_DB_URL", "postgresql://mcp_agent_rw:secure_password_2025@localhost:5432/divorce_case"),
        'llm': os.getenv("RAGLITE_LLM", "gpt-4o-mini"),
        'embedder': os.getenv("RAGLITE_EMBEDDER", "text-embedding-3-large"),
        'chunk_max_size': 2048
    }
    return config

def test_database_connection_minimal(config):
    """Test database connection using SQLAlchemy directly."""
    from sqlalchemy import create_engine, text

    try:
        engine = create_engine(config['db_url'])
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            print("✅ Database connected successfully!")
            print(f"PostgreSQL version: {version}")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_vector_extension(config):
    """Test if pgvector extension is available."""
    from sqlalchemy import create_engine, text

    try:
        engine = create_engine(config['db_url'])
        with engine.connect() as conn:
            # Check if vector extension is available
            result = conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector'"))
            if result.fetchone():
                print("✅ pgvector extension is installed!")
                return True
            else:
                print("❌ pgvector extension not found. Installing...")
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                print("✅ pgvector extension installed!")
                return True
    except Exception as e:
        print(f"❌ Vector extension test failed: {e}")
        return False

def create_raglite_tables_minimal(config):
    """Create basic RAGLite tables manually."""
    from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Text, TIMESTAMP, JSON
    from sqlalchemy.dialects.postgresql import UUID

    try:
        engine = create_engine(config['db_url'])
        metadata = MetaData()

        # Create documents table
        documents_table = Table(
            'raglite_documents',
            metadata,
            Column('id', String, primary_key=True),
            Column('content', Text),
            Column('filename', String),
            Column('metadata_', JSON),
            Column('created_at', TIMESTAMP, server_default=text('NOW()'))
        )

        # Create chunks table
        chunks_table = Table(
            'raglite_chunks',
            metadata,
            Column('id', String, primary_key=True),
            Column('document_id', String),
            Column('content', Text),
            Column('metadata_', JSON),
            Column('created_at', TIMESTAMP, server_default=text('NOW()'))
        )

        # Create chunk embeddings table
        chunk_embeddings_table = Table(
            'raglite_chunk_embeddings',
            metadata,
            Column('chunk_id', String, primary_key=True),
            Column('embedding', String),  # We'll store as JSON for now
            Column('created_at', TIMESTAMP, server_default=text('NOW()'))
        )

        # Create tables
        metadata.create_all(engine)
        print("✅ RAGLite tables created successfully!")
        return True

    except Exception as e:
        print(f"❌ Table creation failed: {e}")
        return False

def test_basic_document_insertion(config):
    """Test basic document insertion."""
    from sqlalchemy import create_engine, text
    import uuid
    import json

    try:
        engine = create_engine(config['db_url'])

        # Create a test document
        doc_id = str(uuid.uuid4())
        test_content = "This is a test legal document for RAGLite integration. It discusses aspects of divorce case management and legal procedures."
        test_filename = "test_doc.txt"

        with engine.connect() as conn:
            # Insert document
            conn.execute(text("""
                INSERT INTO raglite_documents (id, content, filename, metadata_)
                VALUES (:id, :content, :filename, :metadata)
            """), {
                'id': doc_id,
                'content': test_content,
                'filename': test_filename,
                'metadata': json.dumps({'type': 'test', 'created_by': 'integration_test'})
            })

            # Insert a chunk
            chunk_id = str(uuid.uuid4())
            conn.execute(text("""
                INSERT INTO raglite_chunks (id, document_id, content, metadata_)
                VALUES (:id, :document_id, :content, :metadata)
            """), {
                'id': chunk_id,
                'document_id': doc_id,
                'content': test_content,
                'metadata': json.dumps({'chunk_index': 0, 'total_chunks': 1})
            })

            conn.commit()

        print("✅ Test document and chunk inserted successfully!")
        return doc_id, chunk_id

    except Exception as e:
        print(f"❌ Document insertion failed: {e}")
        return None, None

def test_basic_search(config, doc_id, chunk_id):
    """Test basic search functionality."""
    from sqlalchemy import create_engine, text

    try:
        engine = create_engine(config['db_url'])

        with engine.connect() as conn:
            # Simple keyword search
            result = conn.execute(text("""
                SELECT d.filename, c.content
                FROM raglite_documents d
                JOIN raglite_chunks c ON d.id = c.document_id
                WHERE c.content ILIKE :query
                LIMIT 5
            """), {'query': '%divorce%'})

            rows = result.fetchall()
            print(f"✅ Search found {len(rows)} results for 'divorce':")
            for row in rows:
                print(f"  - {row[0]}: {row[1][:100]}...")

        return True

    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def main():
    logging.basicConfig(level=logging.INFO)
    print("🚀 Starting RAGLite Minimal Integration Test...")

    # Create configuration
    config = create_minimal_config()
    print(f"📋 Configuration: {config}")

    # Test database connection
    if not test_database_connection_minimal(config):
        print("❌ Database connection test failed. Exiting.")
        return

    # Test vector extension
    if not test_vector_extension(config):
        print("❌ Vector extension test failed. Exiting.")
        return

    # Create tables
    if not create_raglite_tables_minimal(config):
        print("❌ Table creation failed. Exiting.")
        return

    # Test document insertion
    doc_id, chunk_id = test_basic_document_insertion(config)
    if not doc_id:
        print("❌ Document insertion test failed. Exiting.")
        return

    # Test search
    if not test_basic_search(config, doc_id, chunk_id):
        print("❌ Search test failed. Exiting.")
        return

    print("🎉 RAGLite minimal integration test completed successfully!")
    print("\n📝 Next Steps:")
    print("1. Install full RAGLite dependencies (resolve TensorFlow conflicts)")
    print("2. Set up embedding generation")
    print("3. Implement vector search with pgvector")
    print("4. Add document processing pipeline")
    print("5. Integrate with divorce case documents")

if __name__ == "__main__":
    main()
