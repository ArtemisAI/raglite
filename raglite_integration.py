import os
import logging

# Import RAGLite components
from raglite import RAGLiteConfig, Document, insert_documents, hybrid_search


def load_config():
    # Load configuration from environment variables or use defaults
    db_url = os.getenv("RAGLITE_DB_URL", "postgresql://mcp_agent_rw:secure_password_2025@localhost:5432/divorce_case")
    llm = os.getenv("RAGLITE_LLM", "gpt-4o-mini")
    embedder = os.getenv("RAGLITE_EMBEDDER", "text-embedding-3-large")
    return RAGLiteConfig(
        db_url=db_url,
        llm=llm,
        embedder=embedder,
        chunk_max_size=2048
    )


def test_database_connection(config):
    from raglite._database import create_database_engine
    engine = create_database_engine(config)
    with engine.connect() as conn:
        result = conn.execute("SELECT version()")
        version = result.scalar()
        print("Database connected. PostgreSQL version:", version)


def test_insert_and_search(config):
    # Create a dummy legal document
    dummy_content = (
        "This is a test legal document for RAGLite integration. "
        "It discusses aspects of divorce case management and legal procedures."
    )
    dummy_doc = Document.from_text(dummy_content, filename="dummy_doc.txt")

    # Insert document into the database
    insert_documents([dummy_doc], config=config)
    print("Inserted dummy document into the database.")

    # Execute a hybrid search
    query = "divorce case"
    results = hybrid_search(query=query, num_results=5, config=config)
    print("Search results for query 'divorce case':", results)


def main():
    logging.basicConfig(level=logging.INFO)
    print("Running RAGLite integration tests...")

    config = load_config()
    test_database_connection(config)
    test_insert_and_search(config)

    print("RAGLite integration tests complete.")


if __name__ == "__main__":
    main()
