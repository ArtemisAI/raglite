#!/usr/bin/env python3
"""Test RAGLite setup and functionality."""

import sys
from pathlib import Path

def test_raglite_initialization():
    """Test RAGLite initialization with PostgreSQL."""
    try:
        from raglite._config import RAGLiteConfig
        import raglite
        
        # Initialize RAGLite with PostgreSQL database
        config = RAGLiteConfig(
            db_url="postgresql://raglite_user:raglite_password@localhost:5432/divorce_case",
            embedder="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        print("✅ RAGLite config created successfully")
        
        # Test basic RAGLite functions are available
        print(f"✅ Available functions: {len(raglite.__all__)} functions")
        print("✅ RAGLite functional API ready")
        
        return config, True
        
    except Exception as e:
        print(f"❌ RAGLite initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_mcp_server_creation():
    """Test MCP server creation."""
    try:
        from raglite._mcp import create_mcp_server
        from raglite._config import RAGLiteConfig
        
        config = RAGLiteConfig(
            db_url="postgresql://raglite_user:raglite_password@localhost:5432/divorce_case",
            embedder="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        mcp_server = create_mcp_server("DivorceRAGLite", config=config)
        print("✅ MCP server created successfully")
        print(f"✅ Server name: {mcp_server.name}")
        
        return mcp_server, True
        
    except Exception as e:
        print(f"❌ MCP server creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_document_processing(config):
    """Test basic document processing."""
    try:
        import raglite
        
        # Test document insertion using functional API
        test_content = """
        This is a test legal document about divorce proceedings in France.
        It contains information about asset division (partage des biens) and custody arrangements (garde des enfants).
        The document discusses French family law and includes terms like "jugement de divorce" and "pension alimentaire".
        This is written in English but contains French legal terminology.
        """
        
        metadata = {
            "title": "Test Legal Document",
            "type": "legal_document",
            "language": "mixed",
            "jurisdiction": "french",
            "document_type": "divorce_proceeding"
        }
        
        # Create a Document object
        document = raglite.Document(content=test_content, metadata=metadata)
        
        # Insert document using functional API
        doc_ids = raglite.insert_documents([document], config=config)
        print(f"✅ Document inserted with IDs: {doc_ids}")
        
        # Test search functionality
        search_queries = [
            "divorce asset division",
            "garde des enfants",
            "pension alimentaire",
            "French family law"
        ]
        
        for query in search_queries:
            chunks = raglite.retrieve_chunks(query, num_results=3, config=config)
            print(f"✅ Search '{query}' returned {len(chunks)} chunks")
            if chunks:
                print(f"   📄 First chunk preview: {chunks[0].content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("RAGLite Setup Verification")
    print("=" * 60)
    
    # Test 1: RAGLite initialization
    print("\n🔧 Testing RAGLite initialization...")
    config, init_success = test_raglite_initialization()
    
    # Test 2: MCP server creation
    print("\n🔧 Testing MCP server creation...")
    mcp_server, mcp_success = test_mcp_server_creation()
    
    # Test 3: Document processing (only if RAGLite initialized)
    if init_success and config:
        print("\n🔧 Testing document processing...")
        doc_success = test_document_processing(config)
    else:
        doc_success = False
        print("\n❌ Skipping document processing (RAGLite initialization failed)")
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"✅ RAGLite Initialization: {'PASS' if init_success else 'FAIL'}")
    print(f"✅ MCP Server Creation: {'PASS' if mcp_success else 'FAIL'}")
    print(f"✅ Document Processing: {'PASS' if doc_success else 'FAIL'}")
    
    all_success = init_success and mcp_success and doc_success
    print(f"\n🎯 OVERALL STATUS: {'✅ SUCCESS - Ready for production!' if all_success else '❌ FAILED - Issues need resolution'}")
    
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)