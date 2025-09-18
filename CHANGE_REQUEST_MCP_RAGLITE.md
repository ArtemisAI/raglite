# Change Request: RAGLite MCP Server Implementation

**Date:** September 17, 2025  
**Project:** Divorce Case Management System  
**Component:** RAGLite Document Processing with MCP Server  
**Priority:** High  
**Status:** Implementation Required  

## Executive Summary

Implement a lightweight, functional document processing system using RAGLite architecture with MCP (Model Context Protocol) server capabilities. The system must process legal documents from the divorce case management system while avoiding dependency conflicts that have blocked previous implementation attempts.

## Current Status

### ✅ Infrastructure Available
- PostgreSQL database with pgvector extension running in Docker
- Database schema: `raglite_documents`, `raglite_chunks`, `raglite_chunk_embeddings` tables
- Case_Documents folder with legal documents (PDFs, DOCX, TXT in French/English)
- PDF2MD-HTTP tool for PDF processing
- Working database connection and basic RAGLite integration

### ❌ Critical Blockers
- TensorFlow dependency conflicts preventing MCP server startup
- Heavy ML library conflicts (sentence-transformers, transformers)
- Complex embedding models requiring GPU/heavy dependencies

## Implementation Strategy: DEPENDENCY RESOLUTION & FULL ML CAPABILITIES

### Core Principle
**Preserve RAGLite's ML Power** - RAGLite's strength lies in its ML capabilities (vector embeddings, semantic search, advanced retrieval). The goal is to resolve dependency conflicts while maintaining full functionality, not to strip away the features that make it powerful.

### PHASE 0: DIAGNOSTIC AND DEPENDENCY RESOLUTION (CRITICAL)
**Objective:** Resolve TensorFlow/ML dependency conflicts to enable full RAGLite functionality

**Root Cause Analysis:**
The error `tensorflow.python.framework.errors_impl.AlreadyExistsError: Another metric with the same name already exists` indicates:
1. Multiple TensorFlow imports in the same Python session
2. Conflicting TensorFlow/Keras versions (observed: TF 2.10, 2.12, keras 2.10, keras-nightly 2.14)
3. Background Python processes with TensorFlow already loaded
4. Potential import order issues

**Diagnostic Procedure for Smaller AI Agent:**

**Step 1: Environment Assessment**
```powershell
# Check running Python processes
Get-Process | Where-Object {$_.ProcessName -like "*python*"}

# Check TensorFlow/Keras versions
pip list | Select-String "tensorflow|keras" -CaseSensitive:$false

# Check for multiple Python environments
where python
python --version
```

**Step 2: Clean Environment Setup**
```powershell
# Kill all Python processes to clean memory
Get-Process python | Stop-Process -Force

# Remove conflicting packages
pip uninstall tensorflow tensorflow-gpu tensorflow-intel tensorflow-estimator tensorflow-io-gcs-filesystem keras keras-nightly Keras-Preprocessing tf-nightly-intel tb-nightly tf-estimator-nightly -y

# Install clean, compatible versions
pip install tensorflow==2.15.0
# OR alternative: pip install tensorflow-cpu==2.15.0 (if GPU not needed)
```

**Step 3: Verification Protocol**
```python
# Test 1: Basic TensorFlow import
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

# Test 2: Keras compatibility
python -c "import tensorflow as tf; import keras; print('Keras version:', keras.__version__)"

# Test 3: RAGLite import chain
python -c "from raglite._config import RAGLiteConfig; print('RAGLite config loaded successfully')"

# Test 4: MCP server creation
python -c "from raglite._mcp import create_mcp_server; print('MCP server import successful')"
```

**Step 4: Alternative Resolution Paths**

**Path A: Containerized Environment**
If dependency conflicts persist, use the provided devcontainer:
```bash
# Use the .devcontainer/devcontainer.json configuration
# This provides a clean Python environment with proper dependency management
cd f:\_Divorce_2025\@.tools\raglite
# Open in VS Code devcontainer or use docker-compose
docker-compose up devcontainer
```

**Path B: Virtual Environment Isolation**
```powershell
# Create isolated environment
python -m venv raglite_env
raglite_env\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r @.tools\raglite\pyproject.toml
```

**Path C: Conda Environment (fallback)**
```bash
conda create -n raglite python=3.10
conda activate raglite
conda install tensorflow=2.15
pip install -e @.tools\raglite
```

### Phase 1: RAGLite MCP Server with Full ML Capabilities
**File:** `f:\_Divorce_2025\test_raglite_mcp.py`

**Objective:** Use RAGLite's built-in MCP server with full ML capabilities

**Requirements:**
- Use RAGLite's native MCP server implementation
- Leverage vector embeddings and semantic search
- Implement these MCP tools (already provided by RAGLite):
  - `search_knowledge_base(query)` - Semantic search with embeddings
  - `insert_document(content, metadata)` - With automatic chunking and embedding
  - `list_documents(limit=10)` - Database listing
  - `get_database_stats()` - Analytics

**Database Connection (PostgreSQL with pgvector):**
```python
RAGLITE_DB_URL = "postgresql://postgres:postgres@localhost:5432/divorce_case"
RAGLITE_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight but effective
RAGLITE_LLM = "openai/gpt-4"  # For advanced processing
```

**Implementation Steps:**
```python
# Step 1: Test RAGLite CLI MCP server
python -m raglite mcp run --help

# Step 2: Start MCP server with divorce case database
python -m raglite mcp run \
  --db-url "postgresql://postgres:postgres@localhost:5432/divorce_case" \
  --embedder "sentence-transformers/all-MiniLM-L6-v2"

# Step 3: Test MCP server functionality
# Create test script to verify MCP tools work
```

**Fallback Implementation (if CLI fails):**
```python
# Direct Python implementation using RAGLite classes
from raglite._mcp import create_mcp_server
from raglite._config import RAGLiteConfig

config = RAGLiteConfig(
    db_url="postgresql://postgres:postgres@localhost:5432/divorce_case",
    embedder="sentence-transformers/all-MiniLM-L6-v2"
)
mcp = create_mcp_server("DivorceRAGLite", config=config)
mcp.run()
```

### Phase 2: Document Processing with RAGLite Intelligence
**File:** `f:\_Divorce_2025\raglite_document_processor.py`

**Objective:** Use RAGLite's built-in document processing capabilities

**RAGLite Features to Leverage:**
- **Automatic Chunking:** RAGLite handles optimal text chunking
- **Vector Embeddings:** Semantic embeddings for better search
- **Multi-language Support:** Handle French/English legal documents
- **Metadata Extraction:** Automatic metadata from document content

**Implementation Strategy:**
```python
import raglite

# Initialize RAGLite with divorce case database
rag = raglite.RAGLite(
    db_url="postgresql://postgres:postgres@localhost:5432/divorce_case",
    embedder="sentence-transformers/all-MiniLM-L6-v2"
)

# Process documents using RAGLite's insert method
for document_path in scan_case_documents():
    content = read_document(document_path)
    metadata = extract_metadata(document_path)
    
    # RAGLite handles chunking, embedding, and storage
    rag.insert(content, metadata=metadata)
```

**Document Scanner Integration:**
```python
def scan_case_documents():
    """Scan Case_Documents folder for processable files."""
    base_path = Path("f:/_Divorce_2025/Case_Documents")
    supported_types = ['.txt', '.md', '.pdf', '.docx']
    
    for file_path in base_path.rglob("*"):
        if file_path.suffix.lower() in supported_types:
            yield {
                'path': str(file_path),
                'name': file_path.name,
                'size': file_path.stat().st_size,
                'modified': file_path.stat().st_mtime,
                'type': file_path.suffix
            }
```

### Phase 3: Advanced Search and Retrieval
**File:** `f:\_Divorce_2025\raglite_search_system.py`

**Objective:** Implement RAGLite's advanced search capabilities

**RAGLite Search Features:**
- **Semantic Search:** Vector similarity for conceptual matching
- **Hybrid Search:** Combines keyword and semantic search
- **Query Adaptation:** LLM-enhanced query understanding
- **Multi-vector Search:** Multiple embedding strategies
- **Relevance Ranking:** Advanced scoring algorithms

**Search Implementation:**
```python
def search_legal_documents(query: str, search_type: str = "hybrid"):
    """Search legal documents using RAGLite's advanced capabilities."""
    
    if search_type == "semantic":
        # Pure vector search for conceptual matching
        results = rag.search(query, search_method="vector", num_results=10)
    
    elif search_type == "hybrid":
        # Combined keyword + semantic search
        results = rag.search(query, search_method="hybrid", num_results=10)
    
    elif search_type == "adaptive":
        # LLM-enhanced query understanding
        results = rag.search(query, search_method="adaptive", num_results=10)
    
    return results

# Legal-specific search patterns
def search_legal_concepts(concept: str, jurisdiction: str = "french"):
    """Search for legal concepts with jurisdiction awareness."""
    enhanced_query = f"{concept} legal {jurisdiction} law divorce"
    return search_legal_documents(enhanced_query, "adaptive")
```

### Phase 4: PDF Processing Integration
**File:** `f:\_Divorce_2025\pdf_processing_pipeline.py`

**Objective:** Integrate PDF2MD-HTTP tool with RAGLite processing

**PDF Processing Workflow:**
```python
def process_pdf_documents():
    """Process PDF files using PDF2MD-HTTP then RAGLite."""
    
    # Step 1: Convert PDFs to Markdown using existing tool
    pdf_files = scan_case_documents(file_type='.pdf')
    
    for pdf_file in pdf_files:
        try:
            # Use PDF2MD-HTTP conversion
            markdown_content = convert_pdf_to_markdown(pdf_file['path'])
            
            # Step 2: Process with RAGLite
            metadata = {
                'source_file': pdf_file['path'],
                'original_type': 'pdf',
                'processed_type': 'markdown',
                'file_size': pdf_file['size'],
                'processing_date': datetime.now().isoformat()
            }
            
            # RAGLite insertion with full ML processing
            rag.insert(markdown_content, metadata=metadata)
            
        except Exception as e:
            log_error(f"Failed to process {pdf_file['path']}: {e}")
            continue

def convert_pdf_to_markdown(pdf_path: str) -> str:
    """Convert PDF to markdown using PDF2MD-HTTP tool."""
    # Use the existing PDF2MD-HTTP tool at f:\_Divorce_2025\@.tools\PDF2MD-HTTP
    # Implementation depends on PDF2MD-HTTP API
    pass
```

### Phase 5: System Testing
**File:** `f:\_Divorce_2025\test_document_system.py`

**Test Cases:**
- Process a single legal document from Case_Documents
- Search for legal terms in French and English
- Verify database contains processed documents
- Test error handling with malformed files

## Technical Specifications

### Database Operations

**Insert Document:**
```sql
INSERT INTO raglite_documents (content, metadata) VALUES (%s, %s) RETURNING id;
INSERT INTO raglite_chunks (document_id, content, metadata, chunk_index) VALUES (%s, %s, %s, %s);
```

**Search Documents:**
```sql
SELECT c.content, c.metadata, d.metadata as doc_metadata,
       ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', %s)) as rank
FROM raglite_chunks c 
JOIN raglite_documents d ON c.document_id = d.id
WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', %s)
ORDER BY rank DESC LIMIT %s;
```

### File Structure to Create
```
f:\_Divorce_2025\
├── minimal_mcp_server.py          # Lightweight MCP server
├── document_scanner.py            # Scan Case_Documents folder
├── text_processor.py              # Simple text processing
├── document_processor.py          # Main processing pipeline
├── test_document_system.py        # Testing script
└── requirements_minimal.txt       # Only essential dependencies
```

### Dependencies

**APPROVED (Minimal):**
```
fastmcp
psycopg2-binary
pathlib (built-in)
json (built-in)
re (built-in)
```

**FORBIDDEN (Conflict-prone):**
- ❌ tensorflow
- ❌ keras
- ❌ transformers
- ❌ sentence-transformers
- ❌ torch
- ❌ rerankers

## Success Criteria

1. ✅ MCP server starts without TensorFlow errors
2. ✅ Can process at least one legal document from Case_Documents
3. ✅ Search returns relevant results for legal queries
4. ✅ Database contains processed document chunks
5. ✅ System handles French and English documents

## Risk Mitigation

### Error Recovery Strategy
If any step fails:
1. Log the error with full context
2. Continue processing other documents
3. Provide detailed error report
4. Suggest manual intervention steps

### Fallback Options
- If PostgreSQL full-text search is insufficient, implement simple string matching
- If PDF processing fails, focus on TXT/MD files first
- If MCP server has issues, provide direct Python API

## Implementation Timeline

| Phase | Task | Estimated Effort | Dependencies |
|-------|------|------------------|--------------|
| 1 | Minimal MCP Server | 2-3 hours | Database schema |
| 2 | Document Scanner | 1-2 hours | File system access |
| 3 | Text Processing | 2-3 hours | None |
| 4 | Document Pipeline | 2-3 hours | Phases 1-3 |
| 5 | Testing & Validation | 1-2 hours | Phase 4 |

**Total Estimated Effort:** 8-13 hours

## Acceptance Criteria

### Functional Requirements
- [ ] MCP server responds to all defined tool calls
- [ ] Document scanner identifies all files in Case_Documents
- [ ] Text processor handles French and English documents
- [ ] Search returns ranked results for legal queries
- [ ] System processes at least 10 sample documents

### Non-Functional Requirements
- [ ] No TensorFlow dependency conflicts
- [ ] Startup time under 30 seconds
- [ ] Memory usage under 1GB
- [ ] Error rate under 5% for document processing

## Future Enhancements (Out of Scope)

- Vector embeddings using OpenAI API
- Advanced NLP processing
- Multi-language stemming
- Machine learning-based relevance ranking
- Real-time document monitoring

## Approval Required

This change request requires approval from:
- [ ] Technical Lead
- [ ] Project Manager
- [ ] Quality Assurance

**Estimated Completion Date:** September 18, 2025

---

**Contact:** AI Development Team  
**Last Updated:** September 17, 2025