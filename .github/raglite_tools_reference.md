# Raglite Tools Quick Reference for GitHub Copilot

## Available Tools and APIs

### Core Raglite Components

#### 1. Database Operations (`raglite._database`)
**Purpose**: Vector database management and operations

**Key Classes**:
- `RagliteDB` - Main database interface
- `VectorStore` - Vector storage and retrieval
- `DocumentStore` - Document management

**Key Methods**:
```python
# Database initialization
db = RagliteDB("sqlite:///raglite.db")

# Document operations
db.insert_document(doc: Dict[str, Any]) -> str
db.get_document(doc_id: str) -> Dict[str, Any]
db.update_document(doc_id: str, updates: Dict[str, Any]) -> bool
db.delete_document(doc_id: str) -> bool

# Search operations
db.search(query: str, limit: int = 10) -> List[Dict[str, Any]]
db.search_vectors(vector: np.ndarray, limit: int = 10) -> List[Dict[str, Any]]

# Batch operations
db.insert_documents(docs: List[Dict[str, Any]]) -> List[str]
db.bulk_search(queries: List[str]) -> List[List[Dict[str, Any]]]
```

#### 2. Embedding Operations (`raglite._embed`)
**Purpose**: Text embedding generation and management

**Key Functions**:
```python
# Embedding generation
generate_embeddings(texts: List[str]) -> np.ndarray
generate_embedding(text: str) -> np.ndarray

# Model management
load_embedding_model(model_name: str) -> EmbeddingModel
get_available_models() -> List[str]

# Batch processing
batch_embed(texts: List[str], batch_size: int = 32) -> np.ndarray
```

#### 3. Search Algorithms (`raglite._search`)
**Purpose**: Advanced search and retrieval algorithms

**Key Classes**:
- `VectorSearch` - Vector similarity search
- `HybridSearch` - Combined text and vector search
- `BM25Search` - Traditional text search

**Key Methods**:
```python
# Vector search
searcher = VectorSearch(db)
results = searcher.search(query_vector, top_k=10)

# Hybrid search
hybrid = HybridSearch(db, text_weight=0.3, vector_weight=0.7)
results = hybrid.search("query text", top_k=10)

# Filtered search
results = searcher.search_with_filters(
    query_vector,
    filters={"category": "technical"},
    top_k=10
)
```

#### 4. Text Processing (`raglite._split_*`)
**Purpose**: Text chunking and preprocessing

**Available Splitters**:
```python
# Sentence splitting
from raglite._split_sentences import SentenceSplitter
splitter = SentenceSplitter()
sentences = splitter.split(text)

# Chunk splitting
from raglite._split_chunks import ChunkSplitter
chunker = ChunkSplitter(chunk_size=512, overlap=50)
chunks = chunker.split(text)

# Markdown-aware splitting
from raglite._split_chunklets import MarkdownChunkSplitter
md_splitter = MarkdownChunkSplitter()
chunks = md_splitter.split(markdown_text)
```

#### 5. RAG Pipeline (`raglite._rag`)
**Purpose**: Complete RAG pipeline orchestration

**Key Classes**:
- `RAGPipeline` - Main pipeline class
- `RAGConfig` - Pipeline configuration

**Usage**:
```python
# Initialize pipeline
config = RAGConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=512,
    top_k=5
)
rag = RAGPipeline(config)

# Process documents
rag.add_documents(documents)

# Query
response = rag.query("What is machine learning?")
```

### CLI Tools (`raglite._cli`)

#### Available Commands
```bash
# Database management
raglite db init --path ./data/raglite.db
raglite db status
raglite db optimize

# Document operations
raglite docs add ./documents/
raglite docs list --limit 10
raglite docs search "query text"

# Performance testing
raglite bench embeddings --count 1000
raglite bench search --queries 100
raglite bench memory --vectors 10000
```

### Utility Modules

#### 6. Configuration (`raglite._config`)
**Purpose**: Centralized configuration management

```python
from raglite._config import Config

config = Config()
config.database_url = "sqlite:///raglite.db"
config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
config.chunk_size = 512
```

#### 7. Evaluation (`raglite._eval`)
**Purpose**: Performance evaluation and metrics

```python
from raglite._eval import Evaluator

evaluator = Evaluator()
metrics = evaluator.evaluate_search_quality(
    queries=test_queries,
    ground_truth=relevant_docs
)
print(f"Mean Average Precision: {metrics['map']}")
```

#### 8. Markdown Processing (`raglite._markdown`)
**Purpose**: Markdown document processing

```python
from raglite._markdown import MarkdownProcessor

processor = MarkdownProcessor()
parsed = processor.parse(markdown_content)
metadata = processor.extract_metadata(parsed)
chunks = processor.chunk_by_headers(parsed)
```

## Performance Notes

### Optimization Tips
1. **Batch Operations**: Always use batch processing for multiple items
2. **Vector Indexing**: Use appropriate indexing for large datasets
3. **Memory Management**: Monitor memory usage with large embeddings
4. **Caching**: Implement caching for frequently accessed data
5. **Async Processing**: Use async methods for I/O operations

### Performance Benchmarks
- **Embedding Generation**: ~50ms per document (batch of 32)
- **Vector Search**: <10ms for 100K vectors
- **Document Insertion**: ~5ms per document
- **Text Chunking**: <1ms per KB of text

### Memory Considerations
- **Base Memory**: ~200MB for core functionality
- **Per 1K Documents**: ~50MB additional
- **Embedding Models**: 100-500MB depending on model size
- **Large Datasets**: Monitor memory usage carefully

## Error Handling

### Common Exceptions
```python
from raglite._typing import (
    DatabaseError,
    EmbeddingError,
    SearchError,
    ValidationError
)

try:
    result = db.search(query)
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    # Handle database connection issues
except EmbeddingError as e:
    logger.error(f"Embedding error: {e}")
    # Handle embedding service issues
except SearchError as e:
    logger.error(f"Search error: {e}")
    # Handle search algorithm issues
```

### Best Practices
1. **Always handle exceptions** at appropriate levels
2. **Log errors** with sufficient context
3. **Provide meaningful error messages** to users
4. **Implement retry logic** for transient failures
5. **Validate inputs** before processing

## Usage Examples

### Basic RAG Workflow
```python
from raglite import RagliteDB
from raglite._embed import generate_embeddings
from raglite._search import VectorSearch

# Initialize components
db = RagliteDB("./data/raglite.db")
embeddings = generate_embeddings(documents)
searcher = VectorSearch(db)

# Add documents with embeddings
for doc, emb in zip(documents, embeddings):
    db.insert_document(doc, embedding=emb)

# Search
query_embedding = generate_embeddings([query])[0]
results = searcher.search(query_embedding, top_k=5)
```

### Advanced Configuration
```python
from raglite._rag import RAGPipeline, RAGConfig

config = RAGConfig(
    database_url="sqlite:///raglite.db",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=512,
    chunk_overlap=50,
    top_k=10,
    similarity_threshold=0.7
)

rag = RAGPipeline(config)
response = rag.query("complex query with context")
```

### Performance Monitoring
```python
import time
from raglite._eval import PerformanceMonitor

monitor = PerformanceMonitor()

@monitor.time_operation("embedding_generation")
def generate_embeddings_batch(texts):
    return generate_embeddings(texts)

@monitor.time_operation("vector_search")
def search_vectors(query_vector):
    return searcher.search(query_vector)

# Get performance report
report = monitor.get_report()
print(f"Average embedding time: {report['embedding_generation']['avg_time']}")
```

## Integration Patterns

### FastAPI Integration
```python
from fastapi import FastAPI
from raglite import RagliteDB

app = FastAPI()
db = RagliteDB()

@app.post("/search")
async def search_endpoint(query: str, limit: int = 10):
    results = await db.search_async(query, limit=limit)
    return {"results": results}

@app.post("/add-document")
async def add_document(doc: Dict[str, Any]):
    doc_id = await db.insert_document_async(doc)
    return {"id": doc_id}
```

### Async Processing
```python
import asyncio
from raglite._database import AsyncRagliteDB

async def process_documents_batch(documents):
    async with AsyncRagliteDB() as db:
        tasks = [db.insert_document_async(doc) for doc in documents]
        results = await asyncio.gather(*tasks)
        return results
```

This reference provides quick access to all Raglite tools and APIs with usage examples and performance considerations.
