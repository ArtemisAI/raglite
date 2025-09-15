# SQLite Implementation Change Request for RAGLite

## 🎯 Objective
Implement comprehensive SQLite support in the ArtemisAI/raglite fork to enable local, single-file database operations while maintaining full compatibility with existing DuckDB and PostgreSQL functionality.

## 📋 Executive Summary
This change request adds SQLite as a third database backend option to RAGLite, enabling users to leverage SQLite's simplicity and portability for RAG applications. The implementation follows the proven approach from the reflectionai/raglite fork while incorporating sqlite-vec for optimal vector search performance.

## 🔍 Current State Analysis
- RAGLite currently supports only DuckDB and PostgreSQL
- Database backend selection is handled in `src/raglite/_database.py`
- Vector search implementations exist for DuckDB (VSS extension) and PostgreSQL (pgvector)
- FTS (Full-Text Search) is implemented for both supported backends

## 🎯 Target State
- Add SQLite as a third supported database backend
- Use sqlite-vec extension for vector search capabilities
- Leverage SQLite's native FTS5 for full-text search
- Maintain backward compatibility with existing configurations
- Provide seamless migration path for users

## 📁 Files to Modify

### 1. `pyproject.toml`
**Purpose**: Add SQLite dependencies

**Changes Required**:
```toml
dependencies = [
    # ... existing dependencies ...
    "sqlite-vec>=0.1.0",
    "pynndescent>=0.5.12",  # Interim vector search solution
]
```

**Rationale**: sqlite-vec provides native vector search capabilities, while pynndescent serves as a fallback for complex vector operations.

### 2. `src/raglite/_config.py`
**Purpose**: Update default configuration to support SQLite

**Changes Required**:
- Change default `db_url` from DuckDB to SQLite:
```python
db_url: str | URL = f"sqlite:///{(cache_path / 'raglite.db').as_posix()}"
```

**Rationale**: SQLite provides better portability and zero-configuration setup for most users.

### 3. `src/raglite/_database.py`
**Purpose**: Core SQLite database engine implementation

**Changes Required**:

#### A. Add SQLite Backend Detection
- Extend `create_database_engine()` function to handle `sqlite://` URLs
- Add SQLite-specific connection arguments and optimizations

#### B. SQLite Performance Optimizations
```python
if db_backend == "sqlite":
    connect_args.update({
        "check_same_thread": False,
        "timeout": 30
    })
    # Apply SQLite performance pragmas
    with Session(engine) as session:
        session.execute(text("PRAGMA journal_mode = WAL"))
        session.execute(text("PRAGMA synchronous = NORMAL"))
        session.execute(text("PRAGMA cache_size = 10000"))
        session.execute(text("PRAGMA temp_store = memory"))
        session.commit()
```

#### C. SQLite Vector Extension Setup
```python
elif db_backend == "sqlite":
    with Session(engine) as session:
        # Load sqlite-vec extension
        try:
            session.execute(text("SELECT load_extension('vec0')"))
        except Exception:
            # Fallback to manual loading if needed
            session.execute(text("SELECT load_extension('/path/to/vec0')"))
        
        # Create vector table and index
        session.execute(text(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings_vec 
            USING vec0(
                id INTEGER PRIMARY KEY,
                embedding FLOAT[{embedding_dim}]
            )
        """))
        session.commit()
```

#### D. FTS5 Index Creation
```python
# Create FTS5 index for keyword search
session.execute(text("""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts 
    USING fts5(
        id UNINDEXED,
        body,
        content='chunk',
        content_rowid='id'
    )
"""))
```

### 4. `src/raglite/_search.py`
**Purpose**: Implement SQLite-specific search operations

**Changes Required**:

#### A. Vector Search Implementation
```python
def _sqlite_vector_search(
    query_embedding: List[float],
    limit: int,
    config: RAGLiteConfig
) -> tuple[list[ChunkId], list[float]]:
    """Perform vector similarity search using sqlite-vec."""
    engine = create_database_engine(config)
    
    with Session(engine) as session:
        # Use sqlite-vec for similarity search
        result = session.execute(text(f"""
            SELECT chunk_id, vec_distance_cosine(embedding, ?) as distance
            FROM chunk_embeddings_vec 
            ORDER BY distance 
            LIMIT ?
        """), [query_embedding, limit])
        
        chunk_ids, distances = zip(*result.fetchall()) if result.fetchall() else ([], [])
        scores = [1 - d for d in distances]  # Convert distance to similarity
        return list(chunk_ids), scores
```

#### B. Keyword Search Implementation
```python
def _sqlite_keyword_search(
    query: str,
    limit: int,
    config: RAGLiteConfig
) -> tuple[list[ChunkId], list[float]]:
    """Perform keyword search using SQLite FTS5."""
    engine = create_database_engine(config)
    
    with Session(engine) as session:
        result = session.execute(text("""
            SELECT chunk.id, rank
            FROM chunk_fts 
            JOIN chunk ON chunk.id = chunk_fts.rowid
            WHERE chunk_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """), [query, limit])
        
        chunk_ids, ranks = zip(*result.fetchall()) if result.fetchall() else ([], [])
        # Convert ranks to normalized scores
        scores = [1.0 / (1.0 + rank) for rank in ranks]
        return list(chunk_ids), scores
```

#### C. Hybrid Search Integration
```python
def _sqlite_hybrid_search(
    query: str,
    query_embedding: List[float],
    limit: int,
    config: RAGLiteConfig
) -> tuple[list[ChunkId], list[float]]:
    """Combine vector and keyword search results using RRF."""
    
    # Get vector search results
    vector_ids, vector_scores = _sqlite_vector_search(query_embedding, limit * 2, config)
    
    # Get keyword search results  
    keyword_ids, keyword_scores = _sqlite_keyword_search(query, limit * 2, config)
    
    # Apply Reciprocal Rank Fusion
    return _reciprocal_rank_fusion(
        [(vector_ids, vector_scores), (keyword_ids, keyword_scores)],
        limit=limit
    )
```

### 5. `src/raglite/_insert.py`
**Purpose**: Handle SQLite-specific document insertion and indexing

**Changes Required**:

#### A. SQLite Embedding Storage
```python
def _insert_sqlite_embeddings(
    chunk_embeddings: list[ChunkEmbedding],
    config: RAGLiteConfig
) -> None:
    """Insert embeddings into sqlite-vec table."""
    engine = create_database_engine(config)
    
    with Session(engine) as session:
        for chunk_embedding in chunk_embeddings:
            session.execute(text("""
                INSERT OR REPLACE INTO chunk_embeddings_vec (id, embedding)
                VALUES (?, ?)
            """), [chunk_embedding.chunk_id, chunk_embedding.embedding])
        session.commit()
```

#### B. FTS5 Index Updates
```python
def _update_sqlite_fts_index(
    chunks: list[Chunk],
    config: RAGLiteConfig
) -> None:
    """Update FTS5 index with new chunks."""
    engine = create_database_engine(config)
    
    with Session(engine) as session:
        for chunk in chunks:
            session.execute(text("""
                INSERT OR REPLACE INTO chunk_fts (id, body)
                VALUES (?, ?)
            """), [chunk.id, chunk.body])
        session.commit()
```

### 6. `src/raglite/_typing.py`
**Purpose**: Add SQLite-specific type definitions

**Changes Required**:
```python
# Add SQLite vector distance function
@compiles(EmbeddingDistance, "sqlite")
def _embedding_distance_sqlite(element: EmbeddingDistance, compiler: Any, **kwargs: Any) -> str:
    """Compile embedding distance for SQLite using sqlite-vec."""
    return f"vec_distance_{element.metric}({element.left}, {element.right})"

# SQLite vector type
class SQLiteVec(UserDefinedType[FloatVector]):
    """SQLite vector type using sqlite-vec extension."""
    
    def get_col_spec(self, **kwargs: Any) -> str:
        return f"FLOAT[{self.dimension}]"
```

### 7. `tests/conftest.py`
**Purpose**: Add SQLite testing configuration

**Changes Required**:
```python
@pytest.fixture(scope="session")
def sqlite_url() -> Generator[str, None, None]:
    """Create a temporary SQLite database file and return the database URL."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_file = Path(temp_dir) / "raglite_test.db"
        yield f"sqlite:///{db_file}"

# Add SQLite to database parameter list
@pytest.fixture(
    scope="session",
    params=[
        pytest.param("duckdb", id="duckdb"),
        pytest.param("sqlite", id="sqlite"),  # Add SQLite testing
        pytest.param(
            POSTGRES_URL,
            id="postgres",
            marks=pytest.mark.skipif(not is_postgres_running(), reason="PostgreSQL is not running"),
        ),
    ],
)
def database(request: pytest.FixtureRequest) -> str:
    """Get a database URL to test RAGLite with."""
    if request.param == "sqlite":
        return request.getfixturevalue("sqlite_url")
    elif request.param == "duckdb":
        return request.getfixturevalue("duckdb_url")
    else:
        return request.param
```

### 8. `README.md`
**Purpose**: Update documentation to reflect SQLite support

**Changes Required**:
- Update project description: "RAGLite is a Python toolkit for Retrieval-Augmented Generation (RAG) with DuckDB, PostgreSQL, or SQLite"
- Add SQLite configuration examples
- Document sqlite-vec extension requirements
- Update feature comparison table

## 🔧 Implementation Strategy

### Phase 1: Core SQLite Engine Support
1. Add SQLite dependencies to `pyproject.toml`
2. Implement SQLite backend detection in `_database.py`
3. Add SQLite connection optimization
4. Create basic table structure

### Phase 2: Search Implementation
1. Implement sqlite-vec vector search
2. Add FTS5 keyword search
3. Integrate hybrid search with RRF
4. Add fallback mechanisms

### Phase 3: Integration & Testing
1. Update insertion pipeline for SQLite
2. Add comprehensive test coverage
3. Update documentation
4. Performance optimization

### Phase 4: Advanced Features
1. Query adapter support for SQLite
2. Advanced indexing strategies
3. Migration utilities
4. Performance benchmarking

## 🧪 Testing Requirements

### Unit Tests
- SQLite database engine creation
- Vector search accuracy
- Keyword search functionality
- Hybrid search result fusion
- Embedding insertion and retrieval

### Integration Tests
- End-to-end document processing
- Search performance benchmarks
- Cross-database compatibility
- Migration between backends

### Performance Tests
- Large document corpus handling
- Concurrent query processing
- Memory usage optimization
- Query response time validation

## 📊 Success Criteria

### Functional Requirements
- [ ] SQLite database engine successfully created
- [ ] Vector search returns accurate results
- [ ] Keyword search matches expected behavior
- [ ] Hybrid search properly fuses results
- [ ] All existing tests pass with SQLite backend

### Performance Requirements
- [ ] Vector search completes within 2 seconds for 10K documents
- [ ] Keyword search completes within 500ms
- [ ] Hybrid search maintains sub-2-second response time
- [ ] Memory usage remains below 1GB for typical workloads

### Compatibility Requirements
- [ ] Seamless switching between database backends
- [ ] Existing DuckDB/PostgreSQL functionality unchanged
- [ ] Configuration migration tools available
- [ ] Documentation updated and accurate

## 🚨 Risk Mitigation

### Technical Risks
1. **sqlite-vec Extension Availability**
   - Mitigation: Provide fallback to PyNNDescent
   - Implement extension loading validation

2. **Performance Degradation**
   - Mitigation: Comprehensive benchmarking
   - Optimization strategies documented

3. **Compatibility Issues**
   - Mitigation: Extensive cross-platform testing
   - Clear system requirements documentation

### Implementation Risks
1. **Breaking Changes**
   - Mitigation: Maintain backward compatibility
   - Feature flags for new functionality

2. **Complex Migration**
   - Mitigation: Provide migration utilities
   - Step-by-step migration guide

## 📝 Acceptance Criteria

### Must Have
- SQLite backend fully functional
- All search operations working
- Test suite passing
- Documentation updated

### Should Have
- Performance optimizations implemented
- Migration utilities available
- Advanced indexing options
- Query adapter support

### Could Have
- Advanced SQLite-specific features
- Performance monitoring
- Administrative tools
- Extended configuration options

## 🔄 Implementation Notes

### Dependencies
- Ensure sqlite-vec extension is available
- Handle graceful fallbacks for missing extensions
- Validate SQLite version compatibility

### Error Handling
- Comprehensive error messages for SQLite-specific issues
- Graceful degradation when extensions unavailable
- Clear user guidance for troubleshooting

### Documentation
- Update all configuration examples
- Add SQLite-specific troubleshooting guide
- Include performance tuning recommendations

## 🛠 Known Issues & Diagnostics

- **Import Conflicts**: RAGLite imports fail due to TensorFlow/transformers version incompatibilities (e.g., `register_load_context_function` missing).
- **Rerankers Dependency**: The `rerankers` package pulls in heavy ML frameworks, causing import errors in CI and local tests.
- **Testing Environment**: Full end-to-end tests against the SQLite backend are blocked by the above import errors.
- **Ollama Not Detected**: Scripts currently assume Ollama CLI is available for embedding operations; missing installation leads to failures.
- **GPU Utilization**: Embedding model setup does not configure NVIDIA GPU support, leading to slower CPU-only execution.

## 📦 Environment & Setup Instructions

To ensure the coding agent and CI environment have all prerequisites, add the following to your GitHub Actions workflows (e.g., `.github/workflows/ci.yml`):

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install Ollama CLI
        run: |
          curl -sSfL https://ollama.com/install.sh | sh
      - name: Verify Ollama installation
        run: |
          ollama --version
      - name: Install Embedding Models
        run: |
          ollama install embeddingemma
      - name: Configure GPU Support
        run: |
          echo "OLLAMA_CUDA_SUPPORT=true" >> $GITHUB_ENV
      - name: Install Dependencies
        run: |
          pip install -r requirements.txt
          pip install sqlite-vec pynndescent
      - name: Run Test Suite
        run: pytest -q
```

- **Local Development**: In your `setup.sh` or `install.sh` bootstrap script, include checks:
  ```bash
  if ! command -v ollama &>/dev/null; then
    curl -sSfL https://ollama.com/install.sh | sh
  fi
  ollama install embeddingemma
  ```

- **GPU Configuration**: Ensure the runner has NVIDIA drivers and CUDA toolkit installed, and set `OLLAMA_CUDA_SUPPORT=true` in the environment.
````
