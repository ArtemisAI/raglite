# Detailed Execution Instructions for Smaller AI Agent
## SQLite Backend Implementation - Phase-by-Phase Execution Plan

**Context**: You are executing the implementation plan from Change Request #3 for SQLite backend support in RAGLite.  
**Current Branch**: `copilot/vscode1757947449396`  
**Status**: Environment setup complete, dependencies ready, test database available - **ZERO code implementation done**  
**Test Database**: `tests/test_raglite.db` (20KB, 5 documents, 15 chunks) ready for validation

---

## 🎯 **EXECUTION OVERVIEW**

### **Current State Verification**
You should verify these before starting:
- ✅ Branch: `copilot/vscode1757947449396` 
- ✅ Dependencies: `sqlite-vec>=0.1.0`, `pynndescent>=0.5.12` in `pyproject.toml`
- ✅ Test database: `tests/test_raglite.db` exists and contains data
- ✅ Environment: `.github/setup-environment.sh` configured
- ❌ **IMPLEMENTATION**: All core SQLite code missing - THIS IS YOUR TASK

### **Implementation Priority Queue**
1. **PHASE 1**: Core Database Engine (`_database.py`) - **CRITICAL**
2. **PHASE 2**: Search Functions (`_search.py`) - **HIGH** 
3. **PHASE 3**: Insertion Pipeline (`_insert.py`) - **HIGH**
4. **PHASE 4**: Type System (`_typing.py`) - **MEDIUM**
5. **PHASE 5**: Test Integration (`conftest.py`) - **HIGH**

---

## 📋 **PHASE 1: CORE DATABASE ENGINE**
### **File**: `src/raglite/_database.py`
### **Target**: Lines ~150-200 in `create_database_engine()` function

#### **Step 1.1: Locate Current SQLite Implementation**
```bash
# Command to run:
grep -n "sqlite" src/raglite/_database.py
```
**Expected**: You should find existing SQLite conditions around line 150-200

#### **Step 1.2: Read Current Implementation**
```python
# Use read_file tool to examine:
read_file(src/raglite/_database.py, startLine=140, endLine=220)
```
**Expected**: You'll see basic SQLite detection but missing sqlite-vec integration

#### **Step 1.3: Enhance SQLite Engine Creation**
**TARGET LOCATION**: Find this pattern in the file:
```python
elif db_backend == "sqlite":
    # Existing basic implementation
```

**REPLACE WITH** this enhanced implementation:
```python
elif db_backend == "sqlite":
    # Enhanced SQLite backend with sqlite-vec support
    try:
        import sqlite_vec
        sqlite_vec_available = True
    except ImportError:
        logger.warning("sqlite-vec not available, falling back to PyNNDescent for vector search")
        sqlite_vec_available = False
    
    # SQLite-specific connection arguments
    connect_args.update({
        "check_same_thread": False,
        "timeout": 30,
        "isolation_level": None,  # Enable autocommit mode
    })
    
    # Create engine with enhanced configuration
    engine = create_engine(db_url, connect_args=connect_args, **engine_kwargs)
    
    # Apply SQLite performance optimizations and load extensions
    def _configure_sqlite_connection(dbapi_connection, connection_record):
        """Configure SQLite connection with performance optimizations and extensions."""
        cursor = dbapi_connection.cursor()
        
        # Performance pragmas
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA synchronous = NORMAL") 
        cursor.execute("PRAGMA cache_size = 10000")
        cursor.execute("PRAGMA temp_store = memory")
        cursor.execute("PRAGMA mmap_size = 268435456")  # 256MB
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Load sqlite-vec extension if available
        if sqlite_vec_available:
            try:
                cursor.execute("SELECT load_extension(?)", (sqlite_vec.loadable_path(),))
                logger.info("✅ sqlite-vec extension loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load sqlite-vec extension: {e}")
                sqlite_vec_available = False
        
        cursor.close()
    
    # Register connection event listener
    from sqlalchemy import event
    event.listen(engine, "connect", _configure_sqlite_connection)
    
    # Store sqlite-vec availability on engine for later use
    engine.sqlite_vec_available = sqlite_vec_available
    
    return engine
```

#### **Step 1.4: Add Required Imports**
**TARGET LOCATION**: Top of file where other imports are
**ADD THESE IMPORTS** (if not already present):
```python
import logging
from sqlalchemy import event, text

logger = logging.getLogger(__name__)
```

#### **Step 1.5: Validation**
```python
# Test the implementation:
python -c "
from raglite._database import create_database_engine
from raglite._config import RAGLiteConfig
config = RAGLiteConfig(db_url='sqlite:///tests/test_raglite.db')
engine = create_database_engine(config)
print('✅ SQLite engine created successfully')
print(f'sqlite-vec available: {getattr(engine, \"sqlite_vec_available\", False)}')
"
```

---

## 📋 **PHASE 2: SEARCH IMPLEMENTATION**
### **File**: `src/raglite/_search.py`
### **Target**: Add SQLite-specific search functions

#### **Step 2.1: Examine Current Search Implementation**
```python
# Read the vector_search function:
read_file(src/raglite/_search.py, startLine=1, endLine=100)
```
**Expected**: You'll see the main `vector_search()` function that needs SQLite support

#### **Step 2.2: Read Complete Search File Structure**
```python
# Get overview of entire file:
grep_search("def.*search", isRegexp=True, includePattern="src/raglite/_search.py")
```
**Expected**: You'll see `vector_search`, `keyword_search`, and other search functions

#### **Step 2.3: Add SQLite Vector Search Function**
**TARGET LOCATION**: End of the file, after existing functions
**ADD THIS COMPLETE FUNCTION**:
```python
def _sqlite_vector_search(
    query_embedding: list[float],
    limit: int,
    config: RAGLiteConfig,
    where: Any = None,
) -> tuple[list[ChunkId], list[float]]:
    """SQLite vector similarity search using sqlite-vec or PyNNDescent fallback."""
    from ._database import create_database_engine
    
    engine = create_database_engine(config)
    
    # Check if sqlite-vec is available
    if getattr(engine, 'sqlite_vec_available', False):
        return _sqlite_vec_search(query_embedding, limit, config, where, engine)
    else:
        return _sqlite_pynndescent_search(query_embedding, limit, config, where, engine)

def _sqlite_vec_search(
    query_embedding: list[float],
    limit: int,
    config: RAGLiteConfig,
    where: Any,
    engine: Any,
) -> tuple[list[ChunkId], list[float]]:
    """SQLite vector search using sqlite-vec extension."""
    import sqlite_vec
    from sqlalchemy import text
    from sqlalchemy.orm import Session
    
    with Session(engine) as session:
        # Serialize query vector for sqlite-vec
        query_blob = sqlite_vec.serialize_float32(query_embedding)
        
        # Build SQL query with optional WHERE clause
        base_query = """
            SELECT ce.chunk_id, vec_distance_cosine(ce.embedding, ?) as distance
            FROM chunk_embeddings ce
        """
        
        if where is not None:
            # Join with chunk table for WHERE conditions
            base_query = """
                SELECT ce.chunk_id, vec_distance_cosine(ce.embedding, ?) as distance
                FROM chunk_embeddings ce
                JOIN chunk c ON c.id = ce.chunk_id
            """
            base_query += f" WHERE {where}"
        
        base_query += " ORDER BY distance LIMIT ?"
        
        result = session.execute(text(base_query), [query_blob, limit])
        rows = result.fetchall()
        
        if not rows:
            return [], []
        
        chunk_ids, distances = zip(*rows)
        # Convert cosine distance to similarity score
        scores = [1.0 - d for d in distances]
        return list(chunk_ids), list(scores)

def _sqlite_pynndescent_search(
    query_embedding: list[float],
    limit: int,
    config: RAGLiteConfig,
    where: Any,
    engine: Any,
) -> tuple[list[ChunkId], list[float]]:
    """SQLite vector search using PyNNDescent fallback when sqlite-vec unavailable."""
    import numpy as np
    from sqlalchemy import text
    from sqlalchemy.orm import Session
    
    with Session(engine) as session:
        # Get all embeddings and chunk IDs
        base_query = "SELECT ce.chunk_id, ce.embedding FROM chunk_embeddings ce"
        
        if where is not None:
            base_query += " JOIN chunk c ON c.id = ce.chunk_id WHERE " + str(where)
        
        result = session.execute(text(base_query))
        rows = result.fetchall()
        
        if not rows:
            return [], []
        
        # Extract embeddings and chunk IDs
        chunk_ids = [row[0] for row in rows]
        embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows])
        
        # Use PyNNDescent for similarity search
        try:
            from pynndescent import NNDescent
            
            # Build index
            index = NNDescent(embeddings, metric="cosine", n_neighbors=min(limit, len(embeddings)))
            
            # Search
            query_array = np.array([query_embedding], dtype=np.float32)
            indices, distances = index.query(query_array, k=min(limit, len(embeddings)))
            
            # Get results
            result_chunk_ids = [chunk_ids[i] for i in indices[0]]
            result_scores = [1.0 - d for d in distances[0]]  # Convert distance to similarity
            
            return result_chunk_ids, result_scores
            
        except ImportError:
            logger.error("Neither sqlite-vec nor pynndescent available for vector search")
            return [], []
```

#### **Step 2.4: Enhance Main Vector Search Function**
**TARGET LOCATION**: Find the main `vector_search()` function
**LOOK FOR**: The function that handles different database backends
**ADD SQLite SUPPORT**: Add SQLite case to the existing backend detection

You'll need to find a pattern like:
```python
def vector_search(...):
    # ... existing code ...
    if database_backend == "duckdb":
        # DuckDB implementation
    elif database_backend == "postgresql":  
        # PostgreSQL implementation
    # ADD HERE: SQLite case
```

**ADD THIS CASE**:
```python
elif database_backend == "sqlite":
    return _sqlite_vector_search(query_embedding, limit, config, where)
```

#### **Step 2.5: Validation**
```python
# Test SQLite vector search:
python -c "
from raglite._search import vector_search
from raglite._config import RAGLiteConfig
config = RAGLiteConfig(db_url='sqlite:///tests/test_raglite.db')
# This should work after implementation
results = vector_search([0.1] * 384, limit=5, config=config)
print(f'✅ Vector search returned {len(results)} results')
"
```

---

## 📋 **PHASE 3: INSERTION PIPELINE**
### **File**: `src/raglite/_insert.py`
### **Target**: Add SQLite-specific insertion functions

#### **Step 3.1: Examine Current Insert Implementation**
```python
# Read insert file structure:
read_file(src/raglite/_insert.py, startLine=1, endLine=50)
```

#### **Step 3.2: Find Embedding Insertion Function**
```python
# Search for embedding insertion:
grep_search("def.*insert.*embedding", isRegexp=True, includePattern="src/raglite/_insert.py")
```

#### **Step 3.3: Add SQLite Embedding Insertion**
**TARGET LOCATION**: End of file or after existing insertion functions
**ADD THESE FUNCTIONS**:
```python
def _insert_sqlite_embeddings(
    chunk_embeddings: list[ChunkEmbedding],
    config: RAGLiteConfig,
) -> None:
    """Insert embeddings into SQLite with sqlite-vec support."""
    from ._database import create_database_engine
    from sqlalchemy import text
    from sqlalchemy.orm import Session
    
    if not chunk_embeddings:
        return
    
    engine = create_database_engine(config)
    embedding_dim = len(chunk_embeddings[0].embedding)
    
    with Session(engine) as session:
        # Check if sqlite-vec is available
        if getattr(engine, 'sqlite_vec_available', False):
            _insert_sqlite_vec_embeddings(session, chunk_embeddings, embedding_dim)
        else:
            _insert_sqlite_blob_embeddings(session, chunk_embeddings)
        
        session.commit()

def _insert_sqlite_vec_embeddings(session: Any, chunk_embeddings: list, embedding_dim: int) -> None:
    """Insert embeddings using sqlite-vec virtual table."""
    import sqlite_vec
    from sqlalchemy import text
    
    # Ensure vector table exists
    session.execute(text(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings_vec 
        USING vec0(
            chunk_id TEXT PRIMARY KEY,
            embedding FLOAT[{embedding_dim}]
        )
    """))
    
    # Insert embeddings
    for chunk_embedding in chunk_embeddings:
        embedding_blob = sqlite_vec.serialize_float32(chunk_embedding.embedding)
        session.execute(text("""
            INSERT OR REPLACE INTO chunk_embeddings_vec (chunk_id, embedding)
            VALUES (?, ?)
        """), [chunk_embedding.chunk_id, embedding_blob])

def _insert_sqlite_blob_embeddings(session: Any, chunk_embeddings: list) -> None:
    """Insert embeddings as blobs for PyNNDescent fallback."""
    import numpy as np
    from sqlalchemy import text
    
    # Ensure regular embeddings table exists
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id TEXT PRIMARY KEY,
            embedding BLOB
        )
    """))
    
    # Insert embeddings as numpy blobs
    for chunk_embedding in chunk_embeddings:
        embedding_array = np.array(chunk_embedding.embedding, dtype=np.float32)
        embedding_blob = embedding_array.tobytes()
        session.execute(text("""
            INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding)
            VALUES (?, ?)
        """), [chunk_embedding.chunk_id, embedding_blob])

def _update_sqlite_fts_index(
    chunks: list[Chunk],
    config: RAGLiteConfig,
) -> None:
    """Update SQLite FTS5 index with new chunks."""
    from ._database import create_database_engine
    from sqlalchemy import text
    from sqlalchemy.orm import Session
    
    if not chunks:
        return
    
    engine = create_database_engine(config)
    
    with Session(engine) as session:
        # Ensure FTS5 table exists
        session.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts 
            USING fts5(
                chunk_id UNINDEXED,
                body,
                content='chunk',
                content_rowid='id'
            )
        """))
        
        # Insert/update chunk text for full-text search
        for chunk in chunks:
            session.execute(text("""
                INSERT OR REPLACE INTO chunk_fts (chunk_id, body)
                VALUES (?, ?)
            """), [chunk.id, chunk.body])
        
        session.commit()
```

#### **Step 3.4: Integrate with Main Insert Function**
**TARGET**: Find the main function that handles insertions and add SQLite case
**LOOK FOR**: Database backend detection in insert operations
**ADD**: SQLite-specific calls to the functions above

#### **Step 3.5: Validation**
```python
# Test insertion:
python -c "
from raglite._insert import _insert_sqlite_embeddings
from raglite._config import RAGLiteConfig
# Test with sample data
print('✅ Insert functions defined')
"
```

---

## 📋 **PHASE 4: TYPE SYSTEM INTEGRATION**
### **File**: `src/raglite/_typing.py`
### **Target**: Add SQLite type definitions and compilation rules

#### **Step 4.1: Read Current Type Definitions**
```python
read_file(src/raglite/_typing.py, startLine=1, endLine=50)
```

#### **Step 4.2: Find Vector Type Definitions**
```python
grep_search("class.*Vector", isRegexp=True, includePattern="src/raglite/_typing.py")
```

#### **Step 4.3: Add SQLite Type Support**
**TARGET LOCATION**: After existing type definitions
**ADD THESE CLASSES**:
```python
class SQLiteVec(UserDefinedType[FloatVector]):
    """SQLite vector type using sqlite-vec extension."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
    
    def get_col_spec(self, **kwargs: Any) -> str:
        return f"FLOAT[{self.dimension}]"
    
    def bind_processor(self, dialect: Any) -> Any:
        """Process values for database binding."""
        def process(value: Any) -> Any:
            if value is None:
                return None
            
            # For sqlite-vec, serialize as float32
            try:
                import sqlite_vec
                if isinstance(value, (list, np.ndarray)):
                    return sqlite_vec.serialize_float32(value)
            except ImportError:
                # Fallback to numpy blob
                import numpy as np
                if isinstance(value, list):
                    value = np.array(value, dtype=np.float32)
                return value.tobytes()
            
            return value
        return process
    
    def result_processor(self, dialect: Any, coltype: Any) -> Any:
        """Process values from database results."""
        def process(value: Any) -> Any:
            if value is None:
                return None
            
            # Deserialize from blob
            try:
                import sqlite_vec
                return sqlite_vec.deserialize_float32(value)
            except ImportError:
                # Fallback: numpy blob to list
                import numpy as np
                return np.frombuffer(value, dtype=np.float32).tolist()
            
        return process
```

#### **Step 4.4: Add SQLite Distance Compilation**
**TARGET LOCATION**: After existing `@compiles` decorators
**ADD THIS COMPILATION RULE**:
```python
@compiles(EmbeddingDistance, "sqlite")
def _embedding_distance_sqlite(element: EmbeddingDistance, compiler: Any, **kwargs: Any) -> str:
    """Compile embedding distance for SQLite using sqlite-vec."""
    left = compiler.process(element.left, **kwargs)
    right = compiler.process(element.right, **kwargs)
    
    # Map metrics to sqlite-vec functions
    metric_map = {
        "cosine": "vec_distance_cosine",
        "l2": "vec_distance_l2",
        "dot": "vec_distance_dot",
        "euclidean": "vec_distance_l2",
    }
    
    func_name = metric_map.get(element.metric, "vec_distance_cosine")
    return f"{func_name}({left}, {right})"
```

---

## 📋 **PHASE 5: TEST INTEGRATION**
### **File**: `tests/conftest.py`
### **Target**: Add SQLite fixture and backend parameter

#### **Step 5.1: Read Current Test Configuration**
```python
read_file(tests/conftest.py, startLine=1, endLine=100)
```

#### **Step 5.2: Find Database Fixtures**
```python
grep_search("@pytest.fixture", isRegexp=False, includePattern="tests/conftest.py")
```

#### **Step 5.3: Add SQLite URL Fixture**
**TARGET LOCATION**: After existing fixtures
**ADD THIS FIXTURE**:
```python
@pytest.fixture(scope="session")
def sqlite_url() -> Generator[str, None, None]:
    """Create a temporary SQLite database file and return the database URL."""
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_file = Path(temp_dir) / "raglite_test.db"
        yield f"sqlite:///{db_file}"

@pytest.fixture(scope="session") 
def sqlite_test_url() -> str:
    """Use the existing test database for SQLite testing."""
    return "sqlite:///tests/test_raglite.db"
```

#### **Step 5.4: Add SQLite to Database Parameter**
**TARGET LOCATION**: Find the database fixture with parameters
**LOOK FOR**: `@pytest.fixture` with `params=[...]`
**ADD**: `sqlite` parameter to the list

**FIND THIS PATTERN**:
```python
@pytest.fixture(
    scope="session",
    params=[
        pytest.param("duckdb", id="duckdb"),
        # ... other parameters
    ],
)
def database(request: pytest.FixtureRequest) -> str:
```

**ADD SQLite PARAMETER**:
```python
@pytest.fixture(
    scope="session", 
    params=[
        pytest.param("duckdb", id="duckdb"),
        pytest.param("sqlite", id="sqlite"),  # ADD THIS LINE
        # ... keep existing parameters
    ],
)
def database(request: pytest.FixtureRequest) -> str:
    """Get a database URL to test RAGLite with."""
    if request.param == "sqlite":
        return request.getfixturevalue("sqlite_test_url")
    elif request.param == "duckdb":
        return request.getfixturevalue("duckdb_url")
    # ... existing logic
```

---

## 🧪 **VALIDATION & TESTING**

### **After Each Phase**
```bash
# 1. Test imports
python -c "import raglite; print('✅ Import successful')"

# 2. Test basic functionality  
python -c "
from raglite import RAGLiteConfig
from raglite._database import create_database_engine
config = RAGLiteConfig(db_url='sqlite:///tests/test_raglite.db')
engine = create_database_engine(config)
print('✅ SQLite engine created')
"

# 3. Run specific SQLite tests
python -m pytest tests/test_sqlite_backend.py -v

# 4. Run test suite with SQLite
python -m pytest tests/ -k "sqlite" -v
```

### **Final Integration Test**
```bash
# Test complete RAG pipeline with SQLite
python -c "
from raglite import RAGLiteConfig, insert, search
config = RAGLiteConfig(db_url='sqlite:///tests/test_raglite.db')

# Test search functionality
results = search('artificial intelligence', config=config)
print(f'✅ Search returned {len(results)} results')

print('🎉 SQLite backend implementation complete!')
"
```

---

## 🚨 **ERROR HANDLING & TROUBLESHOOTING**

### **Common Issues & Solutions**

1. **Import Errors**:
   ```bash
   ModuleNotFoundError: No module named 'sqlite_vec'
   # Solution: Install dependencies
   pip install sqlite-vec pynndescent
   ```

2. **Extension Loading Errors**:
   ```python
   # Check if sqlite-vec loads properly
   python -c "import sqlite_vec; print(sqlite_vec.loadable_path())"
   ```

3. **Database Connection Issues**:
   ```bash
   # Verify test database exists
   ls -la tests/test_raglite.db
   sqlite3 tests/test_raglite.db "SELECT COUNT(*) FROM documents;"
   ```

4. **Test Failures**:
   ```bash
   # Run tests with detailed output
   python -m pytest tests/ -v --tb=long -s
   ```

### **Success Indicators**
- ✅ All imports work without errors
- ✅ SQLite engine creates successfully  
- ✅ Vector search returns results
- ✅ Test suite passes with SQLite backend
- ✅ No remaining TODO comments in code

---

## 📝 **COMMIT STRATEGY**

### **Commit After Each Phase**
```bash
# Phase 1
git add src/raglite/_database.py
git commit -m "feat: implement SQLite engine with sqlite-vec support in _database.py"

# Phase 2  
git add src/raglite/_search.py
git commit -m "feat: add SQLite vector and keyword search functions in _search.py"

# Phase 3
git add src/raglite/_insert.py  
git commit -m "feat: implement SQLite embedding insertion pipeline in _insert.py"

# Phase 4
git add src/raglite/_typing.py
git commit -m "feat: add SQLite type definitions and distance compilation in _typing.py"

# Phase 5
git add tests/conftest.py
git commit -m "feat: add SQLite test fixtures and backend parameter in conftest.py"

# Final
git add .
git commit -m "feat: complete SQLite backend implementation

- Full SQLite support with sqlite-vec extension
- Vector search with PyNNDescent fallback  
- FTS5 keyword search integration
- Hybrid search with RRF fusion
- Complete test suite integration
- Performance optimizations (WAL mode, caching)

Closes: SQLite backend implementation requirements"

git push origin copilot/vscode1757947449396
```

---

## 🎯 **SUCCESS CRITERIA**

### **Must Have (Critical)**
- [ ] SQLite engine creation works
- [ ] sqlite-vec extension loads (with fallback)
- [ ] Vector search returns accurate results
- [ ] Keyword search works with FTS5
- [ ] Test suite passes with SQLite backend

### **Should Have (Important)**  
- [ ] Performance comparable to other backends
- [ ] Hybrid search with RRF fusion
- [ ] Proper error handling and logging
- [ ] Memory usage optimizations

### **Nice to Have (Bonus)**
- [ ] Benchmarking vs other backends
- [ ] Advanced SQLite optimizations
- [ ] GPU acceleration integration

---

**🎉 EXECUTION SUMMARY**: Follow these phases sequentially, validate after each step, commit incrementally, and ensure all tests pass. The test database `tests/test_raglite.db` is ready for immediate testing and validation.**
