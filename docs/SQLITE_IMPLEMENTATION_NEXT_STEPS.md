# SQLite Implementation Next Steps Action Plan

## 🎯 Immediate Actions (Today - September 15, 2025)

### 1. **PR Status Correction** - HIGH PRIORITY
```bash
# Update PR title and description
gh pr edit 1 \
  --title "📋 SQLite Implementation Planning Document" \
  --body "This PR adds comprehensive planning documentation for SQLite support implementation. **Note**: Actual implementation is pending and will be addressed in follow-up PRs."
```

### 2. **Dependency Resolution** - CRITICAL
```bash
# Install missing dependencies to enable testing
pip install rerankers[api,flashrank]
pip install -e .[dev,test]

# Verify imports work
python -c "import raglite; print('✅ Imports successful')"
```

### 3. **Current State Verification**
```bash
# Test existing functionality
python -m pytest tests/ -v --tb=short

# Verify database backends work
python -c "from raglite._database import create_database_engine; print('✅ Database engine creation works')"
```

---

## 📅 Implementation Timeline (September 16-20, 2025)

### **Day 1: Foundation Setup** (September 16)
**Goal**: Establish SQLite foundation and basic connectivity

#### Tasks:
- [ ] Add SQLite dependencies to `pyproject.toml`
- [ ] Update default configuration in `_config.py`
- [ ] Implement basic SQLite engine detection in `_database.py`
- [ ] Create SQLite connection optimization
- [ ] Test basic SQLite connectivity

#### Deliverables:
- SQLite engine can be created
- Basic table operations work
- Configuration defaults updated

#### Testing:
```bash
# Test SQLite engine creation
python -c "from raglite._database import create_database_engine; engine = create_database_engine('sqlite:///test.db'); print('✅ SQLite engine created')"
```

---

### **Day 2-3: Core Search Implementation** (September 17-18)
**Goal**: Implement SQLite search capabilities

#### Tasks:
- [ ] Implement sqlite-vec vector search in `_search.py`
- [ ] Add FTS5 keyword search functionality
- [ ] Integrate hybrid search with Reciprocal Rank Fusion
- [ ] Add fallback mechanisms for missing extensions
- [ ] Update insertion pipeline in `_insert.py`

#### Deliverables:
- Vector search returns accurate results
- Keyword search works with FTS5
- Hybrid search combines results properly
- Graceful fallbacks implemented

#### Testing:
```python
# Test search functionality
from raglite._search import vector_search
results = vector_search("test query", config=sqlite_config)
assert len(results) > 0
```

---

### **Day 4: Integration & Testing** (September 19)
**Goal**: Full integration and comprehensive testing

#### Tasks:
- [ ] Update test fixtures in `conftest.py`
- [ ] Add SQLite-specific test cases
- [ ] Update documentation in `README.md`
- [ ] Create migration utilities
- [ ] Performance optimization

#### Deliverables:
- All tests pass with SQLite backend
- Documentation updated
- Migration tools available
- Performance benchmarks met

#### Testing:
```bash
# Run full test suite with SQLite
python -m pytest tests/ -k sqlite -v
python -m pytest tests/ --cov=raglite --cov-report=term
```

---

### **Day 5: Validation & Documentation** (September 20)
**Goal**: Final validation and documentation completion

#### Tasks:
- [ ] Cross-platform testing (Windows, macOS, Linux)
- [ ] Performance benchmarking against DuckDB/PostgreSQL
- [ ] Documentation completion
- [ ] Final integration testing
- [ ] Release preparation

#### Deliverables:
- Comprehensive test coverage
- Performance benchmarks documented
- Migration guide completed
- Ready for production release

---

## 🔧 Technical Implementation Details

### Phase 1: Dependencies & Configuration

#### `pyproject.toml` Updates:
```toml
dependencies = [
    # ... existing dependencies ...
    "sqlite-vec>=0.1.0",
    "pynndescent>=0.5.12",
]
```

#### `_config.py` Updates:
```python
# Change default from DuckDB to SQLite
db_url: str | URL = f"sqlite:///{(cache_path / 'raglite.db').as_posix()}"
```

### Phase 2: Database Engine Implementation

#### `_database.py` Key Changes:
```python
def create_database_engine(config: RAGLiteConfig) -> Engine:
    """Create database engine with SQLite support."""
    db_url = make_url(str(config.db_url))

    if db_url.drivername == "sqlite":
        # SQLite-specific optimizations
        connect_args = {
            "check_same_thread": False,
            "timeout": 30
        }

        engine = create_engine(
            config.db_url,
            connect_args=connect_args,
            poolclass=StaticPool
        )

        # Apply SQLite performance pragmas
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode = WAL"))
            conn.execute(text("PRAGMA synchronous = NORMAL"))
            conn.execute(text("PRAGMA cache_size = 10000"))
            conn.commit()

        return engine
    # ... existing DuckDB/PostgreSQL logic
```

### Phase 3: Search Implementation

#### `_search.py` Key Changes:
```python
def _sqlite_vector_search(
    query_embedding: List[float],
    limit: int,
    config: RAGLiteConfig
) -> tuple[list[ChunkId], list[float]]:
    """SQLite vector search with sqlite-vec."""
    engine = create_database_engine(config)

    with Session(engine) as session:
        try:
            # Try sqlite-vec first
            result = session.execute(text("""
                SELECT chunk_id, vec_distance_cosine(embedding, ?) as distance
                FROM chunk_embeddings_vec
                ORDER BY distance
                LIMIT ?
            """), [query_embedding, limit])

        except Exception:
            # Fallback to manual distance calculation
            result = session.execute(text("""
                SELECT chunk_id,
                       1 - (embedding <=> ?) as similarity
                FROM chunk_embeddings
                ORDER BY similarity DESC
                LIMIT ?
            """), [query_embedding, limit])

        return [row[0] for row in result], [row[1] for row in result]
```

### Phase 4: Type System Updates

#### `_typing.py` Key Changes:
```python
# SQLite vector distance compilation
@compiles(EmbeddingDistance, "sqlite")
def _embedding_distance_sqlite(element, compiler, **kwargs):
    """Compile embedding distance for SQLite."""
    if hasattr(element, 'metric') and element.metric == 'cosine':
        return f"vec_distance_cosine({element.left}, {element.right})"
    return f"1 - ({element.left} <=> {element.right})"

# SQLite vector type
class SQLiteVec(UserDefinedType):
    """SQLite vector type for sqlite-vec extension."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def get_col_spec(self, **kwargs):
        return f"FLOAT[{self.dimension}]"
```

---

## 🧪 Testing Strategy

### Unit Tests
```python
class TestSQLiteBackend:
    def test_sqlite_engine_creation(self, sqlite_config):
        """Test SQLite engine can be created."""
        engine = create_database_engine(sqlite_config)
        assert engine is not None

    def test_sqlite_vector_search(self, sqlite_config, sample_embeddings):
        """Test vector search functionality."""
        results = vector_search("test query", config=sqlite_config)
        assert len(results) > 0

    def test_sqlite_keyword_search(self, sqlite_config, sample_documents):
        """Test keyword search functionality."""
        results = keyword_search("test", config=sqlite_config)
        assert len(results) > 0
```

### Integration Tests
```python
def test_sqlite_full_workflow(sqlite_config):
    """Test complete RAG workflow with SQLite."""
    # Insert documents
    insert_documents(documents, config=sqlite_config)

    # Search
    results = rag("What is machine learning?", config=sqlite_config)

    # Verify results
    assert len(results) > 0
    assert all("score" in result for result in results)
```

### Performance Tests
```python
def test_sqlite_performance_benchmarks(sqlite_config, benchmark_documents):
    """Performance benchmarks for SQLite operations."""

    # Vector search benchmark
    start_time = time.time()
    results = vector_search("benchmark query", config=sqlite_config)
    vector_time = time.time() - start_time
    assert vector_time < 2.0  # Less than 2 seconds

    # Keyword search benchmark
    start_time = time.time()
    results = keyword_search("benchmark", config=sqlite_config)
    keyword_time = time.time() - start_time
    assert keyword_time < 0.5  # Less than 500ms
```

---

## 📊 Success Criteria Checklist

### Functional Requirements ✅/❌
- [ ] SQLite database engine successfully created
- [ ] Vector search returns accurate results
- [ ] Keyword search matches expected behavior
- [ ] Hybrid search properly fuses results
- [ ] All existing tests pass with SQLite backend

### Performance Requirements ✅/❌
- [ ] Vector search completes within 2 seconds for 10K documents
- [ ] Keyword search completes within 500ms
- [ ] Hybrid search maintains sub-2-second response time
- [ ] Memory usage remains below 1GB for typical workloads

### Compatibility Requirements ✅/❌
- [ ] Seamless switching between database backends
- [ ] Existing DuckDB/PostgreSQL functionality unchanged
- [ ] Configuration migration tools available
- [ ] Documentation updated and accurate

---

## 🚨 Risk Mitigation Plan

### Technical Risks
1. **sqlite-vec Extension Unavailable**
   - **Detection**: Test extension loading on startup
   - **Fallback**: Implement manual distance calculations
   - **User Guidance**: Clear error messages with installation instructions

2. **Performance Degradation**
   - **Monitoring**: Implement performance benchmarks
   - **Optimization**: WAL mode, optimized pragmas, connection pooling
   - **Comparison**: Benchmark against DuckDB/PostgreSQL baselines

3. **Platform Compatibility**
   - **Testing**: Validate on Windows, macOS, Linux
   - **CI/CD**: Add cross-platform testing to GitHub Actions
   - **Documentation**: Platform-specific installation guides

### Project Risks
1. **Timeline Delays**
   - **Planning**: Break implementation into daily milestones
   - **Tracking**: Daily progress reviews and adjustments
   - **Communication**: Regular status updates

2. **Breaking Changes**
   - **Compatibility**: Maintain backward compatibility
   - **Migration**: Provide migration utilities and guides
   - **Testing**: Comprehensive regression testing

---

## 📞 Support & Resources

### Development Resources
- **sqlite-vec Documentation**: https://github.com/asg017/sqlite-vec
- **SQLite FTS5 Guide**: https://www.sqlite.org/fts5.html
- **RAGLite Architecture**: Review existing DuckDB/PostgreSQL implementations

### Testing Resources
- **Existing Tests**: Use as templates for SQLite tests
- **Performance Benchmarks**: Compare against DuckDB baselines
- **Integration Testing**: Full workflow validation

### Documentation Resources
- **README Updates**: Follow existing format and style
- **Migration Guide**: Step-by-step user instructions
- **Troubleshooting**: Common issues and solutions

---

## 🎯 Final Deliverables

### Code Changes
- [ ] 8 core files modified with SQLite support
- [ ] Comprehensive test coverage added
- [ ] Performance optimizations implemented
- [ ] Migration utilities created

### Documentation
- [ ] README.md updated with SQLite information
- [ ] Installation guide with SQLite-specific instructions
- [ ] Migration guide for existing users
- [ ] Troubleshooting documentation

### Testing
- [ ] Unit tests for all SQLite functionality
- [ ] Integration tests for full workflows
- [ ] Performance benchmarks validated
- [ ] Cross-platform compatibility verified

### Validation
- [ ] All success criteria met
- [ ] Existing functionality preserved
- [ ] Performance requirements satisfied
- [ ] Documentation complete and accurate

---

**Action Plan Created**: September 15, 2025
**Implementation Start**: September 16, 2025
**Expected Completion**: September 20, 2025
**Total Effort**: 5 days

**Status**: Ready for implementation</content>
<parameter name="filePath">f:\_Divorce_2025\@.tools\raglite\docs\SQLITE_IMPLEMENTATION_NEXT_STEPS.md
