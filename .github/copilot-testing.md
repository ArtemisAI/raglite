# GitHub Copilot Testing Framework for Raglite

## Test Suite Structure

### Directory Organization
```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── test_database.py           # Database operations tests
├── test_embed.py              # Embedding functionality tests
├── test_extract.py            # Text extraction tests
├── test_insert.py             # Data insertion tests
├── test_rag.py                # Main RAG pipeline tests
├── test_search.py             # Search algorithm tests
├── test_split_chunks.py       # Text chunking tests
├── test_split_sentences.py    # Sentence splitting tests
├── test_markdown.py           # Markdown processing tests
├── integration/               # Integration tests
│   ├── test_full_pipeline.py
│   └── test_api_endpoints.py
├── performance/               # Performance benchmarks
│   ├── test_benchmarks.py
│   └── benchmark_results.json
└── fixtures/                  # Test data fixtures
    ├── sample_documents.json
    └── test_embeddings.npy
```

## Testing Protocols for GitHub Copilot

### Test Naming Conventions
- **Unit Tests**: `test_function_name.py` or `test_class_name.py`
- **Integration Tests**: `test_integration_feature.py`
- **Performance Tests**: `test_performance_feature.py`
- **Test Functions**: `test_should_do_something_specific`

### Test Categories
1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Benchmark critical operations
4. **Regression Tests**: Prevent reintroduction of bugs

## Test Implementation Guidelines

### Basic Test Structure
```python
import pytest
from raglite import RagliteDB

class TestDatabaseOperations:
    """Test database operations functionality."""

    def test_insert_document(self, test_db):
        """Test inserting a document into the database."""
        # Arrange
        doc = {"content": "Test document", "metadata": {"source": "test"}}

        # Act
        result = test_db.insert_document(doc)

        # Assert
        assert result is not None
        assert result["id"] is not None

    def test_search_documents(self, test_db, sample_documents):
        """Test searching documents with embeddings."""
        # Arrange
        test_db.insert_documents(sample_documents)

        # Act
        results = test_db.search("test query", limit=5)

        # Assert
        assert len(results) <= 5
        assert all("score" in result for result in results)
```

### Async Test Support
```python
import pytest_asyncio

class TestAsyncOperations:
    """Test asynchronous operations."""

    @pytest.mark.asyncio
    async def test_async_embedding_generation(self):
        """Test asynchronous embedding generation."""
        # Arrange
        texts = ["Test document 1", "Test document 2"]

        # Act
        embeddings = await generate_embeddings_async(texts)

        # Assert
        assert len(embeddings) == 2
        assert all(len(emb) == 384 for emb in embeddings)  # Assuming 384-dim embeddings
```

## Performance Benchmarks

### Benchmark Categories
1. **Text Processing Benchmarks**
2. **Embedding Generation Benchmarks**
3. **Database Operation Benchmarks**
4. **Search Performance Benchmarks**
5. **Memory Usage Benchmarks**

### Benchmark Implementation
```python
import time
import pytest
import numpy as np

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_embedding_generation_performance(self, benchmark_documents):
        """Benchmark embedding generation performance."""
        start_time = time.perf_counter()

        # Generate embeddings for benchmark documents
        embeddings = generate_embeddings(benchmark_documents)

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Assert performance requirements
        assert duration < 5.0  # Less than 5 seconds for benchmark
        assert len(embeddings) == len(benchmark_documents)

    def test_vector_search_performance(self, test_db, benchmark_vectors):
        """Benchmark vector search performance."""
        # Insert benchmark vectors
        test_db.insert_vectors(benchmark_vectors)

        start_time = time.perf_counter()

        # Perform search
        results = test_db.search_vectors(np.random.rand(384), limit=10)

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Assert performance requirements
        assert duration < 0.1  # Less than 100ms
        assert len(results) == 10
```

## Error Detection and Recovery

### Error Handling Tests
```python
class TestErrorHandling:
    """Test error handling and recovery."""

    def test_database_connection_error(self):
        """Test handling of database connection errors."""
        with pytest.raises(DatabaseConnectionError):
            db = RagliteDB("invalid_connection_string")
            db.connect()

    def test_embedding_generation_error(self):
        """Test handling of embedding generation errors."""
        with pytest.raises(EmbeddingGenerationError):
            generate_embeddings([""])  # Empty text should raise error

    def test_invalid_search_query(self, test_db):
        """Test handling of invalid search queries."""
        with pytest.raises(InvalidQueryError):
            test_db.search("", limit=0)  # Invalid limit
```

### Recovery Testing
```python
class TestRecoveryScenarios:
    """Test system recovery from various failure scenarios."""

    def test_database_recovery_after_corruption(self, test_db):
        """Test database recovery after corruption."""
        # Simulate corruption
        # Test recovery mechanism
        # Verify data integrity
        pass

    def test_embedding_service_fallback(self):
        """Test fallback when primary embedding service fails."""
        # Simulate service failure
        # Test fallback mechanism
        # Verify continued operation
        pass
```

## Test Fixtures and Mocking

### Common Fixtures
```python
# tests/conftest.py
import pytest
from unittest.mock import Mock
from raglite import RagliteDB

@pytest.fixture
def test_db():
    """Provide a test database instance."""
    db = RagliteDB(":memory:")
    yield db
    db.close()

@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        {"content": "Python is a programming language", "metadata": {"topic": "programming"}},
        {"content": "Machine learning is a subset of AI", "metadata": {"topic": "AI"}},
        {"content": "Databases store structured data", "metadata": {"topic": "database"}}
    ]

@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    mock_service = Mock()
    mock_service.generate_embeddings.return_value = [[0.1] * 384, [0.2] * 384]
    return mock_service
```

## Test Coverage Requirements

### Coverage Goals
- **Overall Coverage**: >90%
- **Core Modules**: >95%
- **Database Module**: >95%
- **Search Module**: >95%
- **API Endpoints**: >90%

### Coverage Configuration
```ini
# .coveragerc
[run]
source = raglite
omit =
    */tests/*
    */venv/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod
```

## Continuous Integration Testing

### GitHub Actions Test Workflow
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[test]
    - name: Run tests
      run: |
        pytest --cov=raglite --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Test Data Management

### Synthetic Data Generation
```python
def generate_test_documents(count: int = 100) -> List[Dict[str, Any]]:
    """Generate synthetic test documents."""
    documents = []
    for i in range(count):
        doc = {
            "content": f"Test document {i} with some content for testing purposes.",
            "metadata": {
                "id": i,
                "category": f"category_{i % 5}",
                "timestamp": f"2024-01-{i % 28 + 1:02d}"
            }
        }
        documents.append(doc)
    return documents
```

## Debugging and Troubleshooting

### Test Debugging Tips
1. **Use verbose output**: `pytest -v -s`
2. **Debug specific test**: `pytest -k "test_name" --pdb`
3. **Check coverage**: `pytest --cov=raglite --cov-report=html`
4. **Profile performance**: `pytest --profile`
5. **Run with warnings**: `pytest -W error`

### Common Test Issues
- **Flaky Tests**: Use retry mechanisms or stabilize test data
- **Slow Tests**: Optimize setup/teardown or use fixtures properly
- **Memory Issues**: Clean up resources in teardown
- **Async Issues**: Ensure proper event loop management
