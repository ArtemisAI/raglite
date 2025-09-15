# GitHub Copilot Configuration for Raglite

## Repository Context
**Project**: Raglite - High-performance RAG library for Python
**Owner**: ArtemisAI
**Language**: Python 3.11+
**License**: MIT

### Project Objectives
- Provide efficient text processing and embedding capabilities
- Support multiple vector databases (SQLite, PostgreSQL, etc.)
- Enable fast semantic search and retrieval
- Offer CLI tools for data management
- Maintain high performance with low latency

### Technology Stack
- **Core Language**: Python 3.11+
- **Key Libraries**:
  - `numpy` - Numerical computing
  - `sqlite3` - Database operations
  - `sentence-transformers` - Text embeddings
  - `openai` - LLM integration
  - `click` - CLI framework
  - `pydantic` - Data validation
- **Testing**: pytest with comprehensive test coverage
- **Documentation**: Markdown-based with examples

### Performance Benchmarks Achieved
- Text chunking: < 100ms for 10KB documents
- Embedding generation: < 500ms for 1000 tokens
- Vector search: < 50ms for 100K vectors
- Memory usage: < 500MB for 1M vectors

## Current Development Context
**Active Branch**: main
**Focus Areas**:
- SQLite implementation optimization
- Performance benchmarking improvements
- API enhancements for better usability
- Testing framework expansion
- Documentation completion

### Key Files to Focus On
- `src/raglite/_database.py` - Database abstraction layer
- `src/raglite/_embed.py` - Embedding operations
- `src/raglite/_search.py` - Search algorithms
- `src/raglite/_rag.py` - Main RAG pipeline
- `tests/` - Comprehensive test suite

### Implementation Requirements
- **Type Safety**: Full mypy compatibility
- **Error Handling**: Comprehensive exception handling
- **Performance**: Optimized database queries and embeddings
- **Testing**: >90% code coverage
- **Documentation**: Complete API documentation

## Technical Dependencies
- Python 3.11+ with typing support
- SQLite 3.35+ for vector operations
- Sentence transformers models
- OpenAI API access (optional)
- Development tools: pytest, mypy, black, isort

## Development Environment Details
- **IDE**: VS Code with Python extensions
- **Version Control**: Git with GitHub
- **CI/CD**: GitHub Actions
- **Package Management**: pip with pyproject.toml
- **Virtual Environment**: Recommended for development
