# API Contract: InstallationValidator

**Interface**: `InstallationValidator`  
**Purpose**: Validate and verify successful installation of RAGLite dependencies  
**Integration**: Works with DependencyResolution to ensure proper installation

## Interface Definition

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from .data_model import (
    SystemInfo, GPUInfo, DependencyResolution, 
    ValidationResults, FeatureSet, ValidationLevel
)

class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    CRITICAL = "critical"    # Installation will not work
    WARNING = "warning"      # Installation may have issues
    INFO = "info"           # Informational only
    SUCCESS = "success"     # Validation passed

class ValidationResult:
    """Individual validation check result"""
    
    def __init__(
        self, 
        component: str,
        severity: ValidationSeverity,
        message: str,
        details: Optional[Dict] = None,
        fix_suggestion: Optional[str] = None
    ):
        self.component = component
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.fix_suggestion = fix_suggestion

class InstallationValidator(ABC):
    """Abstract interface for installation validation"""
    
    @abstractmethod
    def validate_complete_installation(
        self, 
        system_info: SystemInfo,
        gpu_info: GPUInfo, 
        resolution: DependencyResolution,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> ValidationResults:
        """
        Perform comprehensive installation validation.
        
        Args:
            system_info: System environment details
            gpu_info: GPU hardware configuration
            resolution: Installed dependency resolution
            validation_level: Depth of validation to perform
            
        Returns:
            Complete validation results with all component status
            
        Validation Scope:
            - Python environment setup
            - Core dependencies installation
            - GPU acceleration setup  
            - Optional feature functionality
            - RAGLite import and basic operations
        """
        pass
    
    @abstractmethod
    def validate_python_environment(self, system_info: SystemInfo) -> List[ValidationResult]:
        """
        Validate Python environment requirements.
        
        Args:
            system_info: System Python configuration
            
        Returns:
            List of Python environment validation results
            
        Validation Checks:
            - Python version compatibility (>=3.10)
            - pip installation and version
            - Virtual environment detection
            - Required system libraries (SSL, etc.)
        """
        pass
    
    @abstractmethod
    def validate_core_dependencies(self, resolution: DependencyResolution) -> List[ValidationResult]:
        """
        Validate core RAGLite dependencies.
        
        Args:
            resolution: Dependency resolution to validate
            
        Returns:
            List of core dependency validation results
            
        Validation Components:
            - raglite package importability
            - sqlite-vec extension functionality
            - numpy/scipy mathematical operations
            - sentence-transformers embedding generation
        """
        pass
    
    @abstractmethod
    def validate_gpu_acceleration(
        self, 
        gpu_info: GPUInfo, 
        resolution: DependencyResolution
    ) -> List[ValidationResult]:
        """
        Validate GPU acceleration setup.
        
        Args:
            gpu_info: GPU hardware configuration
            resolution: GPU-specific dependency resolution
            
        Returns:
            List of GPU acceleration validation results
            
        Validation Checks:
            - PyTorch GPU availability and functionality
            - CUDA/ROCm/Metal runtime accessibility  
            - llama-cpp-python GPU acceleration
            - VRAM allocation and memory management
            - GPU inference speed benchmarking
        """
        pass
    
    @abstractmethod
    def validate_optional_features(
        self, 
        features: Set[FeatureSet], 
        resolution: DependencyResolution
    ) -> List[ValidationResult]:
        """
        Validate optional feature installations.
        
        Args:
            features: Selected feature set
            resolution: Complete dependency resolution
            
        Returns:
            List of feature validation results
            
        Feature Validation:
            - CHAINLIT_UI: Chainlit import and basic UI setup
            - PANDOC_CONVERSION: pypandoc functionality
            - RAGAS_EVALUATION: pandas/ragas import and basic operations
            - BENCHMARKING: faiss/ir_datasets functionality
            - DEVELOPMENT_TOOLS: pytest/mypy/ruff availability
        """
        pass
    
    @abstractmethod
    def validate_raglite_operations(self, gpu_info: GPUInfo) -> List[ValidationResult]:
        """
        Validate core RAGLite functionality.
        
        Args:
            gpu_info: GPU configuration for optimization testing
            
        Returns:
            List of RAGLite operation validation results
            
        Validation Operations:
            - Database initialization (SQLite/DuckDB/PostgreSQL)
            - Document embedding and insertion
            - Vector similarity search
            - Full-text search functionality
            - Hybrid search (RRF) operations
            - Basic RAG query execution
        """
        pass
    
    @abstractmethod
    def benchmark_performance(
        self, 
        gpu_info: GPUInfo,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> Dict[str, float]:
        """
        Benchmark installation performance metrics.
        
        Args:
            gpu_info: GPU configuration for acceleration testing
            validation_level: Depth of benchmarking (QUICK/STANDARD/THOROUGH)
            
        Returns:
            Performance metrics dictionary
            
        Benchmark Metrics:
            - embedding_speed: Embeddings per second
            - search_latency: Search response time (ms) 
            - memory_usage: Peak memory consumption (MB)
            - gpu_utilization: GPU usage percentage (if available)
            - throughput_qps: Queries per second sustained
        """
        pass
    
    @abstractmethod
    def generate_validation_report(
        self, 
        results: ValidationResults,
        include_benchmarks: bool = True
    ) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            results: Complete validation results
            include_benchmarks: Whether to include performance metrics
            
        Returns:
            Formatted validation report string
            
        Report Sections:
            - Executive Summary (pass/fail status)
            - System Configuration Summary
            - Component Validation Details
            - Performance Benchmarks (optional)
            - Issues and Recommendations
        """
        pass
    
    @abstractmethod
    def suggest_fixes(self, results: ValidationResults) -> Dict[str, str]:
        """
        Generate automated fix suggestions for validation issues.
        
        Args:
            results: Validation results with identified issues
            
        Returns:
            Dictionary mapping issue to fix command/instruction
            
        Fix Categories:
            - Package reinstallation commands
            - Environment variable corrections
            - Configuration file updates
            - System dependency installations
            - Permission and path fixes
        """
        pass
```

## Component-Specific Validation Logic

### PyTorch GPU Validation Contract
```python
class PyTorchGPUValidator:
    """PyTorch GPU acceleration validation"""
    
    def validate_pytorch_cuda_setup(self, gpu_info: GPUInfo) -> List[ValidationResult]:
        """
        Validate PyTorch CUDA installation and functionality.
        
        Returns:
            List of CUDA validation results
            
        Validation Steps:
            1. torch.cuda.is_available() == True
            2. CUDA version compatibility check
            3. GPU device detection and enumeration
            4. CUDA memory allocation test
            5. Simple tensor operations on GPU
            6. cuDNN availability (if required)
        """
        results = []
        
        try:
            import torch
            
            # Basic CUDA availability
            if not torch.cuda.is_available():
                results.append(ValidationResult(
                    component="pytorch_cuda",
                    severity=ValidationSeverity.CRITICAL,
                    message="CUDA not available in PyTorch installation",
                    fix_suggestion="Reinstall PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu121"
                ))
                return results
            
            # CUDA version compatibility
            cuda_version = torch.version.cuda
            if cuda_version != gpu_info.cuda_version:
                results.append(ValidationResult(
                    component="pytorch_cuda",
                    severity=ValidationSeverity.WARNING,
                    message=f"CUDA version mismatch: PyTorch {cuda_version} vs System {gpu_info.cuda_version}",
                    details={"pytorch_cuda": cuda_version, "system_cuda": gpu_info.cuda_version}
                ))
            
            # GPU device enumeration
            gpu_count = torch.cuda.device_count()
            if gpu_count != len(gpu_info.devices):
                results.append(ValidationResult(
                    component="pytorch_cuda", 
                    severity=ValidationSeverity.WARNING,
                    message=f"GPU count mismatch: PyTorch sees {gpu_count}, system has {len(gpu_info.devices)}"
                ))
            
            # Memory allocation test
            try:
                test_tensor = torch.zeros(1000, 1000, device='cuda')
                test_result = torch.sum(test_tensor)
                del test_tensor  # Clean up
                
                results.append(ValidationResult(
                    component="pytorch_cuda",
                    severity=ValidationSeverity.SUCCESS,
                    message="PyTorch CUDA functionality verified"
                ))
            except Exception as e:
                results.append(ValidationResult(
                    component="pytorch_cuda",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"CUDA memory allocation failed: {str(e)}",
                    fix_suggestion="Check CUDA drivers and available VRAM"
                ))
                
        except ImportError:
            results.append(ValidationResult(
                component="pytorch_cuda",
                severity=ValidationSeverity.CRITICAL,
                message="PyTorch not installed",
                fix_suggestion="Install PyTorch: pip install torch torchvision torchaudio"
            ))
            
        return results
    
    def validate_pytorch_metal_setup(self, gpu_info: GPUInfo) -> List[ValidationResult]:
        """
        Validate PyTorch Metal (Apple Silicon) functionality.
        
        Returns:
            List of Metal validation results
        """
        results = []
        
        try:
            import torch
            
            # Metal availability check
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    # Test Metal device operations
                    test_tensor = torch.zeros(1000, 1000, device='mps')
                    test_result = torch.sum(test_tensor)
                    del test_tensor
                    
                    results.append(ValidationResult(
                        component="pytorch_metal",
                        severity=ValidationSeverity.SUCCESS,
                        message="PyTorch Metal functionality verified"
                    ))
                except Exception as e:
                    results.append(ValidationResult(
                        component="pytorch_metal",
                        severity=ValidationSeverity.WARNING,
                        message=f"Metal operations failed: {str(e)}",
                        fix_suggestion="Update macOS and PyTorch to latest versions"
                    ))
            else:
                results.append(ValidationResult(
                    component="pytorch_metal",
                    severity=ValidationSeverity.WARNING,
                    message="Metal Performance Shaders not available",
                    details={"mps_available": hasattr(torch.backends, 'mps')}
                ))
                
        except ImportError:
            results.append(ValidationResult(
                component="pytorch_metal",
                severity=ValidationSeverity.CRITICAL,
                message="PyTorch not installed"
            ))
            
        return results
```

### RAGLite Core Validation Contract
```python
class RAGLiteCoreValidator:
    """RAGLite core functionality validation"""
    
    def validate_raglite_import_and_init(self) -> List[ValidationResult]:
        """
        Validate RAGLite package importability and basic initialization.
        
        Returns:
            List of RAGLite import validation results
        """
        results = []
        
        try:
            # Core imports
            import raglite
            from raglite import RAG
            from raglite._database import get_database_url
            from raglite._embed import embed_sentences
            from raglite._search import search
            
            results.append(ValidationResult(
                component="raglite_import",
                severity=ValidationSeverity.SUCCESS,
                message="RAGLite core imports successful"
            ))
            
            # Basic initialization test
            try:
                # Test database URL generation
                db_url = get_database_url(":memory:")
                
                results.append(ValidationResult(
                    component="raglite_init",
                    severity=ValidationSeverity.SUCCESS,
                    message="RAGLite initialization functions working"
                ))
            except Exception as e:
                results.append(ValidationResult(
                    component="raglite_init",
                    severity=ValidationSeverity.WARNING,
                    message=f"RAGLite initialization warning: {str(e)}"
                ))
                
        except ImportError as e:
            results.append(ValidationResult(
                component="raglite_import",
                severity=ValidationSeverity.CRITICAL,
                message=f"RAGLite import failed: {str(e)}",
                fix_suggestion="Install RAGLite: pip install raglite"
            ))
            
        return results
    
    def validate_embedding_functionality(self, gpu_info: GPUInfo) -> List[ValidationResult]:
        """
        Validate text embedding functionality.
        
        Args:
            gpu_info: GPU configuration for acceleration testing
            
        Returns:
            List of embedding validation results
        """
        results = []
        
        try:
            from raglite._embed import embed_sentences
            import time
            
            # Test embedding generation
            test_sentences = ["This is a test sentence.", "Another test for embedding."]
            
            start_time = time.time()
            embeddings = embed_sentences(test_sentences)
            embedding_time = time.time() - start_time
            
            # Validate embedding dimensions
            if embeddings.shape[0] != len(test_sentences):
                results.append(ValidationResult(
                    component="embedding_generation",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Embedding count mismatch: expected {len(test_sentences)}, got {embeddings.shape[0]}"
                ))
            elif embeddings.shape[1] < 100:  # Reasonable dimension check
                results.append(ValidationResult(
                    component="embedding_generation",
                    severity=ValidationSeverity.WARNING,
                    message=f"Embedding dimensions seem low: {embeddings.shape[1]}"
                ))
            else:
                results.append(ValidationResult(
                    component="embedding_generation",
                    severity=ValidationSeverity.SUCCESS,
                    message=f"Embedding generation successful ({embedding_time:.2f}s for {len(test_sentences)} sentences)",
                    details={
                        "embedding_shape": embeddings.shape,
                        "generation_time": embedding_time,
                        "sentences_per_second": len(test_sentences) / embedding_time
                    }
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                component="embedding_generation",
                severity=ValidationSeverity.CRITICAL,
                message=f"Embedding generation failed: {str(e)}",
                fix_suggestion="Check sentence-transformers installation and model availability"
            ))
            
        return results
    
    def validate_database_operations(self) -> List[ValidationResult]:
        """
        Validate database functionality (SQLite, vector operations).
        
        Returns:
            List of database validation results
        """
        results = []
        
        try:
            import sqlite3
            from raglite._database import get_database_url
            import tempfile
            import os
            
            # Test in-memory database
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                db_path = tmp_db.name
                
            try:
                db_url = get_database_url(db_path)
                
                # Basic SQLite connection test
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                conn.close()
                
                if result == (1,):
                    results.append(ValidationResult(
                        component="database_basic",
                        severity=ValidationSeverity.SUCCESS,
                        message="SQLite database operations working"
                    ))
                
                # Test sqlite-vec extension if available
                try:
                    conn = sqlite3.connect(db_path)
                    conn.enable_load_extension(True)
                    conn.load_extension("sqlite-vec")
                    conn.close()
                    
                    results.append(ValidationResult(
                        component="sqlite_vec",
                        severity=ValidationSeverity.SUCCESS,
                        message="sqlite-vec extension loaded successfully"
                    ))
                except Exception as vec_e:
                    results.append(ValidationResult(
                        component="sqlite_vec",
                        severity=ValidationSeverity.WARNING,
                        message=f"sqlite-vec extension not available: {str(vec_e)}",
                        fix_suggestion="Install sqlite-vec extension for optimal performance"
                    ))
                    
            finally:
                # Clean up test database
                if os.path.exists(db_path):
                    os.unlink(db_path)
                    
        except Exception as e:
            results.append(ValidationResult(
                component="database_basic",
                severity=ValidationSeverity.CRITICAL,
                message=f"Database operations failed: {str(e)}",
                fix_suggestion="Check SQLite installation and permissions"
            ))
            
        return results
```

## Performance Validation Contract

### Benchmark Configuration
```python
class PerformanceBenchmarkValidator:
    """Performance benchmarking and validation"""
    
    # Benchmark thresholds by validation level
    PERFORMANCE_THRESHOLDS = {
        ValidationLevel.QUICK: {
            "min_embedding_speed": 10.0,      # embeddings/second
            "max_search_latency": 1000.0,     # milliseconds
            "max_memory_usage": 2048.0        # MB
        },
        ValidationLevel.STANDARD: {
            "min_embedding_speed": 50.0,
            "max_search_latency": 500.0,
            "max_memory_usage": 1024.0
        },
        ValidationLevel.THOROUGH: {
            "min_embedding_speed": 100.0,
            "max_search_latency": 200.0,
            "max_memory_usage": 512.0
        }
    }
    
    def benchmark_embedding_performance(
        self, 
        gpu_info: GPUInfo,
        validation_level: ValidationLevel
    ) -> ValidationResult:
        """
        Benchmark text embedding performance.
        
        Args:
            gpu_info: GPU configuration for acceleration testing
            validation_level: Benchmark rigor level
            
        Returns:
            Embedding performance validation result
        """
        try:
            from raglite._embed import embed_sentences
            import time
            import random
            import string
            
            # Generate test sentences based on validation level
            sentence_counts = {
                ValidationLevel.QUICK: 10,
                ValidationLevel.STANDARD: 100, 
                ValidationLevel.THOROUGH: 1000
            }
            
            sentence_count = sentence_counts[validation_level]
            test_sentences = [
                ''.join(random.choices(string.ascii_lowercase + ' ', k=50))
                for _ in range(sentence_count)
            ]
            
            # Benchmark embedding generation
            start_time = time.time()
            embeddings = embed_sentences(test_sentences)
            total_time = time.time() - start_time
            
            embeddings_per_second = sentence_count / total_time
            threshold = self.PERFORMANCE_THRESHOLDS[validation_level]["min_embedding_speed"]
            
            if embeddings_per_second >= threshold:
                severity = ValidationSeverity.SUCCESS
                message = f"Embedding performance good: {embeddings_per_second:.1f} embeddings/sec"
            else:
                severity = ValidationSeverity.WARNING
                message = f"Embedding performance below threshold: {embeddings_per_second:.1f} < {threshold} embeddings/sec"
                
            return ValidationResult(
                component="embedding_performance",
                severity=severity,
                message=message,
                details={
                    "embeddings_per_second": embeddings_per_second,
                    "total_time": total_time,
                    "sentence_count": sentence_count,
                    "threshold": threshold
                }
            )
            
        except Exception as e:
            return ValidationResult(
                component="embedding_performance",
                severity=ValidationSeverity.CRITICAL,
                message=f"Embedding benchmark failed: {str(e)}"
            )
```

## Integration Testing Contract

### End-to-End Validation
```python
class EndToEndValidator:
    """Complete RAGLite workflow validation"""
    
    def validate_complete_rag_workflow(self, gpu_info: GPUInfo) -> List[ValidationResult]:
        """
        Validate complete RAG workflow from document to query.
        
        Args:
            gpu_info: GPU configuration for optimization
            
        Returns:
            List of end-to-end validation results
            
        Workflow Steps:
            1. Initialize RAG with test database
            2. Insert test documents
            3. Perform vector search
            4. Perform full-text search
            5. Execute hybrid search
            6. Generate RAG response
        """
        results = []
        
        try:
            import tempfile
            import os
            from raglite import RAG
            
            with tempfile.TemporaryDirectory() as temp_dir:
                db_path = os.path.join(temp_dir, "test_rag.db")
                
                # Initialize RAG
                rag = RAG(db_url=f"sqlite:///{db_path}")
                
                # Test document insertion
                test_docs = [
                    "The quick brown fox jumps over the lazy dog.",
                    "Python is a programming language used for data science.",
                    "Machine learning models require training data."
                ]
                
                rag.insert(test_docs)
                
                results.append(ValidationResult(
                    component="rag_document_insertion",
                    severity=ValidationSeverity.SUCCESS,
                    message=f"Successfully inserted {len(test_docs)} test documents"
                ))
                
                # Test search functionality
                search_results = rag.search("programming language", k=3)
                
                if len(search_results) > 0:
                    results.append(ValidationResult(
                        component="rag_search",
                        severity=ValidationSeverity.SUCCESS,
                        message=f"Search returned {len(search_results)} results"
                    ))
                else:
                    results.append(ValidationResult(
                        component="rag_search",
                        severity=ValidationSeverity.WARNING,
                        message="Search returned no results"
                    ))
                    
                # Test RAG query (if LLM available)
                try:
                    response = rag.query("What is Python used for?")
                    
                    results.append(ValidationResult(
                        component="rag_query_generation",
                        severity=ValidationSeverity.SUCCESS,
                        message="RAG query generation successful",
                        details={"response_length": len(response)}
                    ))
                except Exception as llm_e:
                    results.append(ValidationResult(
                        component="rag_query_generation",
                        severity=ValidationSeverity.INFO,
                        message=f"RAG query generation skipped (no LLM configured): {str(llm_e)}"
                    ))
                    
        except Exception as e:
            results.append(ValidationResult(
                component="rag_workflow",
                severity=ValidationSeverity.CRITICAL,
                message=f"End-to-end RAG workflow failed: {str(e)}"
            ))
            
        return results
```

## Testing Requirements

### Validator Testing Contract
```python
class TestInstallationValidator:
    """Test contract for InstallationValidator implementations"""
    
    def test_validation_coverage(self):
        """MUST validate all critical components"""
        # Ensure all major components are validated
        pass
    
    def test_gpu_detection_accuracy(self):
        """MUST accurately detect GPU acceleration capabilities"""
        pass
    
    def test_performance_benchmarks(self):
        """MUST provide meaningful performance metrics"""
        pass
    
    def test_fix_suggestion_quality(self):
        """MUST provide actionable fix suggestions"""
        pass
    
    def test_validation_report_completeness(self):
        """MUST generate comprehensive validation reports"""
        pass
```

---

**Status**: ✅ InstallationValidator interface contract defined  
**Next**: Create quickstart guide and finalize specification documents
