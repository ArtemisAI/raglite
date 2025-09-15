from raglite._database import create_database_engine
from raglite._config import RAGLiteConfig
config = RAGLiteConfig(db_url='sqlite:///tests/test_raglite.db')
engine = create_database_engine(config)
print('✅ SQLite engine created successfully')
print(f'sqlite-vec available: {getattr(engine, "sqlite_vec_available", False)}')