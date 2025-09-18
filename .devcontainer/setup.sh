#!/bin/bash
set -e

echo "🚀 Starting RAGLite devcontainer setup..."

# 1. Fix permissions for user context
echo "📁 Fixing ownership permissions..."
sudo chown -R user:user /opt/ || echo "Warning: Could not change ownership of /opt/"
sudo chown -R user:user /workspaces/ || echo "Warning: Could not change ownership of /workspaces/"

# 2. Wait for PostgreSQL to be ready
echo "🐘 Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if pg_isready -h postgres -p 5432 -U raglite_user; then
        echo "✅ PostgreSQL is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ PostgreSQL failed to start within 30 seconds"
        exit 1
    fi
    echo "⏳ Waiting for PostgreSQL... ($i/30)"
    sleep 1
done

# 3. Initialize database and pgvector extension
echo "🔧 Initializing database..."
PGPASSWORD=raglite_password psql -h postgres -U raglite_user -d divorce_case -c "CREATE EXTENSION IF NOT EXISTS vector;" || echo "Warning: Could not create vector extension (may already exist)"

# 4. Sync Python environment with error handling
echo "🐍 Setting up Python environment..."
if ! uv sync --python 3.11 --resolution highest --all-extras; then
    echo "❌ uv sync failed, trying with Python 3.10..."
    uv sync --python 3.10 --resolution highest --all-extras || {
        echo "❌ uv sync failed completely"
        exit 1
    }
fi

# 5. Install pre-commit hooks
echo "🔍 Installing pre-commit hooks..."
pre-commit install --install-hooks || echo "Warning: pre-commit installation failed"

# 6. Initialize RAGLite with PostgreSQL
echo "🔌 Initializing RAGLite..."
python3 -c "
import os
os.environ['RAGLITE_DATABASE_URL'] = 'postgresql://raglite_user:raglite_password@postgres:5432/divorce_case'
try:
    from raglite._config import RAGLiteConfig
    config = RAGLiteConfig()
    print('✅ RAGLite configuration successful')
    
    from raglite._mcp import create_mcp_server
    server = create_mcp_server()
    print('✅ MCP server creation successful')
    
    # Test basic database connection
    import psycopg2
    conn = psycopg2.connect(
        host='postgres',
        database='divorce_case', 
        user='raglite_user',
        password='raglite_password'
    )
    conn.close()
    print('✅ Database connection successful')
    
    print('🎉 RAGLite devcontainer setup completed successfully!')
except Exception as e:
    print(f'❌ RAGLite setup failed: {e}')
    exit(1)
"

# 7. Create verification script for later use
echo "📝 Creating verification script..."
cat > /workspaces/${localWorkspaceFolderBasename}/verify_setup.py << 'EOF'
#!/usr/bin/env python3
"""
RAGLite Devcontainer Setup Verification Script
Verifies that all components are working correctly.
"""
import os
import sys

def check_environment():
    """Check environment variables"""
    required_vars = [
        'RAGLITE_DATABASE_URL',
        'POSTGRES_HOST',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD',
        'POSTGRES_DB'
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        return False
    
    print("✅ All required environment variables are set")
    return True

def check_database():
    """Check database connectivity"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST'),
            database=os.getenv('POSTGRES_DB'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD')
        )
        
        # Check pgvector extension
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        if not cur.fetchone():
            print("❌ pgvector extension not installed")
            return False
        
        conn.close()
        print("✅ Database connection and pgvector extension working")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def check_raglite():
    """Check RAGLite functionality"""
    try:
        from raglite._config import RAGLiteConfig
        config = RAGLiteConfig()
        print("✅ RAGLite configuration working")
        
        from raglite._mcp import create_mcp_server
        server = create_mcp_server()
        print("✅ MCP server creation working")
        
        return True
    except Exception as e:
        print(f"❌ RAGLite check failed: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🔍 Verifying RAGLite devcontainer setup...")
    
    checks = [
        ("Environment Variables", check_environment),
        ("Database Connection", check_database),
        ("RAGLite Functionality", check_raglite)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        if not check_func():
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All checks passed! RAGLite devcontainer is ready.")
        return 0
    else:
        print("❌ Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x /workspaces/${localWorkspaceFolderBasename}/verify_setup.py

echo "✨ Setup script completed! Run 'python verify_setup.py' to check everything is working."