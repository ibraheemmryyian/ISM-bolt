"""
Redis Fix for All Services
Updates all Python services to use Redis mock when Redis is unavailable
"""

import os
import re
import glob
from pathlib import Path

def add_redis_mock_to_file(file_path):
    """Add Redis mock import and connection logic to a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file already has Redis mock logic
        if 'redis_mock' in content or 'REDIS_AVAILABLE' in content:
            print(f"✅ {file_path} already has Redis mock logic")
            return False
        
        # Check if file uses Redis
        if 'import redis' in content or 'redis.Redis' in content:
            print(f"🔧 Adding Redis mock to {file_path}")
            
            # Add Redis mock import after existing imports
            redis_import_pattern = r'(import redis)'
            redis_mock_import = '''try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: redis not available. Using mock Redis.")

# Import mock Redis if real Redis is not available
if not REDIS_AVAILABLE:
    from redis_mock import redis_connect, redis_get_client'''
            
            content = re.sub(redis_import_pattern, redis_mock_import, content)
            
            # Replace Redis connection patterns
            redis_connect_patterns = [
                (r'redis\.Redis\([^)]*\)', 'redis_get_client() if not REDIS_AVAILABLE else redis.Redis(host=\'localhost\', port=6379, db=0, decode_responses=True)'),
                (r'redis\.Redis\(\)', 'redis_get_client() if not REDIS_AVAILABLE else redis.Redis(host=\'localhost\', port=6379, db=0, decode_responses=True)')
            ]
            
            for pattern, replacement in redis_connect_patterns:
                content = re.sub(pattern, replacement, content)
            
            # Add error handling for Redis connection
            redis_error_pattern = r'(redis_client\.ping\(\))'
            redis_error_handling = '''try:
                if REDIS_AVAILABLE:
                    redis_client.ping()
                    print("✅ Redis connection established")
                else:
                    redis_client = redis_get_client()
                    print("✅ Mock Redis connection established")
            except Exception as e:
                print(f"⚠️ Redis connection failed: {e}")
                redis_client = redis_get_client()
                print("✅ Fallback to mock Redis")'''
            
            content = re.sub(redis_error_pattern, redis_error_handling, content)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def main():
    """Update all Python files in the backend directory"""
    backend_dir = Path('.')
    python_files = list(backend_dir.glob('*.py'))
    
    print("🔧 Updating all Python services to use Redis mock...")
    print("=" * 60)
    
    updated_count = 0
    total_files = len(python_files)
    
    for file_path in python_files:
        if add_redis_mock_to_file(file_path):
            updated_count += 1
    
    print("=" * 60)
    print(f"✅ Updated {updated_count}/{total_files} files with Redis mock support")
    print("🎉 All services should now work without Redis server!")

if __name__ == "__main__":
    main() 