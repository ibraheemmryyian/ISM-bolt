"""
Production-ready database connection manager
"""
import os
import asyncio
import logging
from typing import Optional, Any, Dict
from contextlib import asynccontextmanager
import json

try:
    import psycopg2
    from psycopg2.pool import ThreadedConnectionPool
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

class ProductionDatabaseManager:
    """Production-ready database manager"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.postgres_pool: Optional[ThreadedConnectionPool] = None
        self.redis_client: Optional[Any] = None
        self.sqlite_conn: Optional[sqlite3.Connection] = None
        
        # Initialize connections
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections"""
        # PostgreSQL
        if HAS_POSTGRES:
            try:
                database_url = os.getenv('DATABASE_URL')
                if database_url:
                    self.postgres_pool = ThreadedConnectionPool(
                        1, 20, database_url
                    )
                    self.logger.info("✅ PostgreSQL connection pool initialized")
            except Exception as e:
                self.logger.warning(f"PostgreSQL initialization failed: {e}")
        
        # Redis
        if HAS_REDIS:
            try:
                redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                self.logger.info("✅ Redis connection initialized")
            except Exception as e:
                self.logger.warning(f"Redis initialization failed: {e}")
                self.redis_client = None
        
        # SQLite fallback
        if HAS_SQLITE and not self.postgres_pool:
            try:
                db_path = os.getenv('SQLITE_DB_PATH', 'symbioflows.db')
                self.sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
                self.sqlite_conn.row_factory = sqlite3.Row
                self.logger.info("✅ SQLite fallback initialized")
            except Exception as e:
                self.logger.warning(f"SQLite initialization failed: {e}")
    
    @asynccontextmanager
    async def get_db_connection(self):
        """Get database connection context manager"""
        if self.postgres_pool:
            conn = self.postgres_pool.getconn()
            try:
                yield conn
            finally:
                self.postgres_pool.putconn(conn)
        elif self.sqlite_conn:
            yield self.sqlite_conn
        else:
            raise Exception("No database connection available")
    
    async def execute_query(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute database query"""
        try:
            async with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or ())
                
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return cursor.rowcount
        except Exception as e:
            self.logger.error(f"Database query failed: {e}")
            raise
    
    async def cache_set(self, key: str, value: Any, ttl: int = 3600):
        """Set cache value"""
        if self.redis_client:
            try:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                self.redis_client.setex(key, ttl, value)
            except Exception as e:
                self.logger.warning(f"Cache set failed: {e}")
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    try:
                        return json.loads(value)
                    except:
                        return value
            except Exception as e:
                self.logger.warning(f"Cache get failed: {e}")
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        status = {
            'postgres': False,
            'redis': False,
            'sqlite': False
        }
        
        # Check PostgreSQL
        if self.postgres_pool:
            try:
                with self.postgres_pool.getconn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT 1')
                    status['postgres'] = True
                    self.postgres_pool.putconn(conn)
            except:
                pass
        
        # Check Redis
        if self.redis_client:
            try:
                self.redis_client.ping()
                status['redis'] = True
            except:
                pass
        
        # Check SQLite
        if self.sqlite_conn:
            try:
                cursor = self.sqlite_conn.cursor()
                cursor.execute('SELECT 1')
                status['sqlite'] = True
            except:
                pass
        
        return status
    
    def close_connections(self):
        """Close all database connections"""
        if self.postgres_pool:
            self.postgres_pool.closeall()
        if self.redis_client:
            self.redis_client.close()
        if self.sqlite_conn:
            self.sqlite_conn.close()

# Global database manager instance
db_manager = ProductionDatabaseManager()
