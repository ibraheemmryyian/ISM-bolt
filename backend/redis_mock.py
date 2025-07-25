"""
Redis Mock Service for SymbioFlows
Provides Redis-like functionality without requiring Redis server
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

class RedisMock:
    """Mock Redis implementation for local development"""
    
    def __init__(self):
        self.data = {}
        self.expiry = {}
        self.lists = defaultdict(list)
        self.sets = defaultdict(set)
        self.hashes = defaultdict(dict)
        self.logger = logging.getLogger(__name__)
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_keys, daemon=True)
        self.cleanup_thread.start()
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set a key-value pair with optional expiration"""
        try:
            self.data[key] = value
            if ex:
                self.expiry[key] = datetime.now() + timedelta(seconds=ex)
            else:
                self.expiry.pop(key, None)
            return True
        except Exception as e:
            self.logger.error(f"Error setting key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value for a key"""
        try:
            if key in self.expiry and datetime.now() > self.expiry[key]:
                self.delete(key)
                return None
            return self.data.get(key)
        except Exception as e:
            self.logger.error(f"Error getting key {key}: {e}")
            return None
    
    def delete(self, key: str) -> int:
        """Delete a key"""
        try:
            count = 0
            if key in self.data:
                del self.data[key]
                count += 1
            if key in self.expiry:
                del self.expiry[key]
            if key in self.lists:
                del self.lists[key]
                count += 1
            if key in self.sets:
                del self.sets[key]
                count += 1
            if key in self.hashes:
                del self.hashes[key]
                count += 1
            return count
        except Exception as e:
            self.logger.error(f"Error deleting key {key}: {e}")
            return 0
    
    def exists(self, key: str) -> int:
        """Check if key exists"""
        try:
            if key in self.expiry and datetime.now() > self.expiry[key]:
                self.delete(key)
                return 0
            return 1 if key in self.data or key in self.lists or key in self.sets or key in self.hashes else 0
        except Exception as e:
            self.logger.error(f"Error checking existence of key {key}: {e}")
            return 0
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for a key"""
        try:
            if key in self.data or key in self.lists or key in self.sets or key in self.hashes:
                self.expiry[key] = datetime.now() + timedelta(seconds=seconds)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting expiration for key {key}: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """Get time to live for a key"""
        try:
            if key in self.expiry:
                ttl = (self.expiry[key] - datetime.now()).total_seconds()
                return max(0, int(ttl))
            return -1  # No expiration
        except Exception as e:
            self.logger.error(f"Error getting TTL for key {key}: {e}")
            return -1
    
    # List operations
    def lpush(self, key: str, *values) -> int:
        """Push values to the left of a list"""
        try:
            for value in reversed(values):
                self.lists[key].insert(0, value)
            return len(self.lists[key])
        except Exception as e:
            self.logger.error(f"Error in lpush for key {key}: {e}")
            return 0
    
    def rpush(self, key: str, *values) -> int:
        """Push values to the right of a list"""
        try:
            self.lists[key].extend(values)
            return len(self.lists[key])
        except Exception as e:
            self.logger.error(f"Error in rpush for key {key}: {e}")
            return 0
    
    def lpop(self, key: str) -> Optional[Any]:
        """Pop value from the left of a list"""
        try:
            if self.lists[key]:
                return self.lists[key].pop(0)
            return None
        except Exception as e:
            self.logger.error(f"Error in lpop for key {key}: {e}")
            return None
    
    def rpop(self, key: str) -> Optional[Any]:
        """Pop value from the right of a list"""
        try:
            if self.lists[key]:
                return self.lists[key].pop()
            return None
        except Exception as e:
            self.logger.error(f"Error in rpop for key {key}: {e}")
            return None
    
    def lrange(self, key: str, start: int, end: int) -> List[Any]:
        """Get range of values from a list"""
        try:
            if key not in self.lists:
                return []
            lst = self.lists[key]
            if end == -1:
                end = len(lst)
            return lst[start:end+1]
        except Exception as e:
            self.logger.error(f"Error in lrange for key {key}: {e}")
            return []
    
    def llen(self, key: str) -> int:
        """Get length of a list"""
        try:
            return len(self.lists.get(key, []))
        except Exception as e:
            self.logger.error(f"Error in llen for key {key}: {e}")
            return 0
    
    # Set operations
    def sadd(self, key: str, *members) -> int:
        """Add members to a set"""
        try:
            added = 0
            for member in members:
                if member not in self.sets[key]:
                    self.sets[key].add(member)
                    added += 1
            return added
        except Exception as e:
            self.logger.error(f"Error in sadd for key {key}: {e}")
            return 0
    
    def srem(self, key: str, *members) -> int:
        """Remove members from a set"""
        try:
            removed = 0
            for member in members:
                if member in self.sets[key]:
                    self.sets[key].remove(member)
                    removed += 1
            return removed
        except Exception as e:
            self.logger.error(f"Error in srem for key {key}: {e}")
            return 0
    
    def smembers(self, key: str) -> set:
        """Get all members of a set"""
        try:
            return self.sets.get(key, set()).copy()
        except Exception as e:
            self.logger.error(f"Error in smembers for key {key}: {e}")
            return set()
    
    # Hash operations
    def hset(self, key: str, field: str, value: Any) -> int:
        """Set field in hash"""
        try:
            if field not in self.hashes[key]:
                self.hashes[key][field] = value
                return 1
            else:
                self.hashes[key][field] = value
                return 0
        except Exception as e:
            self.logger.error(f"Error in hset for key {key}: {e}")
            return 0
    
    def hget(self, key: str, field: str) -> Optional[Any]:
        """Get field from hash"""
        try:
            return self.hashes.get(key, {}).get(field)
        except Exception as e:
            self.logger.error(f"Error in hget for key {key}: {e}")
            return None
    
    def hgetall(self, key: str) -> Dict[str, Any]:
        """Get all fields from hash"""
        try:
            return self.hashes.get(key, {}).copy()
        except Exception as e:
            self.logger.error(f"Error in hgetall for key {key}: {e}")
            return {}
    
    def _cleanup_expired_keys(self):
        """Clean up expired keys periodically"""
        while True:
            try:
                current_time = datetime.now()
                expired_keys = []
                
                for key, expiry_time in self.expiry.items():
                    if current_time > expiry_time:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self.delete(key)
                
                if expired_keys:
                    self.logger.info(f"Cleaned up {len(expired_keys)} expired keys")
                
                time.sleep(60)  # Clean up every minute
                
            except Exception as e:
                self.logger.error(f"Error in cleanup thread: {e}")
                time.sleep(60)

# Global Redis mock instance
redis_mock = RedisMock()

# Redis-like interface functions
def redis_connect(host='localhost', port=6379, db=0, **kwargs):
    """Mock Redis connection function"""
    return redis_mock

def redis_get_client():
    """Get Redis mock client"""
    return redis_mock 