"""
Fallback implementation for Redis
"""
import json
from typing import Any, Optional, Dict
import threading
import time

class FallbackRedis:
    """In-memory fallback for Redis"""
    
    def __init__(self, **kwargs):
        self._data: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        with self._lock:
            self._cleanup_expired()
            return self._data.get(key)
    
    def set(self, key: str, value: Any) -> bool:
        """Set key-value pair"""
        with self._lock:
            self._data[key] = str(value)
            return True
    
    def setex(self, key: str, time: int, value: Any) -> bool:
        """Set key-value pair with expiry"""
        with self._lock:
            self._data[key] = str(value)
            self._expiry[key] = time.time() + time
            return True
    
    def delete(self, key: str) -> int:
        """Delete key"""
        with self._lock:
            if key in self._data:
                del self._data[key]
                if key in self._expiry:
                    del self._expiry[key]
                return 1
            return 0
    
    def ping(self) -> bool:
        """Health check"""
        return True
    
    def close(self):
        """Close connection"""
        pass
    
    def _cleanup_expired(self):
        """Remove expired keys"""
        current_time = time.time()
        expired_keys = [
            key for key, expiry_time in self._expiry.items()
            if current_time > expiry_time
        ]
        for key in expired_keys:
            if key in self._data:
                del self._data[key]
            del self._expiry[key]

def from_url(url: str, **kwargs):
    """Create Redis client from URL"""
    return FallbackRedis(**kwargs)

# Create an alias for the main class
Redis = FallbackRedis
