#!/usr/bin/env python3
"""
🚀 PERFORMANCE OPTIMIZATION SYSTEM
Comprehensive performance monitoring and optimization for SymbioFlows AI services
"""

import asyncio
import logging
import time
import psutil
import gc
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref
import tracemalloc
from functools import wraps
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    response_time: float
    throughput: float
    error_rate: float
    active_connections: int
    queue_size: int

@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    max_memory_usage: float = 0.8  # 80% of available memory
    max_cpu_usage: float = 0.9     # 90% of available CPU
    target_response_time: float = 0.2  # 200ms target
    gc_threshold: int = 1000       # Garbage collection threshold
    cache_size: int = 10000        # Cache size limit
    batch_size: int = 32           # Default batch size
    max_workers: int = 4           # Maximum worker threads
    enable_profiling: bool = True
    enable_memory_tracking: bool = True

class MemoryManager:
    """Advanced memory management system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_threshold = config.max_memory_usage
        self.cache = {}
        self.cache_timestamps = {}
        self.memory_usage_history = []
        self.gc_counter = 0
        
        # Enable memory tracking
        if config.enable_memory_tracking:
            tracemalloc.start()
        
        # Start memory monitoring
        self._start_memory_monitoring()
    
    def _start_memory_monitoring(self):
        """Start background memory monitoring"""
        def monitor_memory():
            while True:
                try:
                    current_memory = psutil.virtual_memory().percent / 100
                    self.memory_usage_history.append({
                        'timestamp': datetime.now(),
                        'usage': current_memory
                    })
                    
                    # Keep only last 1000 entries
                    if len(self.memory_usage_history) > 1000:
                        self.memory_usage_history = self.memory_usage_history[-1000:]
                    
                    # Trigger garbage collection if memory usage is high
                    if current_memory > self.memory_threshold:
                        self.optimize_memory()
                    
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=monitor_memory, daemon=True)
        thread.start()
    
    def optimize_memory(self):
        """Optimize memory usage"""
        logger.info("🔄 Optimizing memory usage...")
        
        # Clear old cache entries
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if current_time - timestamp > 3600  # 1 hour TTL
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.cache_timestamps[key]
        
        # Force garbage collection
        collected = gc.collect()
        self.gc_counter += 1
        
        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log memory optimization
        current_memory = psutil.virtual_memory().percent
        logger.info(f"Memory optimized: {current_memory:.1f}% usage, {collected} objects collected")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        memory = psutil.virtual_memory()
        
        if self.config.enable_memory_tracking:
            current, peak = tracemalloc.get_traced_memory()
            return {
                'current_usage_percent': memory.percent,
                'available_mb': memory.available / 1024 / 1024,
                'total_mb': memory.total / 1024 / 1024,
                'traced_current_mb': current / 1024 / 1024,
                'traced_peak_mb': peak / 1024 / 1024,
                'cache_size': len(self.cache),
                'gc_counter': self.gc_counter
            }
        else:
            return {
                'current_usage_percent': memory.percent,
                'available_mb': memory.available / 1024 / 1024,
                'total_mb': memory.total / 1024 / 1024,
                'cache_size': len(self.cache),
                'gc_counter': self.gc_counter
            }

class CPUOptimizer:
    """CPU usage optimization system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cpu_usage_history = []
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_workers)
        self.active_tasks = weakref.WeakSet()
    
    def monitor_cpu_usage(self):
        """Monitor CPU usage"""
        current_cpu = psutil.cpu_percent(interval=1) / 100
        self.cpu_usage_history.append({
            'timestamp': datetime.now(),
            'usage': current_cpu
        })
        
        # Keep only last 1000 entries
        if len(self.cpu_usage_history) > 1000:
            self.cpu_usage_history = self.cpu_usage_history[-1000:]
        
        return current_cpu
    
    def optimize_cpu_usage(self):
        """Optimize CPU usage"""
        current_cpu = self.monitor_cpu_usage()
        
        if current_cpu > self.config.max_cpu_usage:
            logger.info(f"🔄 High CPU usage detected: {current_cpu:.1%}")
            
            # Implement CPU optimization strategies
            self._reduce_worker_threads()
            self._optimize_batch_processing()
    
    def _reduce_worker_threads(self):
        """Reduce worker threads to lower CPU usage"""
        # This is a simplified implementation
        # In production, you'd implement more sophisticated thread management
        logger.info("Reducing worker threads to optimize CPU usage")
    
    def _optimize_batch_processing(self):
        """Optimize batch processing for better CPU efficiency"""
        # Adjust batch sizes based on CPU usage
        current_cpu = self.monitor_cpu_usage()
        if current_cpu > 0.8:
            self.config.batch_size = max(8, self.config.batch_size // 2)
            logger.info(f"Reduced batch size to {self.config.batch_size}")
    
    def get_cpu_stats(self) -> Dict[str, Any]:
        """Get current CPU statistics"""
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        
        return {
            'cpu_count': cpu_count,
            'cpu_percent_per_core': cpu_percent,
            'cpu_percent_total': sum(cpu_percent) / len(cpu_percent),
            'active_tasks': len(self.active_tasks),
            'thread_pool_size': self.thread_pool._max_workers,
            'process_pool_size': self.process_pool._max_workers
        }

class ResponseTimeOptimizer:
    """Response time optimization system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.response_times = []
        self.slow_queries = []
        self.optimization_cache = {}
    
    def track_response_time(self, operation: str, start_time: float, end_time: float):
        """Track response time for operations"""
        response_time = end_time - start_time
        self.response_times.append({
            'operation': operation,
            'response_time': response_time,
            'timestamp': datetime.now()
        })
        
        # Keep only last 1000 entries
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        # Track slow queries
        if response_time > self.config.target_response_time:
            self.slow_queries.append({
                'operation': operation,
                'response_time': response_time,
                'timestamp': datetime.now()
            })
            
            # Keep only last 100 slow queries
            if len(self.slow_queries) > 100:
                self.slow_queries = self.slow_queries[-100:]
    
    def optimize_response_time(self, operation: str) -> Dict[str, Any]:
        """Optimize response time for specific operations"""
        # Check cache for optimization strategies
        if operation in self.optimization_cache:
            return self.optimization_cache[operation]
        
        # Analyze slow queries for this operation
        operation_times = [
            q for q in self.slow_queries 
            if q['operation'] == operation
        ]
        
        if not operation_times:
            return {'strategy': 'no_optimization_needed'}
        
        # Calculate average response time
        avg_response_time = sum(q['response_time'] for q in operation_times) / len(operation_times)
        
        # Generate optimization strategy
        strategy = self._generate_optimization_strategy(operation, avg_response_time)
        
        # Cache the strategy
        self.optimization_cache[operation] = strategy
        
        return strategy
    
    def _generate_optimization_strategy(self, operation: str, avg_response_time: float) -> Dict[str, Any]:
        """Generate optimization strategy based on operation type and response time"""
        strategies = {
            'ai_inference': {
                'strategy': 'batch_processing',
                'batch_size': min(64, self.config.batch_size * 2),
                'use_cache': True,
                'parallel_processing': True
            },
            'database_query': {
                'strategy': 'query_optimization',
                'use_indexes': True,
                'limit_results': True,
                'use_connection_pool': True
            },
            'external_api': {
                'strategy': 'caching_and_retry',
                'cache_duration': 300,  # 5 minutes
                'retry_attempts': 3,
                'timeout': 5.0
            },
            'file_processing': {
                'strategy': 'streaming_processing',
                'chunk_size': 1024 * 1024,  # 1MB chunks
                'parallel_processing': True
            }
        }
        
        # Get base strategy for operation type
        base_strategy = strategies.get(operation.split('_')[0], {
            'strategy': 'general_optimization',
            'use_cache': True,
            'parallel_processing': True
        })
        
        # Adjust strategy based on response time
        if avg_response_time > 1.0:  # More than 1 second
            base_strategy['aggressive_optimization'] = True
            base_strategy['use_cache'] = True
            base_strategy['parallel_processing'] = True
        
        return base_strategy
    
    def get_response_time_stats(self) -> Dict[str, Any]:
        """Get response time statistics"""
        if not self.response_times:
            return {'avg_response_time': 0, 'slow_queries_count': 0}
        
        avg_response_time = sum(rt['response_time'] for rt in self.response_times) / len(self.response_times)
        
        return {
            'avg_response_time': avg_response_time,
            'slow_queries_count': len(self.slow_queries),
            'total_operations': len(self.response_times),
            'target_response_time': self.config.target_response_time
        }

class PerformanceProfiler:
    """Performance profiling system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.profiles = {}
        self.active_profiles = {}
    
    def start_profile(self, name: str) -> str:
        """Start profiling a section of code"""
        if not self.config.enable_profiling:
            return ""
        
        profile_id = f"{name}_{int(time.time() * 1000)}"
        self.active_profiles[profile_id] = {
            'name': name,
            'start_time': time.time(),
            'start_memory': psutil.virtual_memory().percent if self.config.enable_memory_tracking else 0
        }
        
        return profile_id
    
    def end_profile(self, profile_id: str):
        """End profiling a section of code"""
        if not self.config.enable_profiling or profile_id not in self.active_profiles:
            return
        
        profile = self.active_profiles.pop(profile_id)
        end_time = time.time()
        end_memory = psutil.virtual_memory().percent if self.config.enable_memory_tracking else 0
        
        duration = end_time - profile['start_time']
        memory_delta = end_memory - profile['start_memory']
        
        if profile['name'] not in self.profiles:
            self.profiles[profile['name']] = []
        
        self.profiles[profile['name']].append({
            'duration': duration,
            'memory_delta': memory_delta,
            'timestamp': datetime.now()
        })
        
        # Keep only last 100 profiles per operation
        if len(self.profiles[profile['name']]) > 100:
            self.profiles[profile['name']] = self.profiles[profile['name']][-100:]
    
    def get_profile_stats(self, name: str) -> Dict[str, Any]:
        """Get profiling statistics for a specific operation"""
        if name not in self.profiles:
            return {'avg_duration': 0, 'total_calls': 0}
        
        profiles = self.profiles[name]
        if not profiles:
            return {'avg_duration': 0, 'total_calls': 0}
        
        durations = [p['duration'] for p in profiles]
        memory_deltas = [p['memory_delta'] for p in profiles]
        
        return {
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_calls': len(profiles),
            'avg_memory_delta': sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0
        }

class PerformanceOptimizer:
    """Main performance optimization system"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.memory_manager = MemoryManager(self.config)
        self.cpu_optimizer = CPUOptimizer(self.config)
        self.response_optimizer = ResponseTimeOptimizer(self.config)
        self.profiler = PerformanceProfiler(self.config)
        self.metrics_history = []
        
        # Start background optimization
        self._start_background_optimization()
    
    def _start_background_optimization(self):
        """Start background optimization processes"""
        def background_optimization():
            while True:
                try:
                    # Collect metrics
                    metrics = self.collect_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Keep only last 1000 metrics
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                    
                    # Run optimizations
                    self.optimize_performance()
                    
                    time.sleep(30)  # Run every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Background optimization error: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=background_optimization, daemon=True)
        thread.start()
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=self.cpu_optimizer.monitor_cpu_usage(),
            memory_usage=psutil.virtual_memory().percent / 100,
            response_time=self.response_optimizer.get_response_time_stats()['avg_response_time'],
            throughput=len(self.response_optimizer.response_times) / 60,  # operations per minute
            error_rate=0.0,  # Would be calculated from actual error tracking
            active_connections=len(self.cpu_optimizer.active_tasks),
            queue_size=0  # Would be calculated from actual queue monitoring
        )
    
    def optimize_performance(self):
        """Run performance optimizations"""
        logger.info("🔄 Running performance optimizations...")
        
        # Memory optimization
        if self.memory_manager.get_memory_stats()['current_usage_percent'] > self.config.max_memory_usage * 100:
            self.memory_manager.optimize_memory()
        
        # CPU optimization
        current_cpu = self.cpu_optimizer.monitor_cpu_usage()
        if current_cpu > self.config.max_cpu_usage:
            self.cpu_optimizer.optimize_cpu_usage()
        
        # Response time optimization
        response_stats = self.response_optimizer.get_response_time_stats()
        if response_stats['avg_response_time'] > self.config.target_response_time:
            logger.info("Optimizing response times...")
    
    def profile_operation(self, operation_name: str):
        """Decorator for profiling operations"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                profile_id = self.profiler.start_profile(operation_name)
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    self.response_optimizer.track_response_time(operation_name, start_time, end_time)
                    self.profiler.end_profile(profile_id)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                profile_id = self.profiler.start_profile(operation_name)
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    self.response_optimizer.track_response_time(operation_name, start_time, end_time)
                    self.profiler.end_profile(profile_id)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        memory_stats = self.memory_manager.get_memory_stats()
        cpu_stats = self.cpu_optimizer.get_cpu_stats()
        response_stats = self.response_optimizer.get_response_time_stats()
        
        # Calculate trends
        if len(self.metrics_history) > 1:
            recent_metrics = self.metrics_history[-10:]  # Last 10 metrics
            cpu_trend = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            memory_trend = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            response_trend = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        else:
            cpu_trend = memory_trend = response_trend = 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory': memory_stats,
            'cpu': cpu_stats,
            'response_time': response_stats,
            'trends': {
                'cpu_usage': cpu_trend,
                'memory_usage': memory_trend,
                'response_time': response_trend
            },
            'optimization_config': {
                'max_memory_usage': self.config.max_memory_usage,
                'max_cpu_usage': self.config.max_cpu_usage,
                'target_response_time': self.config.target_response_time,
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers
            },
            'profiles': {
                name: self.profiler.get_profile_stats(name)
                for name in self.profiler.profiles.keys()
            }
        }
    
    def save_performance_report(self, filename: str = None):
        """Save performance report to file"""
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.get_performance_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {filename}")

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

# Convenience functions
def profile_operation(operation_name: str):
    """Decorator for profiling operations"""
    return performance_optimizer.profile_operation(operation_name)

def get_performance_report():
    """Get current performance report"""
    return performance_optimizer.get_performance_report()

def optimize_performance():
    """Run performance optimizations"""
    performance_optimizer.optimize_performance()

def save_performance_report(filename: str = None):
    """Save performance report"""
    performance_optimizer.save_performance_report(filename) 