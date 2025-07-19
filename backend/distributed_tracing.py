#!/usr/bin/env python3
"""
Distributed Tracing System for SymbioFlows
Jaeger integration for tracking requests across microservices
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import opentracing
from opentracing import tags
import jaeger_client
from jaeger_client import Config
from functools import wraps
import traceback
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TraceContext:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    baggage: Dict[str, str]

class DistributedTracer:
    """Distributed tracing with Jaeger integration"""
    
    def __init__(self, service_name: str, jaeger_host: str = "localhost", jaeger_port: int = 6831):
        self.service_name = service_name
        self.tracer = self._init_jaeger(jaeger_host, jaeger_port)
        self.active_spans = {}
        
        # Metrics
        self.metrics = {
            'spans_created': Counter('spans_created_total', 'Total spans created', ['service']),
            'spans_completed': Counter('spans_completed_total', 'Total spans completed', ['service']),
            'span_duration': Histogram('span_duration_seconds', 'Span duration', ['service', 'operation']),
            'trace_errors': Counter('trace_errors_total', 'Total trace errors', ['service', 'error_type'])
        }
    
    def _init_jaeger(self, jaeger_host: str, jaeger_port: int):
        """Initialize Jaeger tracer"""
        config = Config(
            config={
                'sampler': {'type': 'const', 'param': 1},
                'logging': True,
                'local_agent': {
                    'reporting_host': jaeger_host,
                    'reporting_port': jaeger_port
                }
            },
            service_name=self.service_name
        )
        return config.initialize_tracer()
    
    def start_span(self, operation_name: str, parent_span=None, tags: Dict[str, Any] = None) -> opentracing.Span:
        """Start a new span"""
        try:
            if parent_span:
                span = self.tracer.start_span(operation_name, child_of=parent_span)
            else:
                span = self.tracer.start_span(operation_name)
            
            # Add default tags
            span.set_tag(tags.SPAN_KIND, tags.SPAN_KIND_RPC_SERVER)
            span.set_tag('service.name', self.service_name)
            
            # Add custom tags
            if tags:
                for key, value in tags.items():
                    span.set_tag(key, value)
            
            # Store span for cleanup
            span_id = str(uuid.uuid4())
            self.active_spans[span_id] = span
            
            self.metrics['spans_created'].labels(self.service_name).inc()
            
            return span
            
        except Exception as e:
            self.metrics['trace_errors'].labels(self.service_name, 'span_creation').inc()
            logger.error(f"Error creating span: {e}")
            # Return a no-op span
            return opentracing.Span()
    
    def finish_span(self, span: opentracing.Span, error: Exception = None):
        """Finish a span"""
        try:
            if error:
                span.set_tag('error', True)
                span.set_tag('error.message', str(error))
                span.log_kv({'event': 'error', 'error.object': error})
            
            span.finish()
            
            # Remove from active spans
            span_id = None
            for sid, s in self.active_spans.items():
                if s == span:
                    span_id = sid
                    break
            
            if span_id:
                del self.active_spans[span_id]
            
            self.metrics['spans_completed'].labels(self.service_name).inc()
            
        except Exception as e:
            self.metrics['trace_errors'].labels(self.service_name, 'span_finish').inc()
            logger.error(f"Error finishing span: {e}")
    
    def inject_headers(self, span: opentracing.Span, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into headers"""
        try:
            self.tracer.inject(span, opentracing.Format.HTTP_HEADERS, headers)
            return headers
        except Exception as e:
            self.metrics['trace_errors'].labels(self.service_name, 'header_injection').inc()
            logger.error(f"Error injecting headers: {e}")
            return headers
    
    def extract_span(self, headers: Dict[str, str], operation_name: str) -> opentracing.Span:
        """Extract span from headers"""
        try:
            span_context = self.tracer.extract(opentracing.Format.HTTP_HEADERS, headers)
            if span_context:
                span = self.tracer.start_span(operation_name, child_of=span_context)
            else:
                span = self.tracer.start_span(operation_name)
            
            return span
            
        except Exception as e:
            self.metrics['trace_errors'].labels(self.service_name, 'span_extraction').inc()
            logger.error(f"Error extracting span: {e}")
            return self.tracer.start_span(operation_name)
    
    def trace_function(self, operation_name: str = None):
        """Decorator to trace function calls"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                span = self.start_span(op_name)
                
                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    span.set_tag('function.duration', duration)
                    span.set_tag('function.success', True)
                    
                    self.metrics['span_duration'].labels(self.service_name, op_name).observe(duration)
                    
                    return result
                    
                except Exception as e:
                    span.set_tag('function.success', False)
                    span.set_tag('function.error', str(e))
                    self.finish_span(span, e)
                    raise
                finally:
                    self.finish_span(span)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                span = self.start_span(op_name)
                
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    span.set_tag('function.duration', duration)
                    span.set_tag('function.success', True)
                    
                    self.metrics['span_duration'].labels(self.service_name, op_name).observe(duration)
                    
                    return result
                    
                except Exception as e:
                    span.set_tag('function.success', False)
                    span.set_tag('function.error', str(e))
                    self.finish_span(span, e)
                    raise
                finally:
                    self.finish_span(span)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator

class TraceMiddleware:
    """Middleware for tracing HTTP requests"""
    
    def __init__(self, tracer: DistributedTracer):
        self.tracer = tracer
    
    def __call__(self, request):
        """Process incoming request with tracing"""
        # Extract trace context from headers
        headers = dict(request.headers)
        span = self.tracer.extract_span(headers, f"HTTP {request.method} {request.path}")
        
        # Add request tags
        span.set_tag(tags.HTTP_METHOD, request.method)
        span.set_tag(tags.HTTP_URL, request.url)
        span.set_tag(tags.HTTP_STATUS_CODE, 200)  # Will be updated later
        
        # Store span in request context
        request.span = span
        
        return request
    
    def finish_request(self, request, response, error=None):
        """Finish tracing for request"""
        if hasattr(request, 'span'):
            span = request.span
            
            if response:
                span.set_tag(tags.HTTP_STATUS_CODE, response.status_code)
            
            if error:
                span.set_tag('error', True)
                span.set_tag('error.message', str(error))
            
            self.tracer.finish_span(span, error)

class TraceContextManager:
    """Context manager for tracing"""
    
    def __init__(self, tracer: DistributedTracer, operation_name: str, tags: Dict[str, Any] = None):
        self.tracer = tracer
        self.operation_name = operation_name
        self.tags = tags or {}
        self.span = None
    
    def __enter__(self):
        self.span = self.tracer.start_span(self.operation_name, tags=self.tags)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            self.tracer.finish_span(self.span, exc_val)
    
    async def __aenter__(self):
        self.span = self.tracer.start_span(self.operation_name, tags=self.tags)
        return self.span
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            self.tracer.finish_span(self.span, exc_val)

# Initialize distributed tracer
distributed_tracer = DistributedTracer("symbioflows-tracing")

# Flask app for API endpoints
from flask import Flask, request, jsonify
app = Flask(__name__)

# Add tracing middleware
trace_middleware = TraceMiddleware(distributed_tracer)

@app.before_request
def before_request():
    """Add tracing to incoming requests"""
    trace_middleware(request)

@app.after_request
def after_request(response):
    """Finish tracing for requests"""
    trace_middleware.finish_request(request, response)
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle exceptions with tracing"""
    trace_middleware.finish_request(request, None, e)
    return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
@distributed_tracer.trace_function("health_check")
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Distributed Tracing',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/trace/span', methods=['POST'])
@distributed_tracer.trace_function("create_span")
def create_span():
    """Create a custom span"""
    try:
        data = request.json
        operation_name = data.get('operation_name', 'custom_span')
        tags = data.get('tags', {})
        
        with TraceContextManager(distributed_tracer, operation_name, tags) as span:
            # Simulate some work
            time.sleep(0.1)
            
            # Add custom log
            span.log_kv({'event': 'custom_work', 'message': 'Custom span work completed'})
            
            return jsonify({
                'span_id': span.span_id,
                'trace_id': span.trace_id,
                'operation_name': operation_name
            })
        
    except Exception as e:
        logger.error(f"Create span error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/trace/inject', methods=['POST'])
@distributed_tracer.trace_function("inject_trace")
def inject_trace():
    """Inject trace context into headers"""
    try:
        data = request.json
        headers = data.get('headers', {})
        operation_name = data.get('operation_name', 'inject_span')
        
        span = distributed_tracer.start_span(operation_name)
        injected_headers = distributed_tracer.inject_headers(span, headers)
        distributed_tracer.finish_span(span)
        
        return jsonify({
            'original_headers': headers,
            'injected_headers': injected_headers
        })
        
    except Exception as e:
        logger.error(f"Inject trace error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/trace/extract', methods=['POST'])
@distributed_tracer.trace_function("extract_trace")
def extract_trace():
    """Extract trace context from headers"""
    try:
        data = request.json
        headers = data.get('headers', {})
        operation_name = data.get('operation_name', 'extract_span')
        
        span = distributed_tracer.extract_span(headers, operation_name)
        distributed_tracer.finish_span(span)
        
        return jsonify({
            'span_id': span.span_id,
            'trace_id': span.trace_id,
            'operation_name': operation_name
        })
        
    except Exception as e:
        logger.error(f"Extract trace error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/trace/metrics', methods=['GET'])
def get_trace_metrics():
    """Get tracing metrics"""
    try:
        metrics = {}
        
        # Get Prometheus metrics
        for metric_name, metric in distributed_tracer.metrics.items():
            if hasattr(metric, '_metrics'):
                metrics[metric_name] = {
                    'type': type(metric).__name__,
                    'value': metric._metrics
                }
        
        return jsonify({
            'service_name': distributed_tracer.service_name,
            'active_spans': len(distributed_tracer.active_spans),
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Get trace metrics error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Distributed Tracing on port 5022...")
    app.run(host='0.0.0.0', port=5022, debug=False) 