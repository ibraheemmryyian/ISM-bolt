#!/usr/bin/env python3
"""
Event-Driven Architecture for SymbioFlows
Message queues, event sourcing, and CQRS patterns
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import redis
import threading
import queue
from functools import wraps
import traceback
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    COMPANY_CREATED = "company_created"
    MATERIAL_ADDED = "material_added"
    MATCH_CREATED = "match_created"
    PRICING_UPDATED = "pricing_updated"
    FEEDBACK_RECEIVED = "feedback_received"
    MODEL_TRAINED = "model_trained"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    SERVICE_HEALTH_CHANGED = "service_health_changed"

class EventStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Event:
    event_id: str
    event_type: EventType
    aggregate_id: str
    aggregate_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    version: int
    status: EventStatus
    processed_at: Optional[datetime] = None
    error: Optional[str] = None

@dataclass
class Command:
    command_id: str
    command_type: str
    aggregate_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    status: EventStatus
    processed_at: Optional[datetime] = None
    error: Optional[str] = None

class EventStore:
    """Event store for event sourcing"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.event_prefix = "event:"
        self.aggregate_prefix = "aggregate:"
        self.stream_prefix = "event_stream:"
        
        self.metrics = {
            'events_stored': Counter('events_stored_total', 'Total events stored', ['event_type']),
            'events_retrieved': Counter('events_retrieved_total', 'Total events retrieved', ['event_type']),
            'store_errors': Counter('event_store_errors_total', 'Total store errors', ['error_type'])
        }
    
    def store_event(self, event: Event) -> bool:
        """Store an event in the event store"""
        try:
            # Store event
            event_key = f"{self.event_prefix}{event.event_id}"
            event_data = {
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'aggregate_id': event.aggregate_id,
                'aggregate_type': event.aggregate_type,
                'data': json.dumps(event.data),
                'metadata': json.dumps(event.metadata),
                'timestamp': event.timestamp.isoformat(),
                'version': event.version,
                'status': event.status.value,
                'processed_at': event.processed_at.isoformat() if event.processed_at else None,
                'error': event.error
            }
            
            self.redis.hset(event_key, mapping=event_data)
            self.redis.expire(event_key, 86400 * 30)  # 30 days TTL
            
            # Add to aggregate stream
            stream_key = f"{self.stream_prefix}{event.aggregate_type}:{event.aggregate_id}"
            self.redis.xadd(stream_key, {
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'version': event.version,
                'timestamp': event.timestamp.isoformat()
            })
            
            # Update aggregate version
            aggregate_key = f"{self.aggregate_prefix}{event.aggregate_type}:{event.aggregate_id}"
            self.redis.hset(aggregate_key, 'version', event.version)
            self.redis.expire(aggregate_key, 86400 * 30)  # 30 days TTL
            
            self.metrics['events_stored'].labels(event.event_type.value).inc()
            return True
            
        except Exception as e:
            self.metrics['store_errors'].labels('store_event').inc()
            logger.error(f"Error storing event: {e}")
            return False
    
    def get_events(self, aggregate_type: str, aggregate_id: str, from_version: int = 0) -> List[Event]:
        """Get events for an aggregate"""
        try:
            stream_key = f"{self.stream_prefix}{aggregate_type}:{aggregate_id}"
            events = []
            
            # Get event IDs from stream
            stream_events = self.redis.xrange(stream_key, min=f"{from_version}-0", max="+")
            
            for event_id, event_data in stream_events:
                event_key = f"{self.event_prefix}{event_data[b'event_id'].decode()}"
                event_data_full = self.redis.hgetall(event_key)
                
                if event_data_full:
                    event = Event(
                        event_id=event_data_full['event_id'],
                        event_type=EventType(event_data_full['event_type']),
                        aggregate_id=event_data_full['aggregate_id'],
                        aggregate_type=event_data_full['aggregate_type'],
                        data=json.loads(event_data_full['data']),
                        metadata=json.loads(event_data_full['metadata']),
                        timestamp=datetime.fromisoformat(event_data_full['timestamp']),
                        version=int(event_data_full['version']),
                        status=EventStatus(event_data_full['status']),
                        processed_at=datetime.fromisoformat(event_data_full['processed_at']) if event_data_full['processed_at'] else None,
                        error=event_data_full['error']
                    )
                    events.append(event)
            
            self.metrics['events_retrieved'].labels('aggregate_events').inc()
            return events
            
        except Exception as e:
            self.metrics['store_errors'].labels('get_events').inc()
            logger.error(f"Error retrieving events: {e}")
            return []
    
    def get_events_by_type(self, event_type: EventType, limit: int = 100) -> List[Event]:
        """Get events by type"""
        try:
            events = []
            pattern = f"{self.event_prefix}*"
            
            for key in self.redis.scan_iter(match=pattern, count=1000):
                event_data = self.redis.hgetall(key)
                if event_data and event_data.get('event_type') == event_type.value:
                    event = Event(
                        event_id=event_data['event_id'],
                        event_type=EventType(event_data['event_type']),
                        aggregate_id=event_data['aggregate_id'],
                        aggregate_type=event_data['aggregate_type'],
                        data=json.loads(event_data['data']),
                        metadata=json.loads(event_data['metadata']),
                        timestamp=datetime.fromisoformat(event_data['timestamp']),
                        version=int(event_data['version']),
                        status=EventStatus(event_data['status']),
                        processed_at=datetime.fromisoformat(event_data['processed_at']) if event_data['processed_at'] else None,
                        error=event_data['error']
                    )
                    events.append(event)
                    
                    if len(events) >= limit:
                        break
            
            self.metrics['events_retrieved'].labels(event_type.value).inc()
            return events
            
        except Exception as e:
            self.metrics['store_errors'].labels('get_events_by_type').inc()
            logger.error(f"Error retrieving events by type: {e}")
            return []

class MessageQueue:
    """Message queue for event processing"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.queue_prefix = "queue:"
        self.processing_prefix = "processing:"
        
        self.metrics = {
            'messages_published': Counter('messages_published_total', 'Total messages published', ['queue']),
            'messages_consumed': Counter('messages_consumed_total', 'Total messages consumed', ['queue']),
            'queue_errors': Counter('queue_errors_total', 'Total queue errors', ['error_type'])
        }
    
    def publish(self, queue_name: str, message: Dict[str, Any]) -> bool:
        """Publish message to queue"""
        try:
            queue_key = f"{self.queue_prefix}{queue_name}"
            message_id = str(uuid.uuid4())
            
            message_data = {
                'id': message_id,
                'data': json.dumps(message),
                'timestamp': datetime.now().isoformat(),
                'retry_count': 0
            }
            
            self.redis.lpush(queue_key, json.dumps(message_data))
            self.redis.expire(queue_key, 86400)  # 24 hours TTL
            
            self.metrics['messages_published'].labels(queue_name).inc()
            return True
            
        except Exception as e:
            self.metrics['queue_errors'].labels('publish').inc()
            logger.error(f"Error publishing message: {e}")
            return False
    
    def consume(self, queue_name: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
        """Consume message from queue"""
        try:
            queue_key = f"{self.queue_prefix}{queue_name}"
            processing_key = f"{self.processing_prefix}{queue_name}"
            
            # Move message to processing queue
            message_data = self.redis.brpoplpush(queue_key, processing_key, timeout=timeout)
            
            if message_data:
                message = json.loads(message_data)
                self.metrics['messages_consumed'].labels(queue_name).inc()
                return message
            
            return None
            
        except Exception as e:
            self.metrics['queue_errors'].labels('consume').inc()
            logger.error(f"Error consuming message: {e}")
            return None
    
    def acknowledge(self, queue_name: str, message_id: str) -> bool:
        """Acknowledge message processing"""
        try:
            processing_key = f"{self.processing_prefix}{queue_name}"
            self.redis.lrem(processing_key, 0, message_id)
            return True
            
        except Exception as e:
            self.metrics['queue_errors'].labels('acknowledge').inc()
            logger.error(f"Error acknowledging message: {e}")
            return False

class EventHandler(ABC):
    """Abstract event handler"""
    
    @abstractmethod
    async def handle(self, event: Event) -> bool:
        """Handle an event"""
        pass

class CommandHandler(ABC):
    """Abstract command handler"""
    
    @abstractmethod
    async def handle(self, command: Command) -> bool:
        """Handle a command"""
        pass

class EventBus:
    """Event bus for event routing"""
    
    def __init__(self, event_store: EventStore, message_queue: MessageQueue):
        self.event_store = event_store
        self.message_queue = message_queue
        self.event_handlers: Dict[EventType, List[EventHandler]] = {}
        self.command_handlers: Dict[str, CommandHandler] = {}
        
        self.metrics = {
            'events_published': Counter('events_published_total', 'Total events published', ['event_type']),
            'events_handled': Counter('events_handled_total', 'Total events handled', ['event_type']),
            'commands_published': Counter('commands_published_total', 'Total commands published', ['command_type']),
            'commands_handled': Counter('commands_handled_total', 'Total commands handled', ['command_type']),
            'bus_errors': Counter('event_bus_errors_total', 'Total bus errors', ['error_type'])
        }
    
    def register_event_handler(self, event_type: EventType, handler: EventHandler):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def register_command_handler(self, command_type: str, handler: CommandHandler):
        """Register command handler"""
        self.command_handlers[command_type] = handler
    
    async def publish_event(self, event: Event) -> bool:
        """Publish an event"""
        try:
            # Store event
            if not self.event_store.store_event(event):
                return False
            
            # Publish to message queue
            queue_name = f"events_{event.event_type.value}"
            message = {
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'aggregate_id': event.aggregate_id,
                'data': event.data,
                'metadata': event.metadata
            }
            
            if not self.message_queue.publish(queue_name, message):
                return False
            
            self.metrics['events_published'].labels(event.event_type.value).inc()
            return True
            
        except Exception as e:
            self.metrics['bus_errors'].labels('publish_event').inc()
            logger.error(f"Error publishing event: {e}")
            return False
    
    async def publish_command(self, command: Command) -> bool:
        """Publish a command"""
        try:
            # Publish to message queue
            queue_name = f"commands_{command.command_type}"
            message = {
                'command_id': command.command_id,
                'command_type': command.command_type,
                'aggregate_id': command.aggregate_id,
                'data': command.data,
                'metadata': command.metadata
            }
            
            if not self.message_queue.publish(queue_name, message):
                return False
            
            self.metrics['commands_published'].labels(command.command_type).inc()
            return True
            
        except Exception as e:
            self.metrics['bus_errors'].labels('publish_command').inc()
            logger.error(f"Error publishing command: {e}")
            return False
    
    async def process_events(self, event_type: EventType):
        """Process events of a specific type"""
        queue_name = f"events_{event_type.value}"
        
        while True:
            try:
                message = self.message_queue.consume(queue_name, timeout=5)
                
                if message:
                    # Create event from message
                    event = Event(
                        event_id=message['id'],
                        event_type=event_type,
                        aggregate_id=message['data']['aggregate_id'],
                        aggregate_type=message['data'].get('aggregate_type', 'unknown'),
                        data=message['data']['data'],
                        metadata=message['data']['metadata'],
                        timestamp=datetime.fromisoformat(message['timestamp']),
                        version=1,
                        status=EventStatus.PROCESSING
                    )
                    
                    # Handle event
                    if event_type in self.event_handlers:
                        for handler in self.event_handlers[event_type]:
                            try:
                                success = await handler.handle(event)
                                if success:
                                    event.status = EventStatus.COMPLETED
                                    event.processed_at = datetime.now()
                                else:
                                    event.status = EventStatus.FAILED
                                    event.error = "Handler failed"
                            except Exception as e:
                                event.status = EventStatus.FAILED
                                event.error = str(e)
                                logger.error(f"Error handling event: {e}")
                    
                    # Update event status
                    self.event_store.store_event(event)
                    
                    # Acknowledge message
                    self.message_queue.acknowledge(queue_name, message['id'])
                    
                    self.metrics['events_handled'].labels(event_type.value).inc()
                
                await asyncio.sleep(0.1)  # Small delay
                
            except Exception as e:
                self.metrics['bus_errors'].labels('process_events').inc()
                logger.error(f"Error processing events: {e}")
                await asyncio.sleep(1)  # Wait before retry

class CQRSProjection:
    """CQRS projection for read models"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.projection_prefix = "projection:"
        
        self.metrics = {
            'projections_updated': Counter('projections_updated_total', 'Total projections updated', ['projection_type']),
            'projection_errors': Counter('projection_errors_total', 'Total projection errors', ['error_type'])
        }
    
    def update_projection(self, projection_type: str, key: str, data: Dict[str, Any]):
        """Update a projection"""
        try:
            projection_key = f"{self.projection_prefix}{projection_type}:{key}"
            self.redis.hset(projection_key, mapping=data)
            self.redis.expire(projection_key, 86400 * 7)  # 7 days TTL
            
            self.metrics['projections_updated'].labels(projection_type).inc()
            
        except Exception as e:
            self.metrics['projection_errors'].labels('update').inc()
            logger.error(f"Error updating projection: {e}")
    
    def get_projection(self, projection_type: str, key: str) -> Optional[Dict[str, Any]]:
        """Get a projection"""
        try:
            projection_key = f"{self.projection_prefix}{projection_type}:{key}"
            data = self.redis.hgetall(projection_key)
            return data if data else None
            
        except Exception as e:
            self.metrics['projection_errors'].labels('get').inc()
            logger.error(f"Error getting projection: {e}")
            return None

# Initialize components
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
event_store = EventStore(redis_client)
message_queue = MessageQueue(redis_client)
event_bus = EventBus(event_store, message_queue)
cqrs_projection = CQRSProjection(redis_client)

# Flask app for API endpoints
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Event-Driven Architecture',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/events/publish', methods=['POST'])
async def publish_event():
    """Publish an event"""
    try:
        data = request.json
        event_type = EventType(data['event_type'])
        aggregate_id = data['aggregate_id']
        aggregate_type = data['aggregate_type']
        event_data = data['data']
        metadata = data.get('metadata', {})
        
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            data=event_data,
            metadata=metadata,
            timestamp=datetime.now(),
            version=1,
            status=EventStatus.PENDING
        )
        
        success = await event_bus.publish_event(event)
        
        if success:
            return jsonify({
                'event_id': event.event_id,
                'status': 'published'
            })
        else:
            return jsonify({'error': 'Failed to publish event'}), 500
        
    except Exception as e:
        logger.error(f"Publish event error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/events/<aggregate_type>/<aggregate_id>', methods=['GET'])
def get_events(aggregate_type, aggregate_id):
    """Get events for an aggregate"""
    try:
        from_version = request.args.get('from_version', 0, type=int)
        events = event_store.get_events(aggregate_type, aggregate_id, from_version)
        
        return jsonify({
            'aggregate_type': aggregate_type,
            'aggregate_id': aggregate_id,
            'events': [
                {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'data': event.data,
                    'timestamp': event.timestamp.isoformat(),
                    'version': event.version,
                    'status': event.status.value
                }
                for event in events
            ]
        })
        
    except Exception as e:
        logger.error(f"Get events error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/projections/<projection_type>/<key>', methods=['GET'])
def get_projection(projection_type, key):
    """Get a projection"""
    try:
        projection = cqrs_projection.get_projection(projection_type, key)
        
        if projection:
            return jsonify(projection)
        else:
            return jsonify({'error': 'Projection not found'}), 404
        
    except Exception as e:
        logger.error(f"Get projection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/projections/<projection_type>/<key>', methods=['PUT'])
def update_projection(projection_type, key):
    """Update a projection"""
    try:
        data = request.json
        
        cqrs_projection.update_projection(projection_type, key, data)
        
        return jsonify({
            'projection_type': projection_type,
            'key': key,
            'status': 'updated'
        })
        
    except Exception as e:
        logger.error(f"Update projection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/queue/<queue_name>/publish', methods=['POST'])
def publish_message(queue_name):
    """Publish message to queue"""
    try:
        data = request.json
        
        success = message_queue.publish(queue_name, data)
        
        if success:
            return jsonify({'status': 'published'})
        else:
            return jsonify({'error': 'Failed to publish message'}), 500
        
    except Exception as e:
        logger.error(f"Publish message error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/queue/<queue_name>/consume', methods=['GET'])
def consume_message(queue_name):
    """Consume message from queue"""
    try:
        timeout = request.args.get('timeout', 5, type=int)
        
        message = message_queue.consume(queue_name, timeout)
        
        if message:
            return jsonify(message)
        else:
            return jsonify({'message': 'No messages available'})
        
    except Exception as e:
        logger.error(f"Consume message error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Event-Driven Architecture on port 5023...")
    app.run(host='0.0.0.0', port=5023, debug=False) 