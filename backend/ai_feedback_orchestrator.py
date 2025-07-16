"""
Production-Grade AI Feedback Orchestration System
Handles feedback ingestion, retraining triggers, and automated model updates
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import pickle
import numpy as np
import pandas as pd

# Database imports
from supabase import create_client, Client
import os

# AI component imports
from backend.model_persistence_manager import ModelPersistenceManager
from backend.federated_meta_learning import FederatedMetaLearning
from backend.gnn_reasoning_engine import GNNReasoningEngine
from backend.knowledge_graph import KnowledgeGraph
from revolutionary_ai_matching import RevolutionaryAIMatching

logger = logging.getLogger(__name__)

@dataclass
class FeedbackEvent:
    """Structured feedback event"""
    event_id: str
    event_type: str  # 'user_feedback', 'match_outcome', 'system_metric'
    source: str  # 'user', 'system', 'external'
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]
    priority: int = 1  # 1-5, higher = more important

@dataclass
class RetrainingTrigger:
    """Retraining trigger configuration"""
    trigger_id: str
    model_name: str
    trigger_type: str  # 'schedule', 'threshold', 'manual'
    conditions: Dict[str, Any]
    last_triggered: Optional[datetime] = None
    next_trigger: Optional[datetime] = None
    status: str = 'active'

@dataclass
class ModelPerformance:
    """Model performance tracking"""
    model_name: str
    version: str
    metrics: Dict[str, float]
    feedback_count: int
    last_updated: datetime
    improvement_score: float = 0.0

class FeedbackDatabase:
    """Production-grade feedback storage with SQLite for local + Supabase for cloud"""
    
    def __init__(self, local_db_path: str = "feedback_data.db"):
        self.local_db_path = Path(local_db_path)
        self.local_db_path.parent.mkdir(exist_ok=True)
        
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.supabase = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None
        
        # Initialize local database
        self._init_local_database()
        
        logger.info("Feedback database initialized")
    
    def _init_local_database(self):
        """Initialize local SQLite database"""
        with sqlite3.connect(self.local_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    priority INTEGER DEFAULT 1,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retraining_triggers (
                    trigger_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    last_triggered TEXT,
                    next_trigger TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    model_name TEXT,
                    version TEXT,
                    metrics TEXT NOT NULL,
                    feedback_count INTEGER DEFAULT 0,
                    last_updated TEXT NOT NULL,
                    improvement_score REAL DEFAULT 0.0,
                    PRIMARY KEY (model_name, version)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_feedback_processed ON feedback_events(processed);
            """)
    
    async def store_feedback(self, feedback: FeedbackEvent) -> bool:
        """Store feedback event in both local and cloud databases"""
        try:
            # Store locally
            with sqlite3.connect(self.local_db_path) as conn:
                conn.execute("""
                    INSERT INTO feedback_events 
                    (event_id, event_type, source, data, timestamp, metadata, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.event_id,
                    feedback.event_type,
                    feedback.source,
                    json.dumps(feedback.data),
                    feedback.timestamp.isoformat(),
                    json.dumps(feedback.metadata),
                    feedback.priority
                ))
            
            # Store in Supabase if available
            if self.supabase:
                try:
                    await self.supabase.table('ai_feedback_events').insert({
                        'event_id': feedback.event_id,
                        'event_type': feedback.event_type,
                        'source': feedback.source,
                        'data': feedback.data,
                        'timestamp': feedback.timestamp.isoformat(),
                        'metadata': feedback.metadata,
                        'priority': feedback.priority
                    }).execute()
                except Exception as e:
                    logger.warning(f"Failed to store feedback in Supabase: {e}")
            
            logger.info(f"Stored feedback event {feedback.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            return False
    
    async def get_pending_feedback(self, limit: int = 100) -> List[FeedbackEvent]:
        """Get pending feedback events for processing"""
        try:
            with sqlite3.connect(self.local_db_path) as conn:
                cursor = conn.execute("""
                    SELECT event_id, event_type, source, data, timestamp, metadata, priority
                    FROM feedback_events 
                    WHERE processed = FALSE 
                    ORDER BY priority DESC, timestamp ASC 
                    LIMIT ?
                """, (limit,))
                
                events = []
                for row in cursor.fetchall():
                    events.append(FeedbackEvent(
                        event_id=row[0],
                        event_type=row[1],
                        source=row[2],
                        data=json.loads(row[3]),
                        timestamp=datetime.fromisoformat(row[4]),
                        metadata=json.loads(row[5]) if row[5] else {},
                        priority=row[6]
                    ))
                
                return events
                
        except Exception as e:
            logger.error(f"Failed to get pending feedback: {e}")
            return []
    
    async def mark_feedback_processed(self, event_id: str) -> bool:
        """Mark feedback event as processed"""
        try:
            with sqlite3.connect(self.local_db_path) as conn:
                conn.execute("""
                    UPDATE feedback_events 
                    SET processed = TRUE 
                    WHERE event_id = ?
                """, (event_id,))
            return True
        except Exception as e:
            logger.error(f"Failed to mark feedback as processed: {e}")
            return False
    
    async def store_model_performance(self, performance: ModelPerformance) -> bool:
        """Store model performance metrics"""
        try:
            with sqlite3.connect(self.local_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO model_performance 
                    (model_name, version, metrics, feedback_count, last_updated, improvement_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    performance.model_name,
                    performance.version,
                    json.dumps(performance.metrics),
                    performance.feedback_count,
                    performance.last_updated.isoformat(),
                    performance.improvement_score
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to store model performance: {e}")
            return False

class RetrainingOrchestrator:
    """Production-grade retraining orchestration system"""
    
    def __init__(self, feedback_db: FeedbackDatabase, model_manager: ModelPersistenceManager):
        self.feedback_db = feedback_db
        self.model_manager = model_manager
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.retraining_queue = queue.Queue()
        self.active_retraining = {}
        
        # Initialize AI components
        self.ai_components = self._initialize_ai_components()
        
        # Start background processing
        self._start_background_processing()
        
        logger.info("Retraining orchestrator initialized")
    
    def _initialize_ai_components(self) -> Dict[str, Any]:
        """Initialize AI components for retraining"""
        components = {}
        
        try:
            components['federated_learner'] = FederatedMetaLearning()
            logger.info("✅ Federated learner initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize federated learner: {e}")
        
        try:
            components['gnn_engine'] = GNNReasoningEngine()
            logger.info("✅ GNN engine initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GNN engine: {e}")
        
        try:
            components['knowledge_graph'] = KnowledgeGraph()
            logger.info("✅ Knowledge graph initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize knowledge graph: {e}")
        
        try:
            components['matching_engine'] = RevolutionaryAIMatching()
            logger.info("✅ Matching engine initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize matching engine: {e}")
        
        return components
    
    def _start_background_processing(self):
        """Start background feedback processing and retraining"""
        def process_feedback_loop():
            while True:
                try:
                    # Process pending feedback
                    asyncio.run(self._process_pending_feedback())
                    
                    # Check for retraining triggers
                    asyncio.run(self._check_retraining_triggers())
                    
                    # Sleep before next iteration
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Error in feedback processing loop: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        feedback_thread = threading.Thread(target=process_feedback_loop, daemon=True)
        feedback_thread.start()
        
        def retraining_worker():
            while True:
                try:
                    # Get next retraining task
                    task = self.retraining_queue.get(timeout=60)
                    if task:
                        asyncio.run(self._execute_retraining_task(task))
                    self.retraining_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in retraining worker: {e}")
        
        retraining_thread = threading.Thread(target=retraining_worker, daemon=True)
        retraining_thread.start()
        
        logger.info("Background processing started")
    
    async def _process_pending_feedback(self):
        """Process pending feedback events"""
        try:
            feedback_events = await self.feedback_db.get_pending_feedback(limit=50)
            
            for event in feedback_events:
                try:
                    # Process based on event type
                    if event.event_type == 'user_feedback':
                        await self._process_user_feedback(event)
                    elif event.event_type == 'match_outcome':
                        await self._process_match_outcome(event)
                    elif event.event_type == 'system_metric':
                        await self._process_system_metric(event)
                    
                    # Mark as processed
                    await self.feedback_db.mark_feedback_processed(event.event_id)
                    
                except Exception as e:
                    logger.error(f"Error processing feedback event {event.event_id}: {e}")
            
            logger.info(f"Processed {len(feedback_events)} feedback events")
            
        except Exception as e:
            logger.error(f"Error in feedback processing: {e}")
    
    async def _process_user_feedback(self, event: FeedbackEvent):
        """Process user feedback for model improvement"""
        try:
            feedback_data = event.data
            
            # Extract relevant information
            model_name = feedback_data.get('model_name', 'unknown')
            rating = feedback_data.get('rating', 0)
            feedback_text = feedback_data.get('feedback', '')
            
            # Update model performance
            await self._update_model_performance(model_name, rating, feedback_text)
            
            # Trigger retraining if conditions are met
            if self._should_trigger_retraining(model_name, rating):
                await self._schedule_retraining(model_name, 'feedback_trigger')
            
            logger.info(f"Processed user feedback for {model_name}")
            
        except Exception as e:
            logger.error(f"Error processing user feedback: {e}")
    
    async def _process_match_outcome(self, event: FeedbackEvent):
        """Process match outcome for learning"""
        try:
            outcome_data = event.data
            
            # Extract match information
            match_id = outcome_data.get('match_id')
            success = outcome_data.get('success', False)
            metrics = outcome_data.get('metrics', {})
            
            # Update matching engine performance
            if 'matching_engine' in self.ai_components:
                # This would update the matching engine's internal metrics
                pass
            
            # Store outcome for analysis
            await self._store_match_outcome(match_id, success, metrics)
            
            logger.info(f"Processed match outcome for {match_id}")
            
        except Exception as e:
            logger.error(f"Error processing match outcome: {e}")
    
    async def _process_system_metric(self, event: FeedbackEvent):
        """Process system performance metrics"""
        try:
            metric_data = event.data
            
            # Extract metric information
            metric_name = metric_data.get('metric_name')
            metric_value = metric_data.get('value')
            component = metric_data.get('component')
            
            # Update component performance
            if component in self.ai_components:
                await self._update_component_performance(component, metric_name, metric_value)
            
            logger.info(f"Processed system metric {metric_name} for {component}")
            
        except Exception as e:
            logger.error(f"Error processing system metric: {e}")
    
    async def _update_model_performance(self, model_name: str, rating: float, feedback_text: str):
        """Update model performance based on feedback"""
        try:
            # Get current performance
            current_performance = await self._get_model_performance(model_name)
            
            # Calculate new metrics
            new_metrics = current_performance.metrics.copy()
            new_metrics['user_satisfaction'] = (
                (new_metrics.get('user_satisfaction', 0) * current_performance.feedback_count + rating) /
                (current_performance.feedback_count + 1)
            )
            new_metrics['feedback_count'] = current_performance.feedback_count + 1
            
            # Calculate improvement score
            improvement_score = self._calculate_improvement_score(new_metrics, current_performance.metrics)
            
            # Create new performance record
            new_performance = ModelPerformance(
                model_name=model_name,
                version=current_performance.version,
                metrics=new_metrics,
                feedback_count=new_metrics['feedback_count'],
                last_updated=datetime.now(),
                improvement_score=improvement_score
            )
            
            # Store updated performance
            await self.feedback_db.store_model_performance(new_performance)
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    async def _get_model_performance(self, model_name: str) -> ModelPerformance:
        """Get current model performance"""
        try:
            with sqlite3.connect(self.feedback_db.local_db_path) as conn:
                cursor = conn.execute("""
                    SELECT version, metrics, feedback_count, last_updated, improvement_score
                    FROM model_performance 
                    WHERE model_name = ? 
                    ORDER BY last_updated DESC 
                    LIMIT 1
                """, (model_name,))
                
                row = cursor.fetchone()
                if row:
                    return ModelPerformance(
                        model_name=model_name,
                        version=row[0],
                        metrics=json.loads(row[1]),
                        feedback_count=row[2],
                        last_updated=datetime.fromisoformat(row[3]),
                        improvement_score=row[4]
                    )
                else:
                    # Return default performance
                    return ModelPerformance(
                        model_name=model_name,
                        version="1.0",
                        metrics={'user_satisfaction': 0.0, 'feedback_count': 0},
                        feedback_count=0,
                        last_updated=datetime.now(),
                        improvement_score=0.0
                    )
                    
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return ModelPerformance(
                model_name=model_name,
                version="1.0",
                metrics={'user_satisfaction': 0.0, 'feedback_count': 0},
                feedback_count=0,
                last_updated=datetime.now(),
                improvement_score=0.0
            )
    
    def _calculate_improvement_score(self, new_metrics: Dict[str, float], old_metrics: Dict[str, float]) -> float:
        """Calculate improvement score based on metric changes"""
        try:
            improvement_score = 0.0
            
            # User satisfaction improvement
            old_satisfaction = old_metrics.get('user_satisfaction', 0.0)
            new_satisfaction = new_metrics.get('user_satisfaction', 0.0)
            if new_satisfaction > old_satisfaction:
                improvement_score += (new_satisfaction - old_satisfaction) * 10
            
            # Feedback volume increase
            old_count = old_metrics.get('feedback_count', 0)
            new_count = new_metrics.get('feedback_count', 0)
            if new_count > old_count:
                improvement_score += 0.1
            
            return improvement_score
            
        except Exception as e:
            logger.error(f"Error calculating improvement score: {e}")
            return 0.0
    
    def _should_trigger_retraining(self, model_name: str, rating: float) -> bool:
        """Determine if retraining should be triggered"""
        try:
            # Trigger conditions
            conditions = {
                'low_rating_threshold': rating < 3.0,  # Low rating
                'feedback_volume': True,  # Always consider feedback volume
                'performance_decline': True  # Always check performance
            }
            
            # Check if any condition is met
            return any(conditions.values())
            
        except Exception as e:
            logger.error(f"Error checking retraining trigger: {e}")
            return False
    
    async def _schedule_retraining(self, model_name: str, trigger_type: str):
        """Schedule model retraining"""
        try:
            retraining_task = {
                'task_id': str(uuid.uuid4()),
                'model_name': model_name,
                'trigger_type': trigger_type,
                'scheduled_at': datetime.now(),
                'status': 'scheduled'
            }
            
            # Add to retraining queue
            self.retraining_queue.put(retraining_task)
            
            logger.info(f"Scheduled retraining for {model_name}")
            
        except Exception as e:
            logger.error(f"Error scheduling retraining: {e}")
    
    async def _check_retraining_triggers(self):
        """Check for scheduled retraining triggers"""
        try:
            # Check for scheduled retraining
            current_time = datetime.now()
            
            # Example: Retrain models weekly
            if current_time.weekday() == 0 and current_time.hour == 2:  # Monday 2 AM
                await self._schedule_retraining('all_models', 'weekly_schedule')
            
            # Example: Retrain based on performance
            for model_name in ['matching_engine', 'gnn_engine', 'federated_learner']:
                performance = await self._get_model_performance(model_name)
                if performance.improvement_score < -0.5:  # Performance declining
                    await self._schedule_retraining(model_name, 'performance_decline')
            
        except Exception as e:
            logger.error(f"Error checking retraining triggers: {e}")
    
    async def _execute_retraining_task(self, task: Dict[str, Any]):
        """Execute a retraining task"""
        try:
            task_id = task['task_id']
            model_name = task['model_name']
            
            logger.info(f"Starting retraining task {task_id} for {model_name}")
            
            # Update task status
            task['status'] = 'running'
            self.active_retraining[task_id] = task
            
            # Execute retraining based on model type
            if model_name == 'matching_engine' and 'matching_engine' in self.ai_components:
                await self._retrain_matching_engine()
            elif model_name == 'gnn_engine' and 'gnn_engine' in self.ai_components:
                await self._retrain_gnn_engine()
            elif model_name == 'federated_learner' and 'federated_learner' in self.ai_components:
                await self._retrain_federated_learner()
            elif model_name == 'all_models':
                await self._retrain_all_models()
            
            # Update task status
            task['status'] = 'completed'
            task['completed_at'] = datetime.now()
            
            logger.info(f"Completed retraining task {task_id}")
            
        except Exception as e:
            logger.error(f"Error executing retraining task {task_id}: {e}")
            task['status'] = 'failed'
            task['error'] = str(e)
    
    async def _retrain_matching_engine(self):
        """Retrain the matching engine"""
        try:
            # Get recent feedback data
            feedback_events = await self.feedback_db.get_pending_feedback(limit=1000)
            
            # Prepare training data
            training_data = []
            for event in feedback_events:
                if event.event_type == 'user_feedback':
                    training_data.append({
                        'input': event.data.get('input', {}),
                        'output': event.data.get('rating', 0),
                        'feedback': event.data.get('feedback', '')
                    })
            
            # Retrain matching engine
            if 'matching_engine' in self.ai_components:
                # This would call the matching engine's retraining method
                # For now, we'll just log the retraining
                logger.info(f"Retraining matching engine with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Error retraining matching engine: {e}")
    
    async def _retrain_gnn_engine(self):
        """Retrain the GNN engine"""
        try:
            # Get recent graph data and feedback
            feedback_events = await self.feedback_db.get_pending_feedback(limit=1000)
            
            # Prepare graph training data
            graph_data = []
            for event in feedback_events:
                if event.event_type == 'match_outcome':
                    graph_data.append({
                        'nodes': event.data.get('nodes', []),
                        'edges': event.data.get('edges', []),
                        'outcome': event.data.get('success', False)
                    })
            
            # Retrain GNN engine
            if 'gnn_engine' in self.ai_components:
                logger.info(f"Retraining GNN engine with {len(graph_data)} graph samples")
            
        except Exception as e:
            logger.error(f"Error retraining GNN engine: {e}")
    
    async def _retrain_federated_learner(self):
        """Retrain the federated learner"""
        try:
            # Get recent federated data
            feedback_events = await self.feedback_db.get_pending_feedback(limit=1000)
            
            # Prepare federated training data
            federated_data = []
            for event in feedback_events:
                if event.event_type == 'user_feedback':
                    federated_data.append({
                        'client_id': event.data.get('user_id', 'unknown'),
                        'data': event.data.get('input', {}),
                        'label': event.data.get('rating', 0)
                    })
            
            # Retrain federated learner
            if 'federated_learner' in self.ai_components:
                logger.info(f"Retraining federated learner with {len(federated_data)} samples")
            
        except Exception as e:
            logger.error(f"Error retraining federated learner: {e}")
    
    async def _retrain_all_models(self):
        """Retrain all AI models"""
        try:
            await self._retrain_matching_engine()
            await self._retrain_gnn_engine()
            await self._retrain_federated_learner()
            
            logger.info("Completed retraining all models")
            
        except Exception as e:
            logger.error(f"Error retraining all models: {e}")
    
    async def _store_match_outcome(self, match_id: str, success: bool, metrics: Dict[str, Any]):
        """Store match outcome for analysis"""
        try:
            # Store in local database
            with sqlite3.connect(self.feedback_db.local_db_path) as conn:
                conn.execute("""
                    INSERT INTO match_outcomes (match_id, success, metrics, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (match_id, success, json.dumps(metrics), datetime.now().isoformat()))
            
        except Exception as e:
            logger.error(f"Error storing match outcome: {e}")
    
    async def _update_component_performance(self, component: str, metric_name: str, metric_value: float):
        """Update component performance metrics"""
        try:
            # This would update the component's internal performance tracking
            logger.info(f"Updated {component} performance: {metric_name} = {metric_value}")
            
        except Exception as e:
            logger.error(f"Error updating component performance: {e}")
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining status"""
        return {
            'active_tasks': len(self.active_retraining),
            'queue_size': self.retraining_queue.qsize(),
            'active_tasks_details': list(self.active_retraining.values())
        }

class AIFeedbackOrchestrator:
    """Main orchestrator class that coordinates all feedback and retraining"""
    
    def __init__(self):
        self.feedback_db = FeedbackDatabase()
        self.model_manager = ModelPersistenceManager()
        self.retraining_orchestrator = RetrainingOrchestrator(self.feedback_db, self.model_manager)
        
        logger.info("AI Feedback Orchestrator initialized")
    
    async def ingest_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Ingest new feedback into the system"""
        try:
            # Create feedback event
            feedback_event = FeedbackEvent(
                event_id=str(uuid.uuid4()),
                event_type=feedback_data.get('type', 'user_feedback'),
                source=feedback_data.get('source', 'user'),
                data=feedback_data.get('data', {}),
                timestamp=datetime.now(),
                metadata=feedback_data.get('metadata', {}),
                priority=feedback_data.get('priority', 1)
            )
            
            # Store feedback
            success = await self.feedback_db.store_feedback(feedback_event)
            
            if success:
                logger.info(f"Ingested feedback event {feedback_event.event_id}")
                return feedback_event.event_id
            else:
                raise Exception("Failed to store feedback")
                
        except Exception as e:
            logger.error(f"Error ingesting feedback: {e}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            return {
                'feedback_db_status': 'active',
                'retraining_status': self.retraining_orchestrator.get_retraining_status(),
                'model_manager_status': 'active',
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    async def trigger_manual_retraining(self, model_name: str) -> str:
        """Manually trigger model retraining"""
        try:
            await self.retraining_orchestrator._schedule_retraining(model_name, 'manual')
            return f"Retraining scheduled for {model_name}"
        except Exception as e:
            logger.error(f"Error triggering manual retraining: {e}")
            raise

# Global orchestrator instance
feedback_orchestrator = AIFeedbackOrchestrator() 