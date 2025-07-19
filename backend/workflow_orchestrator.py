#!/usr/bin/env python3
"""
Production-Grade Workflow Orchestrator
Manages complex multi-service workflows with state management and error handling
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
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import traceback
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowStep:
    step_id: str
    name: str
    service_id: str
    method: str
    input_data: Dict[str, Any]
    output_mapping: Dict[str, str]
    retry_policy: Dict[str, Any]
    timeout: int
    dependencies: List[str]
    status: StepStatus
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class Workflow:
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    status: WorkflowStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]
    context: Dict[str, Any] = None

class WorkflowStateManager:
    """Manages workflow state persistence"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.workflow_prefix = "workflow:"
        self.step_prefix = "workflow_step:"
    
    def save_workflow(self, workflow: Workflow):
        """Save workflow state to Redis"""
        workflow_key = f"{self.workflow_prefix}{workflow.workflow_id}"
        workflow_data = {
            'workflow_id': workflow.workflow_id,
            'name': workflow.name,
            'description': workflow.description,
            'status': workflow.status.value,
            'created_at': workflow.created_at.isoformat(),
            'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
            'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None,
            'metadata': workflow.metadata,
            'context': workflow.context or {}
        }
        
        self.redis.hset(workflow_key, mapping=workflow_data)
        self.redis.expire(workflow_key, 86400)  # 24 hours TTL
    
    def save_step(self, workflow_id: str, step: WorkflowStep):
        """Save step state to Redis"""
        step_key = f"{self.step_prefix}{workflow_id}:{step.step_id}"
        step_data = {
            'step_id': step.step_id,
            'name': step.name,
            'service_id': step.service_id,
            'method': step.method,
            'input_data': json.dumps(step.input_data),
            'output_mapping': json.dumps(step.output_mapping),
            'retry_policy': json.dumps(step.retry_policy),
            'timeout': step.timeout,
            'dependencies': json.dumps(step.dependencies),
            'status': step.status.value,
            'result': json.dumps(step.result) if step.result else None,
            'error': step.error,
            'start_time': step.start_time.isoformat() if step.start_time else None,
            'end_time': step.end_time.isoformat() if step.end_time else None,
            'retry_count': step.retry_count,
            'max_retries': step.max_retries
        }
        
        self.redis.hset(step_key, mapping=step_data)
        self.redis.expire(step_key, 86400)  # 24 hours TTL
    
    def load_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Load workflow from Redis"""
        workflow_key = f"{self.workflow_prefix}{workflow_id}"
        workflow_data = self.redis.hgetall(workflow_key)
        
        if not workflow_data:
            return None
        
        # Load steps
        steps = self._load_steps(workflow_id)
        
        return Workflow(
            workflow_id=workflow_data['workflow_id'],
            name=workflow_data['name'],
            description=workflow_data['description'],
            steps=steps,
            status=WorkflowStatus(workflow_data['status']),
            created_at=datetime.fromisoformat(workflow_data['created_at']),
            started_at=datetime.fromisoformat(workflow_data['started_at']) if workflow_data['started_at'] else None,
            completed_at=datetime.fromisoformat(workflow_data['completed_at']) if workflow_data['completed_at'] else None,
            metadata=json.loads(workflow_data['metadata']) if isinstance(workflow_data['metadata'], str) else workflow_data['metadata'],
            context=json.loads(workflow_data['context']) if workflow_data['context'] else {}
        )
    
    def _load_steps(self, workflow_id: str) -> List[WorkflowStep]:
        """Load steps for a workflow"""
        pattern = f"{self.step_prefix}{workflow_id}:*"
        step_keys = self.redis.keys(pattern)
        steps = []
        
        for step_key in step_keys:
            step_data = self.redis.hgetall(step_key)
            if step_data:
                step = WorkflowStep(
                    step_id=step_data['step_id'],
                    name=step_data['name'],
                    service_id=step_data['service_id'],
                    method=step_data['method'],
                    input_data=json.loads(step_data['input_data']),
                    output_mapping=json.loads(step_data['output_mapping']),
                    retry_policy=json.loads(step_data['retry_policy']),
                    timeout=int(step_data['timeout']),
                    dependencies=json.loads(step_data['dependencies']),
                    status=StepStatus(step_data['status']),
                    result=json.loads(step_data['result']) if step_data['result'] else None,
                    error=step_data['error'],
                    start_time=datetime.fromisoformat(step_data['start_time']) if step_data['start_time'] else None,
                    end_time=datetime.fromisoformat(step_data['end_time']) if step_data['end_time'] else None,
                    retry_count=int(step_data['retry_count']),
                    max_retries=int(step_data['max_retries'])
                )
                steps.append(step)
        
        return steps

class WorkflowExecutor:
    """Executes workflow steps with dependency management"""
    
    def __init__(self, service_comm, state_manager: WorkflowStateManager):
        self.service_comm = service_comm
        self.state_manager = state_manager
        self.metrics = {
            'workflow_executions': Counter('workflow_executions_total', 'Total workflow executions', ['status']),
            'step_executions': Counter('workflow_step_executions_total', 'Total step executions', ['status']),
            'workflow_duration': Histogram('workflow_duration_seconds', 'Workflow execution time', ['workflow_name']),
            'step_duration': Histogram('workflow_step_duration_seconds', 'Step execution time', ['step_name'])
        }
    
    async def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute a complete workflow"""
        
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()
        self.state_manager.save_workflow(workflow)
        
        self.metrics['workflow_executions'].labels('started').inc()
        
        try:
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(workflow.steps)
            
            # Execute steps in dependency order
            results = {}
            completed_steps = set()
            
            while len(completed_steps) < len(workflow.steps):
                # Find ready steps (all dependencies completed)
                ready_steps = [
                    step for step in workflow.steps
                    if step.step_id not in completed_steps and
                    all(dep in completed_steps for dep in step.dependencies)
                ]
                
                if not ready_steps:
                    # Check for circular dependencies
                    raise Exception("Circular dependency detected in workflow")
                
                # Execute ready steps concurrently
                tasks = [self._execute_step(step, results) for step in ready_steps]
                step_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for step, result in zip(ready_steps, step_results):
                    if isinstance(result, Exception):
                        step.status = StepStatus.FAILED
                        step.error = str(result)
                        step.end_time = datetime.now()
                        self.state_manager.save_step(workflow.workflow_id, step)
                        
                        workflow.status = WorkflowStatus.FAILED
                        workflow.completed_at = datetime.now()
                        self.state_manager.save_workflow(workflow)
                        
                        self.metrics['workflow_executions'].labels('failed').inc()
                        raise result
                    else:
                        step.status = StepStatus.COMPLETED
                        step.result = result
                        step.end_time = datetime.now()
                        completed_steps.add(step.step_id)
                        results[step.step_id] = result
                        
                        self.state_manager.save_step(workflow.workflow_id, step)
            
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now()
            self.state_manager.save_workflow(workflow)
            
            duration = (workflow.completed_at - workflow.started_at).total_seconds()
            self.metrics['workflow_duration'].labels(workflow.name).observe(duration)
            self.metrics['workflow_executions'].labels('completed').inc()
            
            return {
                'workflow_id': workflow.workflow_id,
                'status': 'completed',
                'results': results,
                'duration': duration
            }
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.now()
            self.state_manager.save_workflow(workflow)
            
            self.metrics['workflow_executions'].labels('failed').inc()
            raise
    
    def _build_dependency_graph(self, steps: List[WorkflowStep]) -> Dict[str, List[str]]:
        """Build dependency graph for workflow steps"""
        graph = {}
        for step in steps:
            graph[step.step_id] = step.dependencies
        return graph
    
    async def _execute_step(self, step: WorkflowStep, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        
        step.start_time = datetime.now()
        step.status = StepStatus.RUNNING
        self.state_manager.save_step(step.workflow_id, step)
        
        self.metrics['step_executions'].labels('started').inc()
        
        try:
            # Prepare input data with previous results
            input_data = self._prepare_input_data(step.input_data, previous_results)
            
            # Execute with retry policy
            result = await self._execute_with_retry(
                lambda: self.service_comm.call_service(step.service_id, step.method, input_data),
                step.retry_policy,
                step
            )
            
            step.status = StepStatus.COMPLETED
            step.end_time = datetime.now()
            step.result = result
            
            duration = (step.end_time - step.start_time).total_seconds()
            self.metrics['step_duration'].labels(step.name).observe(duration)
            self.metrics['step_executions'].labels('completed').inc()
            
            return result
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.end_time = datetime.now()
            step.error = str(e)
            
            self.metrics['step_executions'].labels('failed').inc()
            raise
    
    def _prepare_input_data(self, input_template: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input data by substituting previous results"""
        input_data = {}
        for key, value in input_template.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                # Extract step_id and field from ${step_id.field}
                reference = value[2:-1]
                if '.' in reference:
                    step_id, field = reference.split('.', 1)
                    if step_id in previous_results:
                        input_data[key] = previous_results[step_id].get(field)
                else:
                    if reference in previous_results:
                        input_data[key] = previous_results[reference]
            else:
                input_data[key] = value
        return input_data
    
    async def _execute_with_retry(self, func: Callable, retry_policy: Dict[str, Any], step: WorkflowStep) -> Any:
        """Execute function with retry policy"""
        max_retries = retry_policy.get('max_retries', 3)
        backoff_factor = retry_policy.get('backoff_factor', 2)
        initial_delay = retry_policy.get('initial_delay', 1)
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                step.retry_count = attempt + 1
                
                if attempt < max_retries:
                    delay = initial_delay * (backoff_factor ** attempt)
                    await asyncio.sleep(delay)
                else:
                    break
        
        raise last_exception

class WorkflowOrchestrator:
    """Main workflow orchestrator"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.state_manager = WorkflowStateManager(self.redis_client)
        
        # Import service communication
        from real_service_communication import RealServiceCommunication
        self.service_comm = RealServiceCommunication("workflow-orchestrator")
        
        self.executor = WorkflowExecutor(self.service_comm, self.state_manager)
        self.workflows: Dict[str, Workflow] = {}
        
        self.metrics = {
            'orchestrator_requests': Counter('orchestrator_requests_total', 'Total orchestrator requests'),
            'orchestrator_errors': Counter('orchestrator_errors_total', 'Total orchestrator errors')
        }
    
    async def create_ai_company_pipeline(self, company_profile: Dict[str, Any]) -> str:
        """Create AI company analysis pipeline workflow"""
        
        workflow_id = str(uuid.uuid4())
        
        # Define workflow steps
        steps = [
            WorkflowStep(
                step_id="generate_listings",
                name="Generate AI Listings",
                service_id="ai_listings_generator",
                method="generate",
                input_data={"company_profile": company_profile},
                output_mapping={"listings": "predicted_outputs"},
                retry_policy={"max_retries": 3, "backoff_factor": 2, "initial_delay": 1},
                timeout=60,
                dependencies=[],
                status=StepStatus.PENDING,
                result=None,
                error=None,
                start_time=None,
                end_time=None
            ),
            WorkflowStep(
                step_id="find_matches",
                name="Find AI Matches",
                service_id="ai_matchmaking_service",
                method="find_partner_companies",
                input_data={"company_id": company_profile.get("id"), "material_data": "${generate_listings.listings}"},
                output_mapping={"matches": "partner_companies"},
                retry_policy={"max_retries": 3, "backoff_factor": 2, "initial_delay": 1},
                timeout=60,
                dependencies=["generate_listings"],
                status=StepStatus.PENDING,
                result=None,
                error=None,
                start_time=None,
                end_time=None
            ),
            WorkflowStep(
                step_id="calculate_pricing",
                name="Calculate AI Pricing",
                service_id="ai_pricing_service",
                method="calculate_prices",
                input_data={"matches": "${find_matches.matches}"},
                output_mapping={"pricing": "price_data"},
                retry_policy={"max_retries": 3, "backoff_factor": 2, "initial_delay": 1},
                timeout=30,
                dependencies=["find_matches"],
                status=StepStatus.PENDING,
                result=None,
                error=None,
                start_time=None,
                end_time=None
            ),
            WorkflowStep(
                step_id="analyze_financials",
                name="Analyze Financials",
                service_id="financial_analysis_engine",
                method="analyze_partnership",
                input_data={"matches": "${find_matches.matches}", "pricing": "${calculate_pricing.pricing}"},
                output_mapping={"analysis": "financial_analysis"},
                retry_policy={"max_retries": 3, "backoff_factor": 2, "initial_delay": 1},
                timeout=45,
                dependencies=["calculate_pricing"],
                status=StepStatus.PENDING,
                result=None,
                error=None,
                start_time=None,
                end_time=None
            ),
            WorkflowStep(
                step_id="regulatory_check",
                name="Regulatory Compliance Check",
                service_id="regulatory_compliance",
                method="check_compliance",
                input_data={"matches": "${find_matches.matches}", "analysis": "${analyze_financials.analysis}"},
                output_mapping={"compliance": "compliance_report"},
                retry_policy={"max_retries": 2, "backoff_factor": 2, "initial_delay": 1},
                timeout=30,
                dependencies=["analyze_financials"],
                status=StepStatus.PENDING,
                result=None,
                error=None,
                start_time=None,
                end_time=None
            )
        ]
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name="ai_company_pipeline",
            description="Complete AI pipeline for company analysis",
            steps=steps,
            status=WorkflowStatus.PENDING,
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            metadata={"company_id": company_profile.get("id")},
            context={"company_profile": company_profile}
        )
        
        # Save workflow
        self.state_manager.save_workflow(workflow)
        self.workflows[workflow_id] = workflow
        
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow"""
        try:
            # Load workflow
            workflow = self.state_manager.load_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            self.metrics['orchestrator_requests'].inc()
            
            # Execute workflow
            result = await self.executor.execute_workflow(workflow)
            return result
            
        except Exception as e:
            self.metrics['orchestrator_errors'].inc()
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status"""
        try:
            workflow = self.state_manager.load_workflow(workflow_id)
            if not workflow:
                return {"error": "Workflow not found"}
            
            return {
                'workflow_id': workflow.workflow_id,
                'name': workflow.name,
                'status': workflow.status.value,
                'created_at': workflow.created_at.isoformat(),
                'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
                'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None,
                'steps': [
                    {
                        'step_id': step.step_id,
                        'name': step.name,
                        'service_id': step.service_id,
                        'status': step.status.value,
                        'error': step.error,
                        'retry_count': step.retry_count
                    }
                    for step in workflow.steps
                ]
            }
            
        except Exception as e:
            logger.error(f"Get workflow status error: {e}")
            return {"error": str(e)}

# Initialize the orchestrator
workflow_orchestrator = WorkflowOrchestrator()

# Flask app for API endpoints
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Workflow Orchestrator',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/workflows/create', methods=['POST'])
async def create_workflow():
    """Create a new workflow"""
    try:
        data = request.json
        company_profile = data.get('company_profile', {})
        
        if not company_profile:
            return jsonify({'error': 'company_profile is required'}), 400
        
        workflow_id = await workflow_orchestrator.create_ai_company_pipeline(company_profile)
        return jsonify({'workflow_id': workflow_id})
        
    except Exception as e:
        logger.error(f"Create workflow error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/workflows/<workflow_id>/execute', methods=['POST'])
async def execute_workflow_endpoint(workflow_id):
    """Execute a workflow"""
    try:
        result = await workflow_orchestrator.execute_workflow(workflow_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Execute workflow error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/workflows/<workflow_id>/status', methods=['GET'])
async def get_workflow_status_endpoint(workflow_id):
    """Get workflow status"""
    try:
        status = await workflow_orchestrator.get_workflow_status(workflow_id)
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Get workflow status error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Workflow Orchestrator on port 5021...")
    app.run(host='0.0.0.0', port=5021, debug=False) 