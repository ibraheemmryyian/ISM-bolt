"""
Production-Grade AI Feedback Orchestration System
Handles feedback ingestion, retraining triggers, and automated model updates
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.distributions import Categorical, Normal
import gym
from gym import spaces
# import stable_baselines3  # Removed - using CustomReplayBuffer
# from stable_baselines3 import PPO, SAC, TD3  # Removed - not used
# from stable_baselines3.common.buffers import ReplayBuffer  # Removed - using CustomReplayBuffer
try:
    # from stable_baselines3.common.callbacks import BaseCallback  # Removed - not used
    BaseCallback = None
    import logging
    logging.warning('stable_baselines3 is not installed or not available. Feedback orchestrator features will be limited.')
except ImportError:
    BaseCallback = None
    import logging
    logging.warning('stable_baselines3 is not installed or not available. Feedback orchestrator features will be limited.')
import joblib
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="`torch.utils._pytree._register_pytree_node` is deprecated")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*deprecated.*")

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any

from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/feedback/ingest', methods=['POST'])
def ingest_feedback_endpoint():
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        feedback_data = data.get('feedback_data')
        if not model_id or not feedback_data:
            return jsonify({'error': 'model_id and feedback_data are required'}), 400
        ai_feedback_orchestrator.ingest_feedback(model_id, feedback_data)
        return jsonify({'success': True, 'message': 'Feedback ingested and queued for processing'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

class FeedbackEventV1(BaseModel):
    # Core identifiers
    event_id: Optional[str] = Field(None, description="Unique event ID (UUID)")
    model_id: str = Field(..., description="ID of the model receiving feedback")
    user_id: Optional[str] = Field(None, description="ID of the user providing feedback")
    session_id: Optional[str] = Field(None, description="Session or request context ID")
    timestamp: Optional[float] = Field(None, description="Unix timestamp of feedback event")
    feedback_type: Optional[str] = Field(None, description="Type of feedback (explicit, implicit, system, etc.)")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for the feedback event")

    # Feedback content
    input_data: Optional[Any] = Field(None, description="Input data to the model")
    output_data: Optional[Any] = Field(None, description="Output data from the model")
    feedback_score: float = Field(..., ge=0.0, le=1.0, description="Normalized feedback score [0,1]")
    user_satisfaction: Optional[float] = Field(None, ge=0.0, le=1.0, description="User satisfaction score [0,1]")
    rating: Optional[int] = Field(None, ge=1, le=5, description="User rating (1-5 stars)")
    feedback_text: Optional[str] = Field(None, description="Free-text feedback from user")
    categories: Optional[List[str]] = Field(default_factory=list, description="Feedback categories/tags")
    improvement_suggestions: Optional[List[str]] = Field(default_factory=list, description="User suggestions for improvement")

    # Model/Inference metadata
    model_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model's confidence in its output")
    prediction_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Accuracy of prediction (if known)")
    response_time: Optional[float] = Field(None, description="Model response time in seconds")
    explanation_quality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality of model explanation")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance of output to user context")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="System-calculated confidence score")
    novelty_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Novelty of output")
    context_complexity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Complexity of the user/system context")
    user_expertise: Optional[float] = Field(None, ge=0.0, le=1.0, description="Estimated user expertise level")
    previous_interactions: Optional[int] = Field(None, description="Number of previous interactions with this model")
    time_of_day: Optional[float] = Field(None, description="Time of day as a float (0-24)")
    user_segment: Optional[str] = Field(None, description="User segment or cohort")
    request_type: Optional[str] = Field(None, description="Type of request (inference, retrain, etc.)")
    system_load: Optional[float] = Field(None, description="System load at time of inference")
    action_taken: Optional[str] = Field(None, description="Action taken by the model/agent")
    action_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in action taken")
    action_explanation_length: Optional[int] = Field(None, description="Length of explanation provided")
    action_complexity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Complexity of action taken")
    action_novelty: Optional[float] = Field(None, ge=0.0, le=1.0, description="Novelty of action taken")

    # Versioning
    schema_version: str = Field("1.0", description="Feedback schema version")

    @validator('timestamp', pre=True, always=True)
    def set_timestamp_now(cls, v):
        return v or datetime.utcnow().timestamp()

# ML Core imports
from ml_core.models import (
    ModelFactory,
    ModelArchitecture,
    ModelConfig
)
from ml_core.training import (
    ModelTrainer,
    TrainingConfig,
    TrainingMetrics
)
from ml_core.data_processing import (
    DataProcessor,
    DataValidator,
    FeedbackProcessor
)
from ml_core.optimization import (
    HyperparameterOptimizer,
    FeedbackOptimizer
)
from ml_core.monitoring import (
    MLMetricsTracker,
    FeedbackMonitor
)
from ml_core.utils import (
    ModelRegistry,
    FeedbackManager,
    ConfigManager,
    model_save_counter
)

class FeedbackDataset(Dataset):
    """Dataset for feedback learning with real data processing"""
    def __init__(self, 
                 feedback_data: List[Dict],
                 tokenizer=None,
                 max_length: int = 512):
        self.feedback_data = feedback_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Process feedback data
        self.processed_data = self._process_feedback_data()
        
        # Create training samples
        self.training_samples = self._create_training_samples()
    
    def _process_feedback_data(self) -> Dict:
        """Process feedback data with feature engineering"""
        processed = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'feedback_scores': [],
            'user_satisfaction': [],
            'improvement_suggestions': []
        }
        
        for feedback in self.feedback_data:
            # State representation
            state = self._extract_state_features(feedback)
            processed['states'].append(state)
            
            # Action representation
            action = self._extract_action_features(feedback)
            processed['actions'].append(action)
            
            # Reward calculation
            reward = self._calculate_reward(feedback)
            processed['rewards'].append(reward)
            
            # Next state (simplified)
            next_state = state  # In real implementation, this would be the next state
            processed['next_states'].append(next_state)
            
            # Feedback scores
            feedback_score = feedback.get('feedback_score', 0.5)
            processed['feedback_scores'].append(feedback_score)
            
            # User satisfaction
            satisfaction = feedback.get('user_satisfaction', 0.5)
            processed['user_satisfaction'].append(satisfaction)
            
            # Improvement suggestions
            suggestions = feedback.get('improvement_suggestions', [])        
            processed['improvement_suggestions'].append(suggestions)
        
        return processed
    
    def _extract_state_features(self, feedback: Dict) -> np.ndarray:
        """Extract state features from feedback"""
        features = [
            feedback.get('model_confidence',0.5),
            feedback.get('prediction_accuracy',0.5),
            feedback.get('response_time',1),
            feedback.get('user_expertise_level',0.5),
            feedback.get('context_complexity',0.5),
            feedback.get('previous_interactions', 0),
            feedback.get('time_of_day',0.5),
            feedback.get('user_segment',0.5),
            feedback.get('request_type',0.5),
            feedback.get('system_load', 0.5)       ]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_action_features(self, feedback: Dict) -> np.ndarray:
        """Extract action features from feedback"""
        features = [
            feedback.get('action_taken', 0),
            feedback.get('action_confidence',0.5),
            feedback.get('action_explanation_length', 0),
            feedback.get('action_complexity',0.5),
            feedback.get('action_novelty', 0.5)       ]
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self, feedback: Dict) -> float:
        """Calculate reward from feedback"""
        # Base reward from user satisfaction
        base_reward = feedback.get('user_satisfaction', 0.5)* 2 - 1  # Convert to [-1, 1]
        
        # Bonus for positive feedback
        if feedback.get('feedback_score', 0) > 0.7:
            base_reward += 0.2        
        # Penalty for negative feedback
        if feedback.get('feedback_score', 0) < 0.3:
            base_reward -= 0.3        
        # Bonus for improvement suggestions
        if feedback.get('improvement_suggestions'):
            base_reward += 0.1   
        return np.clip(base_reward, -1, 1)
    
    def _create_training_samples(self) -> List[Dict]:
        """Create training samples for RL"""
        samples = []
        
        for i in range(len(self.processed_data['states'])):
            sample = {
                'state': torch.FloatTensor(self.processed_data['states'][i]),
                'action': torch.FloatTensor(self.processed_data['actions'][i]),
                'reward': torch.FloatTensor([self.processed_data['rewards'][i]]),
                'next_state': torch.FloatTensor(self.processed_data['next_states'][i]),
                'feedback_score': torch.FloatTensor([self.processed_data['feedback_scores'][i]]),
                'user_satisfaction': torch.FloatTensor([self.processed_data['user_satisfaction'][i]])
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.training_samples)
    
    def __getitem__(self, idx):
        return self.training_samples[idx]

class FeedbackEnvironment(gym.Env):
    """Custom environment for feedback-based learning"""
    def __init__(self, 
                 state_dim: int = 10,
                 action_dim: int = 5,
                 max_steps: int = 10):
        super().__init__()
        import numpy as np
        import gym
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        # Use only [0, 1] bounds for stable-baselines3 compatibility
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(action_dim,), dtype=np.float32)
        # Environment state
        self.current_step = 0
        self.current_state = None
        self.feedback_history = []
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_state = np.random.uniform(0,1, self.state_dim).astype(np.float32)
        self.feedback_history = []
        
        return self.current_state
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        # Apply action
        self.current_state = self._apply_action(action)
        
        # Calculate reward based on feedback
        reward = self._calculate_reward(action)
        
        # Update step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Store feedback
        self.feedback_history.append({
            'state': self.current_state.copy(),
            'action': action.copy(),
            'reward': reward
        })
        
        info = {
            'feedback_score': reward,
            'step': self.current_step
        }
        
        return self.current_state, reward, done, info
    
    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        """Apply action to current state"""
        # Simple state transition model
        # In real implementation, this would be more sophisticated
        noise = np.random.normal(0,0.1, self.state_dim)
        new_state = self.current_state + 0.1 * action + noise
        return np.clip(new_state, -1, 1)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward based on action and state"""
        # Reward based on action quality and state desirability
        action_quality = 1 - np.mean(np.abs(action))  # Prefer smaller actions
        state_quality = np.mean(self.current_state)  # Prefer positive states
        
        reward = 0.5 * action_quality + 0.5 * state_quality
        return np.clip(reward, -1)
    
    def add_feedback(self, feedback: Dict):
        """Add external feedback to environment"""
        self.feedback_history.append(feedback)

class FeedbackActorCritic(nn.Module):
    """Actor-Critic network for feedback-based learning"""
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 learning_rate: float = 3e-4):
        super().__init__()
        import numpy as np
        import gym
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Output in [0,1]
        )
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        # Custom experience replay buffer (no stable-baselines3 dependency)
        self.replay_buffer = CustomReplayBuffer(
            buffer_size=10000,
            state_dim=state_dim,
            action_dim=action_dim
        )
        # NOTE: Ensure all state data passed to the buffer is a 1D float32 array in [0, 1]

class CustomReplayBuffer:
    """Custom replay buffer that doesn't depend on stable-baselines3 observation spaces"""
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.position = 0
        self.size = 0
        
        # Initialize buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
    
    def add(self, obs, next_obs, action, reward, done):
        """Add experience to buffer"""
        self.states[self.position] = obs
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_obs
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences"""
        if self.size < batch_size:
            batch_size = self.size
        
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return Batch(
            observations=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_states[indices],
            dones=self.dones[indices]
        )
    
    def __len__(self):
        return self.size

class Batch:
    """Simple batch class to mimic stable-baselines3 batch interface"""
    def __init__(self, observations, actions, rewards, next_observations, dones):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.dones = dones

    def add_to_buffer(self, state, next_state, action, reward, done):
        """Add experience to buffer with robust validation"""
        import numpy as np
        # Ensure all inputs are numpy arrays with correct shape and dtype
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # Ensure correct shapes
        if state.ndim == 1:
            state = state.reshape(1, -1)
        if next_state.ndim == 1:
            next_state = next_state.reshape(1, -1)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        
        # Clip to [0, 1] range for stability
        state = np.clip(state, 0, 1).astype(np.float32)
        next_state = np.clip(next_state, 0, 1).astype(np.float32)
        action = np.clip(action, 0, 1).astype(np.float32)
        
        self.replay_buffer.add(
            obs=state.flatten(),
            next_obs=next_state.flatten(),
            action=action.flatten(),
            reward=float(reward),
            done=bool(done)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through actor and critic"""
        # Actor
        action = self.actor(state)
        
        # Critic
        state_action = torch.cat([state, action], dim=1)
        value = self.critic(state_action)
        
        return action, value
    
    def get_action(self, state: torch.Tensor, exploration: float = 0.1) -> torch.Tensor:
        """Get action with exploration"""
        with torch.no_grad():
            action = self.actor(state)
            # Add exploration noise
            if np.random.random() < exploration:
                noise = torch.randn_like(action) * 0.1
                action = torch.clamp(action + noise, 0, 1)  # Clip to [0, 1]
            return action
    
    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """Update actor and critic networks"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        states = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_observations
        dones = batch.dones
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        current_q = self.critic(torch.cat([states, actions], dim=1))
        next_actions, _ = self.forward(next_states)
        next_q = self.critic(torch.cat([next_states, next_actions], dim=1))
        target_q = rewards + 0.99 * next_q * (1 - dones)
        critic_loss = F.mse_loss(current_q, target_q.detach())
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        new_actions, _ = self.forward(states)
        actor_loss = -self.critic(torch.cat([states, new_actions], dim=1)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }

class OnlineLearningModel(nn.Module):
    """Online learning model that adapts to feedback"""
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 learning_rate: float = 1e-3):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Online learning buffer
        self.online_buffer = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_history = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)
    
    def online_update(self, 
                     input_data: torch.Tensor,
                     target: torch.Tensor,
                     feedback_weight: float = 1.0) -> float:
        """Online update with feedback weighting"""
        # Forward pass
        prediction = self.forward(input_data)
        
        # Calculate loss with feedback weighting
        loss = F.mse_loss(prediction, target) * feedback_weight
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Store in buffer
        self.online_buffer.append({
            'input': input_data.detach(),
            'target': target.detach(),
            'prediction': prediction.detach(),
            'loss': loss.item()
        })
        
        # Track performance
        self.performance_history.append(loss.item())
        
        return loss.item()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        if not self.performance_history:
            return {}
        
        recent_losses = self.performance_history[-100:] # Last 100 updates
        
        return {
            'avg_loss': np.mean(recent_losses),
            'std_loss': np.std(recent_losses),
            'min_loss': np.min(recent_losses),
            'max_loss': np.max(recent_losses),
            'improvement_rate': self._calculate_improvement_rate()
        }
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate improvement rate over time"""
        if len(self.performance_history) < 20:
            return 0.0
        recent = np.mean(self.performance_history[-10])
        older = np.mean(self.performance_history[-20])
        
        if older == 0:
            return 0.0   
        return (older - recent) / older  # Positive means improvement

class AdaptiveModelUpdater:
    """Adaptive model updater with feedback integration"""
    def __init__(self, 
                 base_model: nn.Module,
                 feedback_model: OnlineLearningModel,
                 update_threshold: float = 0.1):
        self.base_model = base_model
        self.feedback_model = feedback_model
        self.update_threshold = update_threshold
        
        # Update history
        self.update_history = []
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
    
    def should_update(self, feedback_score: float) -> bool:
        """Determine if model should be updated based on feedback"""
        # Check if feedback indicates significant issues
        if feedback_score < self.update_threshold:
            return True
        
        # Check performance degradation
        if self.performance_tracker.is_degrading():
            return True
        
        return False   
    def update_model(self, feedback_data: Dict) -> Dict:
        """Update model based on feedback"""
        try:
            # Extract feedback features
            feedback_features = self._extract_feedback_features(feedback_data)
            
            # Get current performance
            current_performance = self.performance_tracker.get_current_performance()
            
            # Calculate target improvement
            target_performance = current_performance * (1 + feedback_data.get('desired_improvement', 0.1))

            # Update feedback model
            feedback_tensor = torch.FloatTensor(feedback_features).unsqueeze(0)
            target_tensor = torch.FloatTensor([target_performance])
            
            loss = self.feedback_model.online_update(feedback_tensor, target_tensor)
            
            # Apply updates to base model if needed
            if self.should_update(feedback_data.get('feedback_score', 0.5)):
                self._apply_feedback_updates(feedback_data)
            
            # Record update
            update_record = {
                'timestamp': datetime.now(),
                'feedback_score': feedback_data.get('feedback_score', 0.5),
                'loss': loss,
                'performance_change': target_performance - current_performance
            }
            self.update_history.append(update_record)
            
            return {
                'updated': True,
                'loss': loss,
                'performance_change': update_record['performance_change']
            }
            
        except Exception as e:
            logging.error(f"Error updating model: {e}")
            return {
                'updated': False,
                'error': str(e)
            }
    
    def _extract_feedback_features(self, feedback_data: Dict) -> np.ndarray:
        """Extract features from feedback data"""
        features = [
            feedback_data.get('feedback_score',0.5),
            feedback_data.get('user_satisfaction',0.5),
            feedback_data.get('prediction_accuracy',0.5),
            feedback_data.get('response_time',1),
            feedback_data.get('explanation_quality',0.5),
            feedback_data.get('relevance_score',0.5),
            feedback_data.get('confidence_score',0.5),
            feedback_data.get('novelty_score', 0.5)       ]
        
        return np.array(features, dtype=np.float32)   
    def _apply_feedback_updates(self, feedback_data: Dict):
        """Apply feedback-based updates to base model"""
        # This would implement specific update strategies
        # For now, we'll use a simplified approach
        
        # Get feedback suggestions
        suggestions = feedback_data.get('improvement_suggestions', [])
        
        # Apply learning rate adjustment
        if 'reduce_learning_rate' in suggestions:
            for param_group in self.base_model.optimizer.param_groups:
                param_group['lr'] *= 0.9        
        # Apply regularization adjustment
        if 'increase_regularization' in suggestions:
            # This would modify dropout rates or add weight decay
            pass

class PerformanceTracker:
    """Performance tracking for adaptive updates"""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history = deque(maxlen=window_size)
        self.degradation_threshold = 0.1
    
    def record_performance(self, performance: float):
        """Record performance metric"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'performance': performance
        })
    
    def get_current_performance(self) -> float:
        """Get current performance"""
        if not self.performance_history:
            return 0.0   
        recent_performances = [entry['performance'] for entry in list(self.performance_history)[-10:]]
        return np.mean(recent_performances)
    
    def is_degrading(self) -> bool:
        """Check if performance is degrading"""
        if len(self.performance_history) < 20:
            return False
    
        recent_performances = [entry['performance'] for entry in list(self.performance_history)[-10]]
        older_performances = [entry['performance'] for entry in list(self.performance_history)[-20]]
        
        recent_avg = np.mean(recent_performances)
        older_avg = np.mean(older_performances)
        
        degradation = (older_avg - recent_avg) / older_avg if older_avg > 0 else 0   
        return degradation > self.degradation_threshold

class AIFeedbackOrchestrator:
    """Real ML-powered feedback orchestrator with advanced learning capabilities"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model_registry = ModelRegistry()
        self.feedback_manager = FeedbackManager()
        self.metrics_tracker = MLMetricsTracker()
        self.feedback_monitor = FeedbackMonitor()
        self.config_manager = ConfigManager()
        
        # Initialize models
        self.feedback_models = {}
        self.online_models = {}
        self.adaptive_updaters = {}
        
        # Feedback configuration
        self.feedback_config = {
            'batch_size':32, 
            'learning_rate':1e-4, 
            'update_frequency': 10,
            'feedback_threshold': 0.7,
            'exploration_rate': 0.1,
            'buffer_size': 10000
        }
        
        # Initialize RL environment
        self.feedback_environment = FeedbackEnvironment()
        
        # Initialize RL agent
        self.rl_agent = FeedbackActorCritic(
            state_dim=10,
            action_dim=5,
            learning_rate=3e-4
        ).to(self.device)
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Feedback history
        self.feedback_history = deque(maxlen=100)
    
    def ingest_feedback(self, model_id: str, feedback_data: Dict) -> None:
        """Validate and publish feedback to the Redis Stream as FeedbackEventV1."""
        try:
            validated_feedback = self._validate_feedback(feedback_data)
            feedback_event = FeedbackEventV1(model_id=model_id, **validated_feedback)
            producer = FeedbackEventProducer()
            producer.publish(feedback_event)
            self.logger.info(f"Published feedback event for model {model_id} to Redis Stream.")
        except Exception as e:
            self.logger.error(f"Error ingesting feedback: {e}")
            raise

    def _ml_update_trust_score(self, model_id: str, feedback_score: float):
        """ML-driven trust/confidence update using exponential moving average (EMA)."""
        # Get current trust score from arbiter
        try:
            resp = requests.get("http://localhost:8000/arbiter/trust-scores")
            trust_scores = resp.json().get("trust_scores", {})
            current_score = trust_scores.get(model_id, 0.5)
        except Exception:
            current_score = 0.5
        # EMA update
        alpha = 0.1
        new_score = (1 - alpha) * current_score + alpha * feedback_score
        # Send update to arbiter
        try:
            requests.post("http://localhost:8000/arbiter/update-trust", json={
                "model_id": model_id,
                "new_score": new_score
            })
        except Exception as e:
            self.logger.error(f"Failed to update trust score for {model_id}: {e}")

    async def process_feedback(self,
                             model_id: str,
                             feedback_data: Dict) -> Dict:
        """Process feedback and update models"""
        try:
            self.logger.info(f"Processing feedback for model {model_id}")
            
            # Validate feedback
            validated_feedback = self._validate_feedback(feedback_data)
            
            # Store feedback
            self.feedback_history.append({
                'model_id': model_id,
                'feedback': validated_feedback,
                'timestamp': datetime.now()
            })
            
            # Update performance tracking
            self.performance_tracker.record_performance(
                validated_feedback.get('feedback_score', 0.5)   )
            
            # Process with RL agent
            rl_result = await self._process_with_rl(model_id, validated_feedback)
            
            # Update online learning models
            online_result = await self._update_online_models(model_id, validated_feedback)
            
            # Adaptive model updates
            adaptive_result = await self._adaptive_model_update(model_id, validated_feedback)
            
            # After processing feedback, update trust/confidence
            feedback_score = validated_feedback.get('feedback_score', 0.5)
            self._ml_update_trust_score(model_id, feedback_score)
            
            # Track metrics
            self.metrics_tracker.record_feedback_metrics({
               'model_id': model_id,
               'feedback_score': validated_feedback.get('feedback_score', 0.5),
               'user_satisfaction': validated_feedback.get('user_satisfaction', 0.5),
               'rl_loss': rl_result.get('loss', 0),
               'online_loss': online_result.get('loss', 0),
               'adaptive_updated': adaptive_result.get('updated', False)
            })
            
            return {
                'model_id': model_id,
                'feedback_processed': True,
                'rl_result': rl_result,
                'online_result': online_result,
                'adaptive_result': adaptive_result,
                'performance_metrics': self.performance_tracker.get_current_performance()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing feedback: {e}")
            raise
    
    async def _process_with_rl(self, model_id: str, feedback_data: Dict) -> Dict:
        """Process feedback with reinforcement learning"""
        try:
            # Convert feedback to environment state
            state = self._feedback_to_state(feedback_data)
            
            # Get action from RL agent
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.rl_agent.get_action(state_tensor, self.feedback_config['exploration_rate'])
            
            # Execute action in environment
            next_state, reward, done, info = self.feedback_environment.step(action.cpu().numpy())
            
            # Store experience in replay buffer
            self.rl_agent.add_to_buffer(
                state=state,
                next_state=next_state,
                action=action.cpu().numpy(),
                reward=reward,
                done=done
            )
            
            # Update RL agent
            if len(self.rl_agent.replay_buffer) > self.feedback_config['batch_size']:
                update_result = self.rl_agent.update(self.feedback_config['batch_size'])
            else:
                update_result = {}
            
            return {
                'action_taken': action.cpu().numpy().tolist(),
                'reward_received': reward,
                'next_state': next_state.tolist(),
                'actor_loss': update_result.get('actor_loss', 0),
                'critic_loss': update_result.get('critic_loss', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error in RL processing: {e}")
            return {'error': str(e)}
    
    async def _update_online_models(self, model_id: str, feedback_data: Dict) -> Dict:
        """Update online learning models"""
        try:
            # Get or create online model
            if model_id not in self.online_models:
                self.online_models[model_id] = OnlineLearningModel(
                    input_dim=8,  # Feedback features
                    output_dim=1,  # Performance prediction
                    learning_rate=1e-3                ).to(self.device)
            
            online_model = self.online_models[model_id]
            
            # Prepare training data
            feedback_features = self._extract_feedback_features(feedback_data)
            target_performance = feedback_data.get('feedback_score', 0.5)      
            # Online update
            input_tensor = torch.FloatTensor(feedback_features).unsqueeze(0).to(self.device)
            target_tensor = torch.FloatTensor([target_performance]).to(self.device)
            
            loss = online_model.online_update(input_tensor, target_tensor)
            
            return {
                'loss': loss,
                'performance_metrics': online_model.get_performance_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating online models: {e}")
            return {'error': str(e)}
    
    async def _adaptive_model_update(self, model_id: str, feedback_data: Dict) -> Dict:
        """Perform adaptive model updates"""
        try:
            # Get or create adaptive updater
            if model_id not in self.adaptive_updaters:
                # Get base model from registry
                base_model = self.model_registry.get_model(model_id)
                if not base_model:
                    raise ValueError(f"Model {model_id} not found in registry")
                
                # Create feedback model
                feedback_model = OnlineLearningModel(
                    input_dim=8,
                    output_dim=1,
                    learning_rate=1e-3                ).to(self.device)
                
                self.adaptive_updaters[model_id] = AdaptiveModelUpdater(
                    base_model=base_model,
                    feedback_model=feedback_model
                )
            
            adaptive_updater = self.adaptive_updaters[model_id]
            
            # Perform adaptive update
            update_result = adaptive_updater.update_model(feedback_data)
            
            return update_result
            
        except Exception as e:
            self.logger.error(f"Error in adaptive model update: {e}")
            return {'error': str(e)}
    
    def _validate_feedback(self, feedback_data: Dict) -> Dict:
        """Validate feedback data"""
        required_fields = ['feedback_score', 'user_satisfaction']
        
        for field in required_fields:
            if field not in feedback_data:
                feedback_data[field] = 0.5
        
        # Ensure scores are in valid range
        feedback_data['feedback_score'] = np.clip(feedback_data['feedback_score'], 0, 1)
        feedback_data['user_satisfaction'] = np.clip(feedback_data['user_satisfaction'], 0, 1)
        
        return feedback_data
    
    def _feedback_to_state(self, feedback_data: Dict) -> np.ndarray:
        """Convert feedback to RL state"""
        state = [
            feedback_data.get('feedback_score',0.5),
            feedback_data.get('user_satisfaction',0.5),
            feedback_data.get('prediction_accuracy',0.5),
            feedback_data.get('response_time',1),
            feedback_data.get('explanation_quality',0.5),
            feedback_data.get('relevance_score',0.5),
            feedback_data.get('confidence_score',0.5),
            feedback_data.get('novelty_score', 0.5),
            feedback_data.get('context_complexity',0.5),
            feedback_data.get('user_expertise', 0.5)       ]
        
        return np.array(state, dtype=np.float32)
    
    def _extract_feedback_features(self, feedback_data: Dict) -> np.ndarray:
        """Extract features for online learning"""
        features = [
            feedback_data.get('feedback_score',0.5),
            feedback_data.get('user_satisfaction',0.5),
            feedback_data.get('prediction_accuracy',0.5),
            feedback_data.get('response_time',1),
            feedback_data.get('explanation_quality',0.5),
            feedback_data.get('relevance_score',0.5),
            feedback_data.get('confidence_score',0.5),
            feedback_data.get('novelty_score', 0.5)       ]
        
        return np.array(features, dtype=np.float32)    
    async def get_feedback_insights(self, model_id: str) -> Dict:
        """Get insights from feedback data"""
        try:
            # Filter feedback for specific model
            model_feedback = [
                entry for entry in self.feedback_history
                if entry['model_id'] == model_id
            ]
            
            if not model_feedback:
                return {'message':'No feedback data available'}
            
            # Calculate insights
            feedback_scores = [entry['feedback']['feedback_score'] for entry in model_feedback]
            satisfaction_scores = [entry['feedback']['user_satisfaction'] for entry in model_feedback]
            
            insights = {
                'total_feedback_count': len(model_feedback),
                'avg_feedback_score': np.mean(feedback_scores),
                'avg_satisfaction': np.mean(satisfaction_scores),
                'feedback_trend': self._calculate_feedback_trend(feedback_scores),
                'improvement_areas': self._identify_improvement_areas(model_feedback),
                'performance_metrics': self.performance_tracker.get_current_performance()
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting feedback insights: {e}")
            return {'error': str(e)}
    
    def _calculate_feedback_trend(self, scores: List[float]) -> str:
        """Calculate feedback trend"""
        if len(scores) < 10:
            return 'insufficient_data'   
        recent_scores = scores[-10:]
        older_scores = scores[-20:-10] if len(scores) >= 20 else scores[:-10]
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        if recent_avg > older_avg + 0.1:
            return 'improving'
        elif recent_avg < older_avg - 0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _identify_improvement_areas(self, feedback_data: List[Dict]) -> List[str]:
        """Identify areas for improvement"""
        improvement_areas = []
        
        # Analyze common feedback patterns
        low_scores = [entry for entry in feedback_data if entry['feedback']['feedback_score'] < 0.5]
        
        if len(low_scores) > len(feedback_data) * 0.3:  # More than 30% low scores
            improvement_areas.append('overall_quality')
        
        # Check specific areas
        accuracy_scores = [entry['feedback'].get('prediction_accuracy',0.5) for entry in feedback_data]
        if np.mean(accuracy_scores) < 0.7:
            improvement_areas.append('prediction_accuracy')
        
        response_times = [entry['feedback'].get('response_time',1) for entry in feedback_data]
        if np.mean(response_times) > 2.0:
            improvement_areas.append('response_time')
        
        return improvement_areas
    
    async def get_system_health(self) -> Dict:
        """Get system health metrics"""
        try:
            health_metrics = {
                'status': 'healthy',
                'device':str(self.device),
                'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'feedback_models_count': len(self.feedback_models),
                'online_models_count': len(self.online_models),
                'adaptive_updaters_count': len(self.adaptive_updaters),
                'feedback_history_size': len(self.feedback_history),
                'feedback_monitor_status': self.feedback_monitor.get_status(),
                'metrics_tracker_status': self.metrics_tracker.get_status(),
                'performance_metrics': {
                    'current_performance': self.performance_tracker.get_current_performance(),
                    'rl_agent_performance': self._get_rl_agent_performance(),
                    'online_learning_performance': self._get_online_learning_performance()
                }
            }
            
            return health_metrics
                
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {'status':'error', 'error': str(e)}
    
    def _get_rl_agent_performance(self) -> Dict:
        """Get RL agent performance metrics"""
        try:
            # Get recent rewards from environment
            recent_rewards = [entry['reward'] for entry in self.feedback_environment.feedback_history[-50:]]
            if not recent_rewards:
                return {'avg_reward': 0}      
            return {
                'avg_reward': np.mean(recent_rewards),
                'std_reward': np.std(recent_rewards),
                'min_reward': np.min(recent_rewards),
                'max_reward': np.max(recent_rewards)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_online_learning_performance(self) -> Dict:
        """Get online learning performance metrics"""
        try:
            if not self.online_models:
                return {'avg_loss': 0}      
            all_metrics = []
            for model in self.online_models.values():
                metrics = model.get_performance_metrics()
                if metrics:
                    all_metrics.append(metrics)
            
            if not all_metrics:
                return {'avg_loss': 0}      
            return {
                'avg_loss': np.mean([m.get('avg_loss', 0) for m in all_metrics]),
                'avg_improvement_rate': np.mean([m.get('improvement_rate', 0) for m in all_metrics])
            }
            
        except Exception as e:
            return {'error': str(e)}

class FeedbackEventProducer:
    """Publishes FeedbackEventV1 objects to a Redis Stream for feedback event bus."""
    def __init__(self, stream_name: str = 'feedback_events'):
        self.stream_name = stream_name
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            decode_responses=True
        )

    def publish(self, feedback_event: FeedbackEventV1):
        try:
            event_data = feedback_event.dict()
            event_json = json.dumps(event_data, default=str)
            self.redis_client.xadd(self.stream_name, {'event': event_json})
        except Exception as e:
            logging.error(f"Failed to publish feedback event: {e}")
            raise

class FeedbackEventConsumer:
    """Consumes FeedbackEventV1 events from a Redis Stream and processes them via the orchestrator."""
    def __init__(self, orchestrator: 'AIFeedbackOrchestrator', stream_name: str = 'feedback_events', group_name: str = 'feedback_consumers', consumer_name: str = None):
        self.stream_name = stream_name
        self.group_name = group_name
        self.consumer_name = consumer_name or f'consumer_{os.getpid()}'
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            decode_responses=True
        )
        self.orchestrator = orchestrator
        self._ensure_group_exists()

    def _ensure_group_exists(self):
        try:
            self.redis_client.xgroup_create(self.stream_name, self.group_name, id='0', mkstream=True)
        except redis.exceptions.ResponseError as e:
            if 'BUSYGROUP' in str(e):
                pass  # Group already exists
            else:
                logging.error(f"Error creating Redis stream group: {e}")
                raise

    def consume(self, block_ms: int = 5000, count: int = 10):
        while True:
            try:
                entries = self.redis_client.xreadgroup(self.group_name, self.consumer_name, {self.stream_name: '>'}, count=count, block=block_ms)
                for stream, events in entries:
                    for event_id, event_data in events:
                        try:
                            event_json = event_data.get('event')
                            if not event_json:
                                continue
                            event_dict = json.loads(event_json)
                            feedback_event = FeedbackEventV1(**event_dict)
                            # Process feedback event via orchestrator
                            asyncio.run(self.orchestrator.process_feedback(feedback_event.model_id, feedback_event.dict()))
                            # Acknowledge event
                            self.redis_client.xack(self.stream_name, self.group_name, event_id)
                        except Exception as e:
                            logging.error(f"Error processing feedback event: {e}")
            except Exception as e:
                logging.error(f"Error consuming from Redis stream: {e}")
                time.sleep(2)

# Initialize service
ai_feedback_orchestrator = AIFeedbackOrchestrator() 

if __name__ == "__main__":
    print("🚀 Starting AI Feedback Orchestrator on port 5015...")
    app.run(host='0.0.0.0', port=5015, debug=False) 