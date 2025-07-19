"""
ML Core Training: Training utilities for all model types
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from ml_core.monitoring import log_metrics

@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = 'adam'
    weight_decay: float = 0.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    gradient_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = None
    scheduler: Optional[str] = None
    scheduler_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    early_stopping_patience: Optional[int] = None
    early_stopping_metric: Optional[str] = None
    save_best: bool = True
    log_interval: int = 10
    seed: int = 42

@dataclass
class TrainingMetrics:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    best_val_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)

class ModelTrainer:
    def __init__(self, model, config, loss_fn):
        if model is None or config is None or loss_fn is None:
            raise ValueError('ModelTrainer requires model, config, and loss_fn arguments.')
        self.model = model
        self.loss_fn = loss_fn
        self.metrics_fn = None # Placeholder, will be set later if needed
        
        # Handle config as either dict or TrainingConfig object
        if isinstance(config, dict):
            self.config = TrainingConfig(**config)
        else:
            self.config = config
            
        self.device = self.config.device
        self.model.to(self.device)
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.metrics = TrainingMetrics()
        self.best_state_dict = None

    def _get_optimizer(self):
        if self.config.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _get_scheduler(self):
        if self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, **self.config.scheduler_params)
        elif self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **self.config.scheduler_params)
        elif self.config.scheduler is None:
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, callbacks: Optional[List[Callable]] = None):
        best_val_metric = float('inf')
        best_epoch = 0
        for epoch in range(self.config.epochs):
            self.model.train()
            train_losses = []
            correct = 0
            total = 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                if self.config.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                train_losses.append(loss.item())
                if self.metrics_fn:
                    acc = self.metrics_fn(outputs, targets)
                    correct += acc * inputs.size(0)
                total += inputs.size(0)
                if (i + 1) % self.config.log_interval == 0:
                    log_metrics({'train_loss': np.mean(train_losses)}, step=epoch * len(train_loader) + i)
            avg_train_loss = np.mean(train_losses)
            self.metrics.train_loss.append(avg_train_loss)
            if self.metrics_fn:
                train_acc = correct / total
                self.metrics.train_accuracy.append(train_acc)
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                self.metrics.val_loss.append(val_loss)
                self.metrics.val_accuracy.append(val_acc)
                if val_loss < best_val_metric:
                    best_val_metric = val_loss
                    best_epoch = epoch
                    if self.config.save_best:
                        self.best_state_dict = self.model.state_dict()
            if self.scheduler:
                self.scheduler.step()
            if callbacks:
                for cb in callbacks:
                    cb(self.model, epoch, self.metrics)
        self.metrics.best_val_metric = best_val_metric
        self.metrics.best_epoch = best_epoch
        if self.config.save_best and self.best_state_dict:
            self.model.load_state_dict(self.best_state_dict)
        return self.metrics

    def evaluate(self, data_loader: DataLoader):
        self.model.eval()
        losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                losses.append(loss.item())
                if self.metrics_fn:
                    acc = self.metrics_fn(outputs, targets)
                    correct += acc * inputs.size(0)
                total += inputs.size(0)
        avg_loss = np.mean(losses)
        avg_acc = correct / total if self.metrics_fn else None
        return avg_loss, avg_acc 

def train_supervised(model, dataloader, optimizer, criterion, epochs=10, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Supervised] Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader):.4f}")
    return model

def train_rl(agent, env, episodes=100):
    """Train a reinforcement learning agent in the given environment."""
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        print(f"[RL] Episode {episode+1}/{episodes} Total Reward: {total_reward:.2f}")
    return agent, rewards 

def train_gnn(*args, **kwargs):
    pass  # TODO: Replace with real implementation 