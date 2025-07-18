from typing import Any, Dict, List, Optional, Callable

class BaseOptimizer:
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config

    def step(self):
        raise NotImplementedError('BaseOptimizer.step must be implemented by subclasses.')

    def zero_grad(self):
        raise NotImplementedError('BaseOptimizer.zero_grad must be implemented by subclasses.')

    def optimize(self, *args, **kwargs):
        raise NotImplementedError('BaseOptimizer.optimize must be implemented by subclasses.')

class OptimizationStrategy:
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}

class SearchSpace:
    def __init__(self, space: Dict[str, Any]):
        self.space = space 