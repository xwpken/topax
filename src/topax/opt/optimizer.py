from typing import Dict, Type

class Optimizer:
    
    """Main optimizer class"""
    
    _algorithms: Dict[str, Type] = {}  
    
    def __init__(self, method: str, options: dict = None):
        self.method = method.lower()
        self.options = options or {}
        
        if self.method not in self._algorithms:
            available = list(self._algorithms.keys())
            raise ValueError(f"Unknown optimization method: {method}. Available: {available}")
        
        algorithm_class = self._algorithms[self.method]
        self.algorithm = algorithm_class(**self.options)
    
    def update(self, *args, **kwargs):
        return self.algorithm.update(*args, **kwargs)
    
    @classmethod
    def register_algorithm(cls, name: str):
        def decorator(algorithm_class):
            cls._algorithms[name.lower()] = algorithm_class
            return algorithm_class
        return decorator
    
    @classmethod
    def available_algorithms(cls):
        return list(cls._algorithms.keys())