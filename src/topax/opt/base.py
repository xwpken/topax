from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Base class for all optimization algorithms"""

    @abstractmethod
    def update(self, state, objective_grad, constraint_grads, **kwargs):
        """Update design variables"""
        pass

    def reset(self):
        """Reset algorithm state (optional)"""
        pass
