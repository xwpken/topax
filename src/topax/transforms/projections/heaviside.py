import jax
import jax.numpy as np

from ..base import Transform

class Heaviside(Transform):
    
    """Heaviside projection"""
    
    _method_registry = {}  
    
    @classmethod
    def register_method(cls, name: str, citation: str, defaults: dict):
        """Register a new Heaviside projection method"""
        def decorator(method_func):
            cls._method_registry[name] = {
                'function': method_func,
                'citation': citation,
                'defaults': defaults
            }
            return method_func
        return decorator
    
    def __init__(self, method: str = 'guest', options: dict = None):
        options = options or {}
        
        if method not in self._method_registry:
            available = list(self._method_registry.keys())
            raise ValueError(f"Unknown method: {method}. Available: {available}")
        
        super().__init__(name=f"Heaviside_{method}")
        self.method = method
        self.options = options
        
        method_info = self._method_registry[method]
        self._citation = method_info['citation']

        for param, default in method_info['defaults'].items():
            setattr(self, param, options.get(param, default))
    
    def __call__(self, x: np.ndarray):
        """Apply projection using registered method"""
        method_func = self._method_registry[self.method]['function']
        return method_func(self, x)
    
    def value_and_grad(self, x: np.ndarray) :
        """Compute both projection value and gradient"""
        return jax.vmap(jax.value_and_grad(self.__call__))(x)
    
    def grad(self, x: np.ndarray):
        """Compute only the gradient of projection"""
        scalar_fn = lambda x:self(x)[0]
        return jax.vmap(jax.grad(scalar_fn))(x)
    
    def set_params(self, **params):
        """Update projection parameters"""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Unknown parameter: {param}")
        self.options.update(params)
    
    @property
    def citation(self):
        return self._citation
    
    def __repr__(self):
        method_info = self._method_registry[self.method]
        params = [f"{param}={getattr(self, param)}" for param in method_info['defaults']]
        return f"Heaviside(method='{self.method}', {', '.join(params)})"
    
    @classmethod
    def available_methods(cls):
        """Get list of available projection methods"""
        return list(cls._method_registry.keys())
    

@Heaviside.register_method('Guest2004', 'Guest2004', {'beta': 1.0})
def guest_projection(self, x: np.ndarray):
    """Guest's Heaviside projection: (1 - exp(-beta*x) + x*exp(-beta))"""
    return 1.0 - np.exp(-self.beta * x) + x * np.exp(-self.beta)

@Heaviside.register_method('Wang2010', 'Wang2010', {'beta': 1.0, 'eta': 0.5})
def wang_projection(self, x: np.ndarray):
    """Wang's Heaviside projection"""
    numerator = np.tanh(self.beta * self.eta) + np.tanh(self.beta * (x - self.eta))
    denominator = np.tanh(self.beta * self.eta) + np.tanh(self.beta * (1.0 - self.eta))
    return numerator / denominator
