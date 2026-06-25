from .base import Topology

import jax.numpy as np

class Density(Topology):
    
    """Density-based topology optimization"""
    
    def __init__(self, problem, x, transform, obj, cons, info=None):
        super().__init__(problem, x, transform, obj, cons, info)
    
    def compute_volume_fraction(self, x=None):
        """Compute volume fraction"""
        if x is None:
            x = self.x
        x_phys = self.transform(x)
        # Compute material volume
        material_volume = np.sum(x_phys * self.element_volume)
        volfrac = material_volume / self.domain_volume
        return volfrac