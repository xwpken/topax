from typing import Any

import jax.numpy as np

from .base import Algorithm
from ..optimizer import Optimizer

@Optimizer.register_algorithm("oc")
class OC(Algorithm):
    
    """Optimality Criteria optimizer"""
    
    def __init__(self, move:float=0.2, 
                       damping:float=0.5, 
                       **kwargs:Any):
        self.move = move
        self.damping = damping
    
    def update(self, topology, dJ, dc, vf):
        """OC update implementation"""
        
        xold = topology.x
        compute_vf = topology.compute_volume_fraction  
        
        # Use bisection method to find appropriate lag multiplier
        lag = self._bisection(xold, dJ, dc, vf, compute_vf)
        
        # OC update formula
        xnew = self._oc_update(xold, dJ, dc, lag)
        
        # Apply move limits and boundary constraints
        return self._apply_constraints(xold, xnew)
    
    def _bisection(self, x, dJ, dc, vf, compute_vf):
        """Bisection method for lag multiplier calculation"""
        
        l1, l2 = 0, 1e9
        tol = 1e-3
        
        while (l2-l1)/(l1+l2) > tol:
            
            lmid = 0.5 * (l1 + l2)
            
            # Trial update
            x_test = self._oc_update(x, dJ, dc, lmid)
            x_test = self._apply_constraints(x, x_test)
            
            # Calculate volume fraction
            vf_test = compute_vf(x_test)
            
            if vf_test > vf:
                l1 = lmid
            else:
                l2 = lmid
                
        return lmid
    
    def _oc_update(self, x, dJ, dc, lag):
        """OC update formula"""
        return x * (-dJ / dc /lag) ** self.damping
    
    def _apply_constraints(self, xold, xnew):
        """Apply move limits and boundary constraints"""
        # Move limits
        lower_bound = np.maximum(0.0, xold - self.move)
        upper_bound = np.minimum(1.0, xold + self.move)
        
        xnew = np.clip(xnew, lower_bound, upper_bound)
        
        return xnew
    