
import jax.numpy as np

import numpy as onp

class OC:
    
    def __init__(self, move, volfrac, num_x, proj_fns=None, compute_volume=None):
        
        self.move = move
        self.volfrac = volfrac
        self.proj_fns = proj_fns if proj_fns is not None else lambda x: x
        # default: unit element volume (1)
        self.compute_volume = compute_volume if compute_volume is not None else lambda x: onp.sum(x)
        self.obj_volume = self.volfrac * self.compute_volume(onp.ones((num_x, 1)))
        
    def update(self, x, dJ, dV):
        
        l1, l2 = 0, 1e9
                   
        maximum = np.maximum
        minimum = np.minimum
        
        while (l2-l1)/(l1+l2) > 1e-3:
        
          lmid = 0.5*(l2+l1)
          xnew = maximum(0,
                         maximum(x - self.move,
                                 minimum(1,
                                         minimum(x + self.move, 
                                                 x * np.sqrt(-dJ/dV/lmid)))))
          
          xPhys = self.proj_fns(xnew)
          
          if self.compute_volume(xPhys) > self.obj_volume:
              l1 = lmid
          else:
              l2 = lmid
              
        return xnew
        
    def set_proj_fns(self, proj_fns):
        self.proj_fns = proj_fns
        

class MMA:
    
    pass