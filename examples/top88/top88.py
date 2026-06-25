'''

2D MBB beam topology optimization

See https://link.springer.com/article/10.1007/s00158-010-0594-7

with minor modifications to external loadings

'''

import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)

import numpy as onp
import matplotlib.pyplot as plt

from jax_fem import logger
logger.setLevel("WARNING")

from topax.topology import Density
from topax.models import SIMP
from topax.transforms import ConvolutionFilter
from topax.opt import Optimizer

from mbb_beam import prep_fem


#%% SETUP
vf = 0.5 
rmin = 1.5
use_density_filter = False
use_sensitivity_filter = True

# SIMP model
xPhys2Mat = SIMP(E_max=1, E_min=1e-9, penal=3)

# forward model
Nx, Ny = 60, 20
Lx, Ly = Nx, Ny
fwd_pred, problem = prep_fem(Nx, Ny, Lx, Ly, xPhys2Mat)

# objective function
def J_total(xPhys):
    sol_list = fwd_pred(xPhys)
    compliance = problem.compute_compliance(sol_list[0])
    return compliance

# constraint function
def volume_constraints(xPhys):
    g = np.sum(xPhys) - vf * (xPhys.size)
    return g

# filters
conv2d = ConvolutionFilter(problem, options={'rmin': rmin})
rho_filter = conv2d if use_density_filter else lambda x: x
dJ_filter = conv2d if use_sensitivity_filter else lambda x: x[0]

# optimizer
OC = Optimizer(method='OC', options={'move':0.2, 'damping':0.5})

# topology 
x0 = vf * np.ones((Nx*Ny, 1))
topology = Density(problem, 
                   x=x0, transform=rho_filter,
                   obj=J_total, cons=[volume_constraints,])

#%% OPTIMIZATION LOOP
loop = 0
change = 1
xnew = x0
while change>0.01 and loop<500:
    loop += 1
    # evaluate
    J, dJ, c, dc = topology.evaluate()
    dc = dc[0]
    # filter
    xold = (topology.x).copy()
    dJ = dJ_filter((dJ, xold))
    # update
    xnew = OC.update(topology, dJ, dc, vf)
    topology.update(xnew)
    vol = topology.compute_volume_fraction()
    change = np.max(np.abs(xnew-xold))
    print(f' It.:{loop:5d}, Obj.:{J:11.4f}, Vol.:{vol:7.3f}, ch.:{change:7.3f}')
    # image
    field = onp.flip(topology.xPhys.reshape(Ny, Nx, order='F'),axis=0)
    plt.imshow(field, cmap='gray_r', vmin=0, vmax=1)
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    