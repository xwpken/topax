"""

2D compliant mechanism (force inverter) topology optimization

See Appendix 5.1.5 of Bendsøe & Sigmund, "Topology Optimization", 2004.

"""

import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)

import numpy as onp
import matplotlib.pyplot as plt

from jax_fem import logger
logger.setLevel("WARNING")

from topax.top import Density
from topax.mat import SIMP
from topax.tfp import Conv
from topax.opt import OC

from problem import prep_fem


#%% SETUP
vf = 0.3
rmin = 1.5
use_density_filter = True
use_sensitivity_filter = True

# SIMP model
xPhys2Mat = SIMP(E_max=1, E_min=1e-3, penal=3)

# forward model
Nx, Ny = 60, 30
Lx, Ly = Nx, Ny
fwd_pred, problem = prep_fem(Nx, Ny, Lx, Ly, xPhys2Mat)

output_node = problem.output_node

# objective function
def J_total(xPhys):
    sol_list = fwd_pred(xPhys)
    u_out = sol_list[0][output_node, 0]
    return u_out

# constraint function
def volume_constraint(xPhys):
    g = np.sum(xPhys) - vf * (xPhys.size)
    return g

# filters
conv2d = Conv(problem, rmin=rmin)
rho_filter = conv2d.density() if use_density_filter else lambda x: x
sens_filter = conv2d.sensitivity() if use_sensitivity_filter else lambda dc, x: dc

# optimizer
optimizer = OC(move=0.1, damping=0.3)

# topology
x0 = vf * np.ones((Nx * Ny, 1))
topo = Density(problem,
               x=x0, transform=rho_filter,
               obj=J_total, cons=[volume_constraint])


#%% OPTIMIZATION LOOP
loop = 0
change = 1
xnew = x0
while change > 0.01 and loop < 500:
    loop += 1
    # evaluate
    J, dJ, c, dc = topo.eval()
    dc = dc[0]
    # filter
    xold = topo.x.copy()
    dJ = sens_filter(dJ, xold)
    # update
    xnew = optimizer.update(topo, dJ, dc, vf)
    topo.update(xnew)
    vol = topo.compute_vf()
    change = np.max(np.abs(xnew - xold))
    print(f' It.:{loop:5d}, Obj.:{J:11.4f}, Vol.:{vol:7.3f}, ch.:{change:7.3f}')
    # image
    field_half = topo.x_phys.reshape(Ny, Nx, order='F')
    field_full = onp.vstack([field_half[::-1, :], field_half])
    plt.imshow(1 - field_full, cmap='gray_r', vmin=0, vmax=1)
    plt.axis('equal')
    plt.axis('off')
    plt.pause(0.01)
