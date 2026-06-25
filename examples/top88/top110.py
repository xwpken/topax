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
from topax.transforms import ConvolutionFilter, Heaviside
from topax.opt import Optimizer

from mbb_beam import prep_fem


#%% SETUP
vf = 0.5 
rmin = 1.8
use_density_filter = True
use_sensitivity_filter = False

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

# projection
beta = 1.0
heaviside = Heaviside(method='Guest2004',options={'beta':beta})

# optimizer
OC = Optimizer(method='OC', options={'move':0.2, 'damping':0.5})

# topology 
x0 = vf * np.ones((Nx*Ny, 1))
topology = Density(problem, 
                   x=x0, transform=rho_filter+heaviside,
                   obj=J_total, cons=[volume_constraints,])

#%% OPTIMIZATION LOOP
loop_beta = 0
loop = 0
change = 1
xnew, xTilde, xPhys = topology.get_intermediate_values()
while change>0.01 and loop<500:
    loop_beta += 1
    loop += 1
    # Be careful!
    if loop_beta == 1 and beta > 1:
        # just to be consistent with top110
        # however, this may need further check
        J, dJ = jax.value_and_grad(J_total)(xPhys)
        c, dc = jax.value_and_grad(volume_constraints)(xPhys)
        dx = heaviside.grad(xTilde)
        dJ = conv2d.H @ ((dJ*dx)/conv2d.Hs)
        dc = conv2d.H @ ((dc*dx)/conv2d.Hs)
    else:
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
    # update projection
    if beta < 512 and (loop_beta>=50 or change <=0.01):
        xnew, xTilde, xPhys = topology.get_intermediate_values()
        beta = 2 * beta
        heaviside.set_params(beta=beta)
        loop_beta = 0
        change = 1
        print(f' Parameter beta increased to {beta}')
    