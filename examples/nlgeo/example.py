"""

2D hyperelasticity topology optimization with geometric nonlinearity

See Wang et al., "Interpolation scheme for fictitious domain techniques and
topology optimization of finite strain elastic problems", CMAME, 2014.

Continuation schemes (Section 5.1):
  - p (SIMP penalization): 1 -> 3, Dp=0.05
    updated every 2 iterations when p < 2, every 5 when p >= 2
  - beta (density Heaviside, Eq.7): 4 -> 64, doubled every 10 iterations
    active only after p reaches max

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
from topax.opt import OC
from topax.tfp import Conv, Projection

from problem import prep_fem


#%% SETUP
vf = 0.5

xPhys2Mat = SIMP(E_max=1, E_min=1e-9, penal=3)

Nx, Ny = 120, 30
Lx, Ly = 1.0, 0.25
fwd_pred, problem = prep_fem(Nx, Ny, Lx, Ly, xPhys2Mat)


def J_total(xPhys):
    sol_list = fwd_pred(xPhys)
    compliance = problem.compute_compliance(sol_list[0])
    return compliance


def volume_constraint(xPhys):
    g = np.sum(xPhys) - vf * (xPhys.size)
    return g


# Density filter (r=a/8, a=Ly=0.25, converted to element units: 0.03125/(0.25/30)=3.75)
conv2d = Conv(problem, rmin=3.75)
density_filter = conv2d.density()

# Density Heaviside projection (Eq.7) with continuation beta
density_proj = Projection('wang', beta=4, eta=0.5)
transform = lambda x: density_proj(density_filter(x))

optimizer = OC(move=0.1, damping=0.5)

x0 = vf * np.ones((Nx * Ny, 1))
topo = Density(problem,
               x=x0, transform=transform,
               obj=J_total, cons=[volume_constraint])


#%% OPTIMIZATION LOOP with continuation
p = 1.0
beta = 4.0
heaviside_active = False
iters_since_p_upd = 0
iters_since_b_upd = 0

loop = 0
change = 1
xnew = x0
while change > 0.01 and loop < 200:
    loop += 1
    iters_since_p_upd += 1
    iters_since_b_upd += 1

    # p continuation: 1 -> 3, Dp=0.05
    if not heaviside_active:
        update_interval = 2 if p < 2.0 else 5
        if iters_since_p_upd >= update_interval and p < 3.0:
            p = min(p + 0.05, 3.0)
            iters_since_p_upd = 0
        if p >= 3.0:
            heaviside_active = True

    # beta (Heaviside) continuation: 4 -> 64, doubled every 10 iters
    if heaviside_active:
        if iters_since_b_upd >= 10 and beta < 64.0:
            beta = min(beta * 2, 64.0)
            iters_since_b_upd = 0

    xPhys2Mat.penal = p
    density_proj.set_params(beta=beta)

    J, dJ, c, dc = topo.eval()
    dc = dc[0]
    xold = topo.x.copy()
    xnew = optimizer.update(topo, dJ, dc, vf)
    topo.update(xnew)
    vol = topo.compute_vf()
    change = np.max(np.abs(xnew - xold))
    status = f'p={p:.2f}' if not heaviside_active else f'beta={beta:.0f}'
    print(f' {status:>10s}  It.:{loop:5d}, Obj.:{J:11.4e}, '
          f'Vol.:{vol:7.3f}, ch.:{change:7.3f}')
    field = onp.flip(topo.x_phys.reshape(Ny, Nx, order='F'), axis=0)
    plt.imshow(field, cmap='gray_r', vmin=0, vmax=1)
    plt.axis('equal')
    plt.axis('off')
    plt.pause(0.01)
