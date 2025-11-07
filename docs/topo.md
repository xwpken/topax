# Notes on topology optimization

> Last modified: 26/26/2024
> 
> Editor: Weipeng XU
> 

## Basic concepts

### Formulations

Without loss of generaity, the continuous topology optimization problem can be written as:

$$
\begin{aligned}
    min:\quad&J(\boldsymbol{u}(\boldsymbol{x}),\boldsymbol{x})=\sum_i\int_{\Omega_i}f(\boldsymbol{u}(x_i),x_i)dV\\
    s.t:\quad&G_0(\boldsymbol{x})=\sum_{i}v_ix_i-V_0\leq0\\
    &G_j(\boldsymbol{u}(\boldsymbol{x}),\boldsymbol{x})\leq0,\quad j=1,...,M\\
    & 0\leq x_i\leq1,\quad i=1,...,N
\end{aligned}
$$
 where $J$ represents the objective function determined by the design variable $\boldsymbol{x}$ and the state field $\boldsymbol{u}$. $\boldsymbol{x}$ can continuously change between 0 and 1, which allows for the use of gradient-based optimization algorithms. $G_0$ and $G_i$ denote the common material volume constraints and other constraints, respectively.


### Material interpolation model

The formulation above shows that the objective function $F$ is calculated as the local integral over a local function $f(\boldsymbol{u}(x),x))$. In most cases, $f$ can be written as:

$$
f(\boldsymbol{u}(x),x)=g(x)f_0(\boldsymbol{u})
$$

where $f_0$ is the function of the field of the solid material. $g(x)$ denote the density interpolation function The commonly used $g(x)$ will be explained below.

#### Modified SIMP model

The Young's modulus of each element  is determined by the cell-level design variable $x_e$:

$$
E_e(x_e) = E_{min}+x_e^p(E_0-E_{min})
$$

where $E_0$ is the origin Young's modulus of the material. $E_{min}$ is a small number (ususally $10^{-9}$) assigned to void regions to avoid the singular stiffness matrix. The penalization factor $p$ (usually $3$) is used to move towards black-and-white solution.

### Optimization procedure

~

## Numerical techniques

### Density/sensitivity filters

#### Convolution-based

This kind of filter modifies the sensitivity ${\partial c}/{\partial x_i}$ as follows:

$$
\frac{\partial\tilde{c}}{\partial x_e}=\frac{1}{max(\gamma,x_e)\sum_{i\in N_e}H_{ei}}\sum_{i\in N_e}H_{ei}x_i\frac{\partial c}{\partial x_i}
$$

where $N_e$ is the set of element $i$ which has the "center-to-center" distance $d(e,i)$ less than filter radius $r_{min}$. $H_{ei}$ is a weight factor defined as:

$$
H_{ei} = max(0,r_{min}-d(e,i))
$$

where $\gamma=10^{-3}$ is a small positive number to avoid division by zero.

This filter can also be used as a density filter:

$$
\tilde{x}_e=\frac{\sum_{i\in N_e}H_{ei}x_i}{\sum_{i\in N_e}H_{ei}}
$$

where $x_e$ and $\tilde{x}_e$ represent the design density variables and physical density variables, respectively. $\tilde{x}_e$ should always be provided as the optimal design results.

With the chain rule, the sensitivity $\partial J/\partial x_e$ can be expressed as:

$$
\frac{\partial J}{\partial x_j}=\sum_{e\in N_j}\frac{\partial J}{\partial\tilde{x}_e}\frac{\partial \tilde{x}_e}{x_j}=\sum_{e\in N_j}\frac{1}{\sum_{i\in N_e}H_{ei}}H_{ej}\frac{\partial J}{\partial \tilde{x}_e}
$$

The derivation of $\partial G/\partial x_e$ are similar.


#### Helmholtz PDE-based

The above convolution filter can also be implicitly represented by the solution of a Helmholtz type PDE. This type of filter are suitable for large-scale optimization problmes, which can save the memory usage and computational cost.

The governing equation can be written as:

$$
-R^2_{min}\nabla^2\tilde{\psi}+\tilde{\psi}=\psi
$$

where $\tilde{\psi}$ and $\psi$ represent the filtered and unfiltered design field, respectively. The relationship between $R_{min}$ and $r_{min}$ in convolution filter is:

$$
R_{min}=r_{min}/2\sqrt{3}
$$

The weak form gives

$$\int_{\Omega}(R_{min}^2\nabla\tilde{\psi}\cdot\nabla v+\tilde{\psi}v)dV=\int_{\Omega}\psi vdV$$

which can be implemented directly in JAX-FEM.




#### Morphology-based

~


### Density projection

#### Heaviside projection by Guest et al.

The aim of Heaviside projection filter is $(1)$ to achieve a minimum length scale in optimization design and $(2)$ to obtain black-and-white solutions.

The physical variable $\bar{x}_e$ is expressed as an approximate Heaviside function:

$$
\bar{x}_e=1-e^{-\beta\tilde{x}_e}+\tilde{x}_ee^{-\beta}
$$

where $\tilde{x}_e$ is the intermediate density variable. $\beta$ is the parameter to control the smoothness of the approximation ($\beta\rightarrow 0$, no effect; $\beta\rightarrow inf$,true Heaviside function). $\beta$ is gradually increased during the optimization to avoid a local minima and ensure differentiability.

The sensitivity of the objective/constraint function with respect to the intermediate densities can be obtained by:

$$
\frac{\partial J}{\partial \tilde{x}_e}=\frac{\partial J}{\partial \bar{x}_e}\frac{\partial \bar{x}_e}{\partial\tilde{x}_e}
$$

and

$$
\frac{\partial \bar{x}_e}{\partial\tilde{x}_e}=\beta e^{-\beta\tilde{x}_e}+e^{-\beta}
$$


#### Heaviside projection ($tanh$) by Wang et al.

~



## Optimization problems

### Minimum compliance problem

### Heat conduction

### Compliant mechanism

### Stress constraints

### Minimum volume problem

### Metamaterial





## Optimizer

### Method of Moving Asymptotes (MMA)




### optimality criteria method (OC)









> Reference: 
> 
> 1. Andreassen, Erik, et al. "Efficient topology optimization in MATLAB using 88 lines of code." Structural and Multidisciplinary Optimization 43 (2011): 1-16.
> 
> 2. https://topopt.readthedocs.io/en/documentation/TopOpt.html
> 