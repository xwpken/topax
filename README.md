# topax
![Github Star](https://img.shields.io/github/stars/xwpken/topax) ![Github Fork](https://img.shields.io/github/forks/xwpken/topax) ![License](https://img.shields.io/github/license/xwpken/topax.svg) ![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)

Topology optimization with automatic differentiaition in JAX

## Features

* Differentiable finite element analysis powered by [`jax-fem`](https://github.com/deepmodeling/jax-fem)
* Built-in classical topology optimization methods (e.g., SIMP, LSM)
* Hands-free adjoint sensitivity analysis via automatic differentiation

## Installation
Since `topax` is built on top of [`jax-fem`](https://github.com/deepmodeling/jax-fem), please follow the [instructions](https://github.com/deepmodeling/jax-fem?tab=readme-ov-file#installation) to install [`jax-fem`](https://github.com/deepmodeling/jax-fem) first. Then activate the built conda environment and clone this repository:

```bash
git clone https://github.com/xwpken/topax.git
cd topax
```

then install the package locally:

```bash
pip install -e .
```
For the support of [MMA](https://en.wikipedia.org/wiki/Method_of_moving_asymptotes) optimizer, please install the [`mmapy`](https://github.com/arjendeetman/GCMMA-MMA-Python) package via:

```bash
pip install mmapy
```

## Quick Start

Run the following command:

```bash
python -m examples.top88.example.py
```
to reproduce the 2D cantilever beam topology optimization example in the well-known [educational paper](https://link.springer.com/article/10.1007/s00158-010-0594-7) by Erik Andreassen et al.

More examples can be found in the [examples](./examples) folder.

## Roadmap
- [ ] Support multi-material design
- [ ] Support BESO method
- [ ] More advanced examples (e.g., design-dependent loads)
- [ ] To be added...

## Citation
If you find `topax` useful in your research, please consider citing this repository:
```bibtex
@software{topaxgithub,
  author = {Weipeng Xu and Tianju Xue},
  title = {{topax}: Topology optimization with automatic differentiaition in JAX},
  url = {http://github.com/xwpken/topax},
  year = {2025},
}
```