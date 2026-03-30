# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`ballbox` is a Python CLI tool that simulates a 3D box of charged hard spheres. Particles interact via softened Coulomb forces, an optional uniform external E-field, and elastic hard-core collisions under periodic boundary conditions. After the simulation runs, it plots pair correlation functions (RDF) for like-charge and unlike-charge pairs, normalized by the Debye screening length.

## Stack

- Python 3.12, numpy, matplotlib
- Packaged with `pyproject.toml`

## Commands

```bash
# Install in editable mode
pip install -e .

# Run the CLI
ballbox --n-particles 50 --temperature 1.0 --radius 0.062 --box-size 3.72 --k 2.37 --field 1.0 --steps 2000
```

## Architecture

Three modules with a clean separation of concerns:

- **`ballbox/main.py`** — argparse CLI entry point, constructs `Simulation` and calls `run()`
- **`ballbox/simulation.py`** — `Simulation` class; owns 3D positions/velocities arrays and advances physics each `step()` call
- **`ballbox/visualization.py`** — `run(sim, steps)` runs the simulation loop, computes the RDF, and plots it with matplotlib

### Physics (`simulation.py`)

- Fixed timestep `dt = 0.001`
- Particles carry alternating ±1 charges (half positive, half negative)
- Initial velocities: each component drawn from `N(0, sqrt(T))` — 3D Maxwell-Boltzmann at temperature `T`
- Each `step()`:
  1. Compute softened Coulomb accelerations: `F = k·q_i·q_j / (r² + ε²)` for all pairs (O(n²)); `ε = radius`
  2. Apply uniform external field: `acc_z += q·field`
  3. Integrate: `vel += acc·dt`, `pos += vel·dt`
  4. Periodic boundary conditions: `pos %= L` (minimum image convention for forces)
  5. Hard-core collision response: elastic impulse along line of centers for overlapping pairs; overlap corrected by pushing apart by half the penetration depth

### Visualization (`visualization.py`)

- Runs `steps` simulation steps with progress reporting
- Computes the radial distribution function `g(r)` separately for like-charge (++/−−) and unlike-charge (+−) pairs
- Normalizes `r` by the Debye length `λ_D = sqrt(T / (4π·k·n))`
- Plots both curves on a single dark-themed axes; dashed line at `g = 1` marks the ideal-gas baseline
