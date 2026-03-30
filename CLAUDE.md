# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`ballbox` is a Python CLI tool that simulates elastic hard-sphere collisions in a 2D box, visualized with matplotlib.

## Stack

- Python 3.12, numpy, matplotlib
- Packaged with `pyproject.toml`

## Commands

```bash
# Install in editable mode
pip install -e .

# Run the CLI
ballbox --n-particles 50 --temperature 1.0 --radius 0.02 --box-size 1.0
```

## Architecture

Three modules with a clean separation of concerns:

- **`ballbox/main.py`** — argparse CLI entry point, constructs `Simulation` and calls `run()`
- **`ballbox/simulation.py`** — `Simulation` class; owns positions/velocities arrays and advances physics each `step()` call
- **`ballbox/visualization.py`** — `run(sim)` sets up a two-panel matplotlib figure and drives the loop via `FuncAnimation`; calls `sim.step()` N times per frame

### Physics (`simulation.py`)

- Fixed timestep `dt = 0.001`
- Initial speeds sampled from a Rayleigh distribution (`σ = sqrt(T)`), which is equivalent to the 2D Maxwell-Boltzmann
- Each `step()`: move → wall reflect → pairwise collision resolve (O(n²))
- Collision response: elastic exchange of velocity along the line of centers; overlap corrected by pushing particles apart by half the overlap each

### Visualization (`visualization.py`)

- Left panel: particles as `matplotlib.patches.Circle` objects
- Right panel: live speed histogram (density) with the analytical 2D Maxwell-Boltzmann curve `P(v) = (1/T)·v·exp(−v²/2T)` overlaid
- `steps_per_frame = 5` to advance the simulation faster than the render rate
