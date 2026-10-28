# ballbox

An exploratory Python simulation of charged hard spheres in a periodic 3D box. Particles interact through softened Coulomb forces, an optional electric field, and elastic collisions. The program plots pair correlation functions from the final configuration.

## Installation

**Python 3.12+ required.** Optionally manage your Python version with [pyenv](https://github.com/pyenv/pyenv):

```bash
pyenv install 3.12
pyenv local 3.12
```

Install the package in editable mode:

```bash
pip install -e .
```

## Usage

```bash
ballbox [--n-particles N] [--temperature T] [--radius R] [--box-size L] \
        [--k K] [--field F] [--steps S]
```

**Example:**

```bash
ballbox --n-particles 50 --temperature 1.0 --radius 0.062 --box-size 3.72 --steps 2000
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--n-particles` | `50` | Number of particles (half positive, half negative) |
| `--temperature` | `1.0` | Initial temperature; scales the Maxwell-Boltzmann speed distribution |
| `--radius` | `0.062` | Hard-sphere radius |
| `--box-size` | `3.72` | Side length of the cubic box |
| `--k` | `2.37` | Electrostatic coupling constant |
| `--field` | `1.0` | Uniform external E-field strength in the z direction (set to `0` to disable) |
| `--steps` | `2000` | Number of simulation steps to run before plotting |

## Physics

**Initial conditions.** Each velocity component is drawn independently from a Gaussian with variance `T`, equivalent to a 3D Maxwell-Boltzmann distribution at temperature `T`. Positions are placed without overlaps (random placement with a grid fallback).

**Electrostatic interactions.** Particles carry +1 or −1 charges. At each timestep, every pair feels a softened Coulomb force `F = k·q_i·q_j / (r² + ε²)`, where `ε = radius` prevents the singularity at contact. Like charges repel; unlike charges attract.

**External field.** A uniform field accelerates charges in the z direction each step.

**Hard-core collisions.** When two particles overlap, their velocities are updated with an elastic impulse along the line of centers (only if approaching), and the overlap is corrected by pushing each particle back by half the penetration depth.

**Boundary conditions.** Periodic: particles that exit one face re-enter from the opposite face (minimum image convention for force calculations).

**Output.** The final configuration gives a radial distribution function `g(r)` for like-charge pairs (blue) and unlike-charge pairs (red). Distance is scaled by `λ_D = sqrt(T / (4π·k·n))`, using the initial temperature.

**Scope.** Temperature sets the initial velocities; it is not maintained during the run. The default field drives the system, so this example does not establish equilibrium Debye screening.
