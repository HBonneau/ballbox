import argparse

from ballbox.simulation import Simulation
from ballbox.visualization import run


def main():
    parser = argparse.ArgumentParser(
        description="Simulate elastic hard-sphere collisions in a 3D box."
    )
    parser.add_argument("--n-particles", type=int, default=50, metavar="N",
                        help="Number of particles (default: 50)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Initial temperature — scales particle speeds (default: 1.0)")
    parser.add_argument("--radius", type=float, default=0.062,
                        help="Hard-sphere radius (default: 0.062)")
    parser.add_argument("--box-size", type=float, default=3.72,
                        help="Side length of the square box (default: 3.72)")
    parser.add_argument("--k", type=float, default=2.37,
                        help="Electrostatic coupling constant (default: 2.37)")
    parser.add_argument("--field", type=float, default=1.0,
                        help="Uniform external E field in the z direction (default: 1.0)")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of simulation steps to run before plotting (default: 2000)")
    args = parser.parse_args()

    sim = Simulation(
        n_particles=args.n_particles,
        temperature=args.temperature,
        radius=args.radius,
        box_size=args.box_size,
        k=args.k,
        field=args.field,
    )
    run(sim, steps=args.steps)


if __name__ == "__main__":
    main()
