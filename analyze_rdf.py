"""
Compute and plot the pair radial distribution function g(r) for
same-charge vs opposite-charge particle pairs.

Usage:
    python analyze_rdf.py [--n-particles N] [--temperature T]
                          [--radius R] [--box-size L] [--k K]
                          [--warmup STEPS] [--samples N] [--stride S]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ballbox.simulation import Simulation


def compute_rdf(sim: Simulation, n_samples: int, stride: int, n_bins: int = 100):
    """
    Run the simulation for n_samples * stride steps, sampling every `stride`
    steps.  Returns (r_centers, g_same, g_opp).
    """
    L = sim.L
    r_max = L / 2.0
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    dr = bin_edges[1] - bin_edges[0]
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hist_same = np.zeros(n_bins)
    hist_opp = np.zeros(n_bins)
    n_frames = 0

    pos_idx = np.arange(sim.n)
    charges = sim.charges  # shape (n,)

    for _ in range(n_samples):
        for __ in range(stride):
            sim.step()

        # Vectorised pairwise distances with minimum-image convention
        pos = sim.pos  # (n, 3)
        # All pairs (i, j) with i < j
        i_idx, j_idx = np.triu_indices(sim.n, k=1)
        d = pos[j_idx] - pos[i_idx]              # (n_pairs, 3)
        d -= L * np.round(d / L)                  # minimum image
        dist = np.linalg.norm(d, axis=1)          # (n_pairs,)

        same_mask = charges[i_idx] == charges[j_idx]
        opp_mask = ~same_mask

        in_range = dist < r_max
        hist_same += np.histogram(dist[same_mask & in_range], bins=bin_edges)[0]
        hist_opp  += np.histogram(dist[opp_mask  & in_range], bins=bin_edges)[0]
        n_frames += 1

    # ------------------------------------------------------------------ #
    # Normalise to g(r).
    #
    # n_pos = n_neg = N/2.
    # Same-charge pairs: n_pairs_same = (N/2)*(N/2-1)/2  *2  = (N/2)*(N/2-1)
    #   (counting both ++ and -- pairs, each pair counted once)
    # Opp-charge pairs:  n_pairs_opp  = (N/2)*(N/2)
    #
    # Ideal-gas reference count in shell:
    #   <n_ideal> = n_pairs * (4π r² dr) / V
    # So  g(r) = hist / (n_frames * n_pairs * 4π r² dr / V)
    # ------------------------------------------------------------------ #
    V = L ** 3
    n_half = sim.n // 2
    n_pairs_same = n_half * (n_half - 1)   # ++ pairs + -- pairs
    n_pairs_opp  = n_half * n_half

    shell_vol = 4.0 * np.pi * r_centers ** 2 * dr

    g_same = hist_same * V / (n_frames * n_pairs_same * shell_vol)
    g_opp  = hist_opp  * V / (n_frames * n_pairs_opp  * shell_vol)

    return r_centers, g_same, g_opp


def main():
    parser = argparse.ArgumentParser(description="RDF analysis for charged particles")
    parser.add_argument("--n-particles", type=int,   default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--radius",      type=float, default=0.02)
    parser.add_argument("--box-size",    type=float, default=1.0)
    parser.add_argument("--k",           type=float, default=1.0)
    parser.add_argument("--warmup",      type=int,   default=2000,
                        help="Steps to run before sampling (default: 2000)")
    parser.add_argument("--samples",     type=int,   default=500,
                        help="Number of sampled frames (default: 500)")
    parser.add_argument("--stride",      type=int,   default=10,
                        help="Steps between samples (default: 10)")
    args = parser.parse_args()

    sim = Simulation(
        n_particles=args.n_particles,
        temperature=args.temperature,
        radius=args.radius,
        box_size=args.box_size,
        k=args.k,
    )

    print(f"Warming up for {args.warmup} steps …")
    for _ in range(args.warmup):
        sim.step()

    print(f"Sampling {args.samples} frames (stride={args.stride}) …")
    r, g_same, g_opp = compute_rdf(sim, args.samples, args.stride)

    # ---- plot --------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(r, g_same, label="Same charge  (+/+  and  −/−)", color="tab:blue")
    ax.plot(r, g_opp,  label="Opposite charge  (+/−)",        color="tab:red")
    ax.axhline(1.0, color="0.5", lw=0.8, ls="--", label="ideal gas  g(r) = 1")
    ax.set_xlabel("r  (box units)")
    ax.set_ylabel("g(r)")
    ax.set_title(
        f"Pair radial distribution function\n"
        f"N={args.n_particles}, T={args.temperature}, k={args.k}, R={args.radius}"
    )
    ax.legend()
    ax.set_xlim(0, sim.L / 2)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("rdf.png", dpi=150)
    print("Saved rdf.png")
    plt.show()


if __name__ == "__main__":
    main()
