"""
Compute and plot the 2D pair correlation function g(r_∥, r_⊥) in the presence
of a uniform external electric field (z direction).  Produces two heatmaps:
  - same-charge pairs   (+/+ and −/−)
  - opposite-charge pairs  (+/−)

Usage:
    python analyze_2d_rdf.py [--n-particles N] [--temperature T]
                              [--radius R] [--box-size L] [--k K]
                              [--field E] [--warmup W] [--samples S] [--stride D]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ballbox.simulation import Simulation


def compute_2d_rdf(sim: Simulation, n_samples: int, stride: int, n_bins: int = 60):
    """
    Returns (par_centers, perp_centers, g_same, g_opp).

    The field is along z.  We bin pair displacements by:
      r_∥  = Δz  (signed, along field direction)          → x-axis of heatmap
      r_⊥  = Δx or Δy  (signed, one perpendicular Cartesian component at a time;
                         both x and y are accumulated for 2× statistics)  → y-axis

    Normalisation uses a flat 2-D Cartesian volume element dr_∥ dr_⊥, integrating
    out the third coordinate analytically (factor L), so g → 1 for an ideal gas
    with no singularity near r_⊥ = 0.
    """
    L = sim.L
    r_max = L / 2.0

    par_edges  = np.linspace(-r_max, r_max, n_bins + 1)
    perp_edges = np.linspace(-r_max, r_max, n_bins + 1)

    d_par  = par_edges[1] - par_edges[0]
    d_perp = perp_edges[1] - perp_edges[0]

    par_centers  = 0.5 * (par_edges[:-1] + par_edges[1:])
    perp_centers = 0.5 * (perp_edges[:-1] + perp_edges[1:])

    hist_same = np.zeros((n_bins, n_bins))
    hist_opp  = np.zeros((n_bins, n_bins))
    n_frames  = 0

    charges = sim.charges

    for _ in range(n_samples):
        for __ in range(stride):
            sim.step()

        pos = sim.pos
        i_idx, j_idx = np.triu_indices(sim.n, k=1)
        d = pos[j_idx] - pos[i_idx]
        d -= L * np.round(d / L)           # minimum image

        r_par = d[:, 2]   # Δz, signed, along field

        same_mask = charges[i_idx] == charges[j_idx]
        opp_mask  = ~same_mask

        # Accumulate both perpendicular Cartesian components (Δx and Δy) for 2× statistics.
        # By azimuthal symmetry around z, both give samples of the same underlying distribution.
        for perp_col in (0, 1):
            r_perp = d[:, perp_col]   # Δx or Δy, signed
            in_range = (np.abs(r_par) < r_max) & (np.abs(r_perp) < r_max)

            h_s, _, _ = np.histogram2d(
                r_par[same_mask & in_range], r_perp[same_mask & in_range],
                bins=[par_edges, perp_edges],
            )
            h_o, _, _ = np.histogram2d(
                r_par[opp_mask  & in_range], r_perp[opp_mask  & in_range],
                bins=[par_edges, perp_edges],
            )
            hist_same += h_s
            hist_opp  += h_o

        n_frames += 1

    # ------------------------------------------------------------------
    # Normalise to g(r_∥, r_⊥).
    # Cartesian 2-D bin in (Δz, Δx): integrating out Δy over range L,
    # expected count for ideal gas = n_pairs * d_par * d_perp * L / V
    #                              = n_pairs * d_par * d_perp / L²
    # Factor of 2 because we accumulated both Δx and Δy.
    # g = hist / (2 * n_frames * n_pairs * d_par * d_perp / L²)
    # ------------------------------------------------------------------
    n_half = sim.n // 2
    n_pairs_same = n_half * (n_half - 1)   # both ++ and -- pairs
    n_pairs_opp  = n_half * n_half

    norm_same = 2 * n_frames * n_pairs_same * d_par * d_perp / L**2
    norm_opp  = 2 * n_frames * n_pairs_opp  * d_par * d_perp / L**2

    g_same = hist_same / norm_same
    g_opp  = hist_opp  / norm_opp

    return par_centers, perp_centers, g_same, g_opp


def plot_heatmap(ax, par, perp, g, title, vmax=None):
    # pcolormesh expects (y, x) indexing → transpose g
    # par is x-axis (r_∥), perp is y-axis (r_⊥)
    g_plot = np.where(np.isfinite(g), g, 0.0)
    if vmax is None:
        vmax = np.nanpercentile(g_plot, 98)

    mesh = ax.pcolormesh(
        par, perp, g_plot.T,
        cmap="inferno", vmin=0, vmax=vmax, shading="auto",
    )
    ax.set_xlabel(r"$r_\parallel$  (along $\mathbf{E}$)")
    ax.set_ylabel(r"$r_\perp$  (perpendicular to $\mathbf{E}$)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axhline(0, color="white", lw=0.4, ls="--")
    ax.axvline(0, color="white", lw=0.4, ls="--")
    plt.colorbar(mesh, ax=ax, label="g(r_∥, r_⊥)")


def main():
    parser = argparse.ArgumentParser(description="2D RDF analysis with external field")
    parser.add_argument("--n-particles", type=int,   default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--radius",      type=float, default=0.02)
    parser.add_argument("--box-size",    type=float, default=1.0)
    parser.add_argument("--k",           type=float, default=1.0)
    parser.add_argument("--field",       type=float, default=5.0,
                        help="External E field magnitude in z direction (default: 5.0)")
    parser.add_argument("--warmup",      type=int,   default=2000)
    parser.add_argument("--samples",     type=int,   default=500)
    parser.add_argument("--stride",      type=int,   default=10)
    args = parser.parse_args()

    sim = Simulation(
        n_particles=args.n_particles,
        temperature=args.temperature,
        radius=args.radius,
        box_size=args.box_size,
        k=args.k,
        field=args.field,
    )

    print(f"Warming up for {args.warmup} steps  (field={args.field}) …")
    for _ in range(args.warmup):
        sim.step()

    print(f"Sampling {args.samples} frames (stride={args.stride}) …")
    par, perp, g_same, g_opp = compute_2d_rdf(sim, args.samples, args.stride)

    # Use a shared colour scale so the two panels are directly comparable
    vmax = max(
        np.nanpercentile(g_same, 98),
        np.nanpercentile(g_opp,  98),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"2D pair correlation  g(r_∥, r_⊥)\n"
        f"N={args.n_particles}, T={args.temperature}, k={args.k}, "
        f"E={args.field}, R={args.radius}",
    )

    plot_heatmap(axes[0], par, perp, g_same,
                 "Same charge  (+/+  and  −/−)", vmax=vmax)
    plot_heatmap(axes[1], par, perp, g_opp,
                 "Opposite charge  (+/−)", vmax=vmax)

    plt.tight_layout()
    plt.savefig("rdf_2d.png", dpi=150)
    print("Saved rdf_2d.png")
    plt.show()


if __name__ == "__main__":
    main()
