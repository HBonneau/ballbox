import numpy as np
import matplotlib.pyplot as plt

from ballbox.simulation import Simulation


def _compute_rdf(pos, charges, L, n_bins=100):
    r_max = L / 2
    dr = r_max / n_bins
    bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])

    counts_like = np.zeros(n_bins)
    counts_unlike = np.zeros(n_bins)

    n = len(pos)
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = pos[j] - pos[i]
            d -= L * np.round(d / L)
            dist = np.sqrt(np.dot(d, d))
            if dist >= r_max:
                continue
            idx = int(dist / dr)
            if idx >= n_bins:
                continue
            if charges[i] * charges[j] > 0:
                counts_like[idx] += 1
            else:
                counts_unlike[idx] += 1

    V = L ** 3
    shell_vol = 4 * np.pi * r_centers ** 2 * dr
    n_pos = int(np.sum(charges > 0))
    n_neg = int(np.sum(charges < 0))
    n_like_pairs = (n_pos * (n_pos - 1) + n_neg * (n_neg - 1)) // 2
    n_unlike_pairs = n_pos * n_neg

    g_like = V * counts_like / (n_like_pairs * shell_vol)
    g_unlike = V * counts_unlike / (n_unlike_pairs * shell_vol)

    return r_centers, g_like, g_unlike


def run(sim: Simulation, steps: int = 2000):
    print(f"Running {steps} steps...", flush=True)
    report_every = max(steps // 10, 1)
    for i in range(steps):
        sim.step()
        if (i + 1) % report_every == 0:
            print(f"  {100 * (i + 1) // steps}%", flush=True)

    lD = sim.debye_length
    r_centers, g_like, g_unlike = _compute_rdf(sim.pos, sim.charges, sim.L)
    r_norm = r_centers / lD

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_title("Pair Correlation Functions", color="white", fontsize=13)
    ax.set_xlabel("r / λ_D", color="#aaaacc")
    ax.set_ylabel("g(r)", color="#aaaacc")
    ax.tick_params(colors="#aaaacc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#0f3460")

    ax.axhline(1.0, color="#555577", linewidth=1, linestyle="--", label="ideal gas")
    ax.plot(r_norm, g_like, color="#4a90d9", linewidth=2, label="like charges (++ / −−)")
    ax.plot(r_norm, g_unlike, color="#e74c3c", linewidth=2, label="unlike charges (+−)")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)
    ax.set_xlim(0, r_norm[-1])

    plt.tight_layout()
    plt.show()
