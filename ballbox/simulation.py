import numpy as np


class Simulation:
    def __init__(self, n_particles: int, temperature: float, radius: float, box_size: float, k: float = 0.04, field: float = 1.0):
        self.n = n_particles
        self.T = temperature
        self.r = radius
        self.L = box_size
        self.k = k          # electrostatic coupling constant
        self.field = field  # uniform external E field in the z direction
        self.dt = 0.001

        # Assign alternating +1/-1 charges (half positive, half negative)
        self.charges = np.ones(self.n)
        self.charges[self.n // 2:] = -1

        # Initialize positions without overlaps
        self.pos = self._init_positions()

        # Maxwell-Boltzmann in 3D: each velocity component ~ N(0, sqrt(T))
        self.vel = np.random.normal(0, np.sqrt(self.T), (self.n, 3))

    def _init_positions(self) -> np.ndarray:
        pos = np.empty((self.n, 3))
        placed = 0
        max_attempts = self.n * 1000
        attempts = 0
        while placed < self.n and attempts < max_attempts:
            attempts += 1
            candidate = np.random.uniform(0, self.L, 3)
            if placed == 0:
                pos[0] = candidate
                placed = 1
                continue
            diffs = pos[:placed] - candidate
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            if np.all(dists >= 2 * self.r):
                pos[placed] = candidate
                placed += 1
        if placed < self.n:
            # Fall back: place remaining on a 3D grid
            grid_n = int(np.ceil(self.n ** (1 / 3))) + 1
            spacing = self.L / max(grid_n, 1)
            idx = 0
            for i in range(grid_n):
                for j in range(grid_n):
                    for k in range(grid_n):
                        if idx >= self.n:
                            break
                        pos[idx] = [i * spacing, j * spacing, k * spacing]
                        idx += 1
                    if idx >= self.n:
                        break
                if idx >= self.n:
                    break
        return pos

    def step(self):
        # Electrostatic force: F = k * q_i * q_j / (r^2 + eps^2), softened to avoid singularity
        # Like charges (same sign) repel; opposite charges attract
        eps2 = self.r ** 2
        acc = np.zeros_like(self.pos)
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                d = self.pos[j] - self.pos[i]
                d -= self.L * np.round(d / self.L)  # minimum image convention
                dist2 = np.dot(d, d)
                if dist2 < 1e-24:
                    continue
                dist = np.sqrt(dist2)
                # Force sign: positive = repulsive (pushes j away from i), negative = attractive
                f_mag = self.k * self.charges[i] * self.charges[j] / (dist2 + eps2)
                f = f_mag * (d / dist)
                acc[i] -= f
                acc[j] += f

        # External uniform field: F = q * E in the z direction
        if self.field != 0.0:
            acc[:, 2] += self.charges * self.field

        # Update velocities then positions
        self.vel += acc * self.dt
        self.pos += self.vel * self.dt

        # Periodic boundary conditions: wrap positions into [0, L)
        self.pos %= self.L

        # Hard-core collisions: elastic response when particles overlap
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                d = self.pos[j] - self.pos[i]
                d -= self.L * np.round(d / self.L)  # minimum image convention
                dist2 = np.dot(d, d)
                if dist2 >= (2 * self.r) ** 2 or dist2 < 1e-24:
                    continue
                dist = np.sqrt(dist2)
                d_hat = d / dist
                # Only resolve if particles are approaching
                dv = np.dot(self.vel[i] - self.vel[j], d_hat)
                if dv > 0:
                    self.vel[i] -= dv * d_hat
                    self.vel[j] += dv * d_hat
                # Push apart to remove overlap
                overlap = 2 * self.r - dist
                self.pos[i] -= 0.5 * overlap * d_hat
                self.pos[j] += 0.5 * overlap * d_hat
        self.pos %= self.L

    @property
    def debye_length(self) -> float:
        n = self.n / self.L ** 3
        return np.sqrt(self.T / (4 * np.pi * self.k * n))

    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.vel, axis=1)
