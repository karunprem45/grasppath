"""Simple collision checking: arm links as chains of spheres, obstacles as axis-aligned boxes."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Box:
    center: np.ndarray
    half: np.ndarray  # half extents

    def distance(self, p):
        """Distance from point p to the box (0 if inside)."""
        d = np.abs(np.asarray(p) - self.center) - self.half
        return float(np.linalg.norm(np.maximum(d, 0.0)))


class CollisionChecker:
    def __init__(self, arm, obstacles, link_radius=0.06, points_per_link=4, skip_links=1):
        self.arm = arm
        self.obstacles = list(obstacles)
        self.r = link_radius
        self.k = points_per_link
        self.skip = skip_links  # the base link sits on the table, so skip it

    def link_points(self, q):
        origins = [np.zeros(3)] + [T[:3, 3] for T in self.arm.frames(q)]
        pts = []
        for i in range(self.skip, len(origins) - 1):
            a, b = origins[i], origins[i + 1]
            for t in np.linspace(0.0, 1.0, self.k):
                pts.append(a + t * (b - a))
        return np.array(pts)

    def in_collision(self, q):
        q = np.asarray(q)
        if np.any(q < self.arm.lower - 1e-9) or np.any(q > self.arm.upper + 1e-9):
            return True
        for p in self.link_points(q):
            for box in self.obstacles:
                if box.distance(p) < self.r:
                    return True
        return False

    def edge_free(self, q_a, q_b, step=0.05):
        """Check the straight joint-space segment from q_a to q_b at a fixed resolution."""
        q_a, q_b = np.asarray(q_a, float), np.asarray(q_b, float)
        n = max(2, int(np.ceil(np.max(np.abs(q_b - q_a)) / step)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            if self.in_collision(q_a + t * (q_b - q_a)):
                return False
        return True
