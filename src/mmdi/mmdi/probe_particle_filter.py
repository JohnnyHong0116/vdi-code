#!/usr/bin/env python3

"""Particle filter and rotation utilities for probe tip tracking.

Ported from the standalone probe tracker script.
"""

import numpy as np
from scipy.spatial.transform import Rotation as SciRot


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def average_rotations(rotations):
    """Weighted average of 3x3 rotation matrices (Markley quaternion method).

    Parameters
    ----------
    rotations : list of ndarray(3,3)

    Returns
    -------
    ndarray(3,3) or None if empty list.
    """
    if not rotations:
        return None
    if len(rotations) == 1:
        return rotations[0]
    quats = np.array([SciRot.from_matrix(r).as_quat() for r in rotations])  # xyzw
    n = quats.shape[0]
    w = 1.0 / n
    M = np.zeros((4, 4))
    for q in quats:
        M += w * np.outer(q, q)
    vals, vecs = np.linalg.eigh(M)
    avg_q = vecs[:, -1]
    if avg_q[3] < 0:
        avg_q = -avg_q
    return SciRot.from_quat(avg_q / np.linalg.norm(avg_q)).as_matrix()


def get_angular_distance(R1, R2):
    """Geodesic distance (degrees) between two 3x3 rotation matrices."""
    R_diff = R1.T @ R2
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angle_rad)


# ---------------------------------------------------------------------------
# Rotation stabilisation
# ---------------------------------------------------------------------------

def _rotx_deg(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=np.float64)


def _roty_deg(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=np.float64)


def _rotz_deg(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def stabilize_rotation_to_reference(R_mat, ref_R):
    """Pick the 180-degree-equivalent closest to ref_R."""
    if ref_R is None:
        return R_mat
    candidates = [
        R_mat,
        R_mat @ _rotx_deg(180.0),
        R_mat @ _roty_deg(180.0),
        R_mat @ _rotz_deg(180.0),
    ]
    best = min(candidates, key=lambda Rc: get_angular_distance(Rc, ref_R))
    return best


def enforce_display_z_up(R_c_probe, R_probe_display_fix, ref_R=None):
    """Pick a 180-equivalent so the drawn +Z axis points upward on screen."""
    candidates = [
        R_c_probe,
        R_c_probe @ _rotx_deg(180.0),
        R_c_probe @ _roty_deg(180.0),
        R_c_probe @ _rotz_deg(180.0),
    ]

    def score(Rc):
        z_y = float((Rc @ R_probe_display_fix)[1, 2])
        up_penalty = 0.0 if z_y < 0.0 else (1.0 + z_y)
        cont_penalty = 0.0 if ref_R is None else get_angular_distance(Rc, ref_R) / 180.0
        return (up_penalty, cont_penalty, abs(z_y))

    return min(candidates, key=score)


# ---------------------------------------------------------------------------
# Observation gating (rotation-aware)
# ---------------------------------------------------------------------------

DEFAULT_ROT_JUMP_DEG = 80.0


def gate_and_filter_glitches(positions, rotations, prev_estimate, prev_rot,
                             max_jump=0.40, pos_thresh=0.08,
                             rot_jump_deg=DEFAULT_ROT_JUMP_DEG,
                             rot_thresh_deg=15.0):
    """Reject outlier observations within a single frame.

    Parameters
    ----------
    positions : list of ndarray(3,)
    rotations : list of ndarray(3,3)
    prev_estimate : ndarray(3,) or None
    prev_rot : ndarray(3,3) or None

    Returns
    -------
    (ndarray(N,3), list of ndarray(3,3)) — filtered positions and rotations.
    """
    n = len(positions)
    if n == 0:
        return np.zeros((0, 3)), []

    sane_indices = []
    for i in range(n):
        p = np.asarray(positions[i], dtype=np.float64)
        if np.isfinite(p).all() and p[2] > 0.0:
            sane_indices.append(i)
    if not sane_indices:
        return np.zeros((0, 3)), []

    # Stabilize rotations vs last fused rotation
    if prev_rot is not None:
        for i in sane_indices:
            rotations[i] = stabilize_rotation_to_reference(rotations[i], prev_rot)

    if len(sane_indices) == 1:
        idx = sane_indices[0]
        p = positions[idx]
        if prev_estimate is not None:
            if np.linalg.norm(p - prev_estimate) > max_jump:
                return np.zeros((0, 3)), []
        if prev_rot is not None:
            ang = get_angular_distance(rotations[idx], prev_rot)
            if ang > rot_jump_deg:
                return np.array([p]), []
        return np.array([p]), [rotations[idx]]

    # Multiple observations: find best pivot
    if prev_rot is not None:
        best_pivot_idx = min(sane_indices,
                             key=lambda i: get_angular_distance(rotations[i], prev_rot))
    else:
        best_pivot_idx = -1
        min_total_angle = float("inf")
        for i in sane_indices:
            total_angle = sum(
                get_angular_distance(rotations[i], rotations[j])
                for j in sane_indices if i != j
            )
            if total_angle < min_total_angle:
                min_total_angle = total_angle
                best_pivot_idx = i

    ref_R = rotations[best_pivot_idx]
    subset_pos = np.array([positions[i] for i in sane_indices])
    ref_P = np.median(subset_pos, axis=0)

    clean_pos = []
    clean_rot = []
    for i in sane_indices:
        p = positions[i]
        R_mat = rotations[i]
        if prev_estimate is not None and np.linalg.norm(p - prev_estimate) > max_jump:
            continue
        if prev_rot is not None and get_angular_distance(R_mat, prev_rot) > (rot_jump_deg * 1.5):
            continue
        if np.linalg.norm(p - ref_P) > pos_thresh:
            continue
        if get_angular_distance(R_mat, ref_R) > rot_thresh_deg:
            continue
        clean_pos.append(p)
        clean_rot.append(R_mat)

    if not clean_pos:
        return np.zeros((0, 3)), []
    return np.array(clean_pos), clean_rot


# ---------------------------------------------------------------------------
# Particle filter (position only — rotation handled externally)
# ---------------------------------------------------------------------------

class ProbeParticleFilter:
    """3-D position particle filter."""

    def __init__(self, n_particles=3000, process_noise_std=0.002,
                 meas_noise_std=0.015, init_spread=0.05):
        self.n = n_particles
        self.process_noise_std = process_noise_std
        self.meas_noise_std = meas_noise_std
        self.init_spread = init_spread
        self.particles = None   # (n, 3)
        self.weights = np.ones(n_particles) / n_particles
        self.initialized = False

    def init_from_measurements(self, positions):
        """Scatter particles around the mean of the given positions (N,3)."""
        mean_pos = np.mean(positions, axis=0)
        self.particles = mean_pos + np.random.randn(self.n, 3) * self.init_spread
        self.weights = np.ones(self.n) / self.n
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return
        self.particles += np.random.randn(self.n, 3) * self.process_noise_std

    def update(self, positions):
        """Update weights given positions array (N,3)."""
        if not self.initialized or len(positions) == 0:
            return
        log_w = np.zeros(self.n)
        for pos in positions:
            diff = self.particles - pos
            dist_sq = np.sum(diff ** 2, axis=1)
            log_w -= dist_sq / (2.0 * self.meas_noise_std ** 2)
        log_w -= log_w.max()
        self.weights = np.exp(log_w)
        self.weights /= self.weights.sum() + 1e-300
        # Systematic resampling
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.n * 0.5:
            cumsum = np.cumsum(self.weights)
            cumsum[-1] = 1.0
            u0 = np.random.random() / self.n
            u = u0 + np.arange(self.n) / self.n
            indices = np.searchsorted(cumsum, u)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n) / self.n

    def estimate_position(self):
        """Return weighted mean position (3,) or None."""
        if not self.initialized:
            return None
        return np.average(self.particles, axis=0, weights=self.weights)
