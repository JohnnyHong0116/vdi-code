#!/usr/bin/env python3

"""Particle filter and rotation utilities for probe tip tracking.

The tracker needs to stay smooth without adding noticeable lag, so the filter
uses a constant-velocity state and adaptive temporal blending rather than a
position-only random walk.
"""

import numpy as np
from scipy.spatial.transform import Rotation as SciRot


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def average_rotations(rotations, weights=None):
    """Weighted average of 3x3 rotation matrices via SVD projection.

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
    if weights is None:
        weights = np.ones(len(rotations), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 1e-9, None)
    weights /= weights.sum()
    M = np.zeros((3, 3), dtype=np.float64)
    for R_mat, w in zip(rotations, weights):
        M += w * np.asarray(R_mat, dtype=np.float64)
    U, _, Vt = np.linalg.svd(M)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1.0
        R_mean = U @ Vt
    return R_mean


def get_angular_distance(R1, R2):
    """Geodesic distance (degrees) between two 3x3 rotation matrices."""
    R_diff = R1.T @ R2
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angle_rad)


def adaptive_blend_alpha(delta, min_alpha, max_alpha, response_scale):
    """Blend small jitter heavily and large motion lightly."""
    if response_scale <= 1e-9:
        return float(max_alpha)
    ramp = np.clip(float(delta) / float(response_scale), 0.0, 1.0)
    return float(min_alpha + (max_alpha - min_alpha) * ramp)


def blend_positions(prev_pos, next_pos, alpha):
    if prev_pos is None:
        return np.asarray(next_pos, dtype=np.float64)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    prev_pos = np.asarray(prev_pos, dtype=np.float64)
    next_pos = np.asarray(next_pos, dtype=np.float64)
    return ((1.0 - alpha) * prev_pos) + (alpha * next_pos)


def blend_rotations(prev_R, next_R, alpha):
    """Slerp-like interpolation between rotation matrices."""
    if prev_R is None or alpha >= 1.0:
        return next_R
    alpha = float(np.clip(alpha, 0.0, 1.0))
    R_prev = SciRot.from_matrix(prev_R)
    R_next = SciRot.from_matrix(next_R)
    delta = R_next * R_prev.inv()
    step = SciRot.from_rotvec(delta.as_rotvec() * alpha)
    return (step * R_prev).as_matrix()


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
    """Keep a single physical frame; do not inject 180-degree alternatives."""
    return R_mat


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
                             max_jump=0.15, pos_thresh=0.03,
                             rot_jump_deg=DEFAULT_ROT_JUMP_DEG,
                             rot_thresh_deg=10.0, weights=None):
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
        return np.zeros((0, 3)), [], np.zeros((0,), dtype=np.float64)

    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    sane_indices = []
    for i in range(n):
        p = np.asarray(positions[i], dtype=np.float64)
        if np.isfinite(p).all() and p[2] > 0.0:
            sane_indices.append(i)
    if not sane_indices:
        return np.zeros((0, 3)), [], np.zeros((0,), dtype=np.float64)

    # Stabilize rotations vs last fused rotation
    if prev_rot is not None:
        for i in sane_indices:
            rotations[i] = stabilize_rotation_to_reference(rotations[i], prev_rot)

    if len(sane_indices) == 1:
        idx = sane_indices[0]
        p = positions[idx]
        if prev_estimate is not None:
            if np.linalg.norm(p - prev_estimate) > max_jump:
                return np.zeros((0, 3)), [], np.zeros((0,), dtype=np.float64)
        # A single visible tag after a fast wrist rotation is still better than
        # freezing the state on an outdated orientation.
        return np.array([p]), [rotations[idx]], np.array([weights[idx]], dtype=np.float64)

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
    clean_weights = []
    for i in sane_indices:
        p = positions[i]
        R_mat = rotations[i]
        if prev_estimate is not None and np.linalg.norm(p - prev_estimate) > max_jump:
            continue
        if np.linalg.norm(p - ref_P) > pos_thresh:
            continue
        if get_angular_distance(R_mat, ref_R) > rot_thresh_deg:
            continue
        clean_pos.append(p)
        clean_rot.append(R_mat)
        clean_weights.append(weights[i])

    if not clean_pos:
        return np.zeros((0, 3)), [], np.zeros((0,), dtype=np.float64)

    clean_weights = np.asarray(clean_weights, dtype=np.float64)
    clean_weights = np.clip(clean_weights, 1e-9, None)
    clean_weights /= clean_weights.sum()
    return np.array(clean_pos), clean_rot, clean_weights


# ---------------------------------------------------------------------------
# Particle filter (position only — rotation handled externally)
# ---------------------------------------------------------------------------

class ProbeParticleFilter:
    """3-D constant-velocity particle filter."""

    def __init__(self, n_particles=3000, process_noise_std=0.002,
                 meas_noise_std=0.015, init_spread=0.02,
                 velocity_noise_std=None, velocity_decay=0.92):
        self.n = n_particles
        self.process_noise_std = process_noise_std
        self.meas_noise_std = meas_noise_std
        self.init_spread = init_spread
        self.velocity_noise_std = (
            velocity_noise_std
            if velocity_noise_std is not None
            else max(0.001, process_noise_std * 6.0)
        )
        self.velocity_decay = float(np.clip(velocity_decay, 0.0, 1.0))
        self.particles = None   # (n, 3)
        self.velocities = None  # (n, 3)
        self.weights = np.ones(n_particles) / n_particles
        self.initialized = False

    def init_from_measurements(self, positions, initial_velocity=None):
        """Scatter particles around the mean of the given positions (N,3)."""
        mean_pos = np.mean(positions, axis=0)
        self.particles = mean_pos + np.random.randn(self.n, 3) * self.init_spread
        if initial_velocity is None:
            initial_velocity = np.zeros(3, dtype=np.float64)
        initial_velocity = np.asarray(initial_velocity, dtype=np.float64)
        self.velocities = initial_velocity + (
            np.random.randn(self.n, 3) * self.velocity_noise_std * 0.25
        )
        self.weights = np.ones(self.n) / self.n
        self.initialized = True

    def predict(self, dt):
        if not self.initialized:
            return
        dt = float(np.clip(dt, 1e-3, 0.10))
        frame_scale = np.sqrt(dt * 60.0)
        pos_noise = self.process_noise_std * frame_scale
        vel_noise = self.velocity_noise_std * frame_scale
        self.velocities = (
            self.velocities * self.velocity_decay
            + (np.random.randn(self.n, 3) * vel_noise)
        )
        self.particles += (self.velocities * dt) + (
            np.random.randn(self.n, 3) * pos_noise
        )

    def update(self, positions, measurement_weights=None):
        """Update weights given positions array (N,3)."""
        if not self.initialized or len(positions) == 0:
            return
        if measurement_weights is None:
            measurement_weights = np.ones(len(positions), dtype=np.float64)
        measurement_weights = np.asarray(measurement_weights, dtype=np.float64)
        measurement_weights = np.clip(measurement_weights, 1e-9, None)
        measurement_weights /= measurement_weights.sum()
        log_w = np.zeros(self.n)
        for pos, meas_weight in zip(positions, measurement_weights):
            diff = self.particles - pos
            dist_sq = np.sum(diff ** 2, axis=1)
            log_w -= (
                meas_weight * dist_sq / (2.0 * self.meas_noise_std ** 2)
            )
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
            self.velocities = self.velocities[indices]
            self.weights = np.ones(self.n) / self.n

    def estimate_position(self):
        """Return weighted mean position (3,) or None."""
        if not self.initialized:
            return None
        return np.average(self.particles, axis=0, weights=self.weights)

    def estimate_velocity(self):
        """Return weighted mean velocity (3,) or None."""
        if not self.initialized:
            return None
        return np.average(self.velocities, axis=0, weights=self.weights)
