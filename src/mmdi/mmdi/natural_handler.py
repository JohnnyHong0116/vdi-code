#!/usr/bin/env python3
"""
Natural demonstration handler with optimization-based visual servoing.

Flow:
- In mode 4 (pre-natural): checks head_camera -> target_frame visibility and
  publishes /distance_odom_seen.
- In mode 3 (natural): solves a constrained 6-DoF incremental optimization and
  publishes /ur7e/target_pose.
"""

from typing import Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as ScipyR
from std_msgs.msg import Bool, Int32
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray


def _softplus(x: np.ndarray) -> np.ndarray:
    x_clamped = np.clip(x, -50.0, 50.0)
    return np.log1p(np.exp(x_clamped))


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _quat_from_msg(qmsg) -> np.ndarray:
    return np.array([qmsg.x, qmsg.y, qmsg.z, qmsg.w], dtype=float)


def _safe_normalize(vec: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if (not np.isfinite(norm)) or norm <= eps:
        return np.zeros_like(arr), False
    return arr / norm, True


def _twist_is_informative(lin: np.ndarray, ang: np.ndarray, eps: float = 1e-6) -> bool:
    return bool(
        np.all(np.isfinite(lin))
        and np.all(np.isfinite(ang))
        and (np.linalg.norm(lin) > eps or np.linalg.norm(ang) > eps)
    )


def _predict_target_state(
    tgt_pos: np.ndarray,
    tgt_q: np.ndarray,
    tgt_vel: np.ndarray,
    tgt_omega: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_pred = tgt_pos + tgt_vel * dt
    q_pred = (ScipyR.from_rotvec(tgt_omega * dt) * ScipyR.from_quat(tgt_q)).as_quat()
    return pos_pred, q_pred


def _project_target_points(
    cam_pos: np.ndarray,
    cam_q: np.ndarray,
    tgt_pos: np.ndarray,
    tgt_q: np.ndarray,
    tgt_points: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, bool]:
    r_bc = ScipyR.from_quat(cam_q)
    r_bt = ScipyR.from_quat(tgt_q)

    uv = []
    valid = True
    for pt_t in tgt_points:
        pt_b = r_bt.apply(pt_t) + tgt_pos
        pt_c = r_bc.inv().apply(pt_b - cam_pos)
        if pt_c[2] <= 1e-6:
            valid = False
            continue
        u = fx * (pt_c[0] / pt_c[2]) + cx
        v = fy * (pt_c[1] / pt_c[2]) + cy
        uv.append([u, v])
    return np.array(uv, dtype=float), valid


def _camera_pose_residual(
    x: np.ndarray,
    curr_pos: np.ndarray,
    curr_rotvec: np.ndarray,
    tgt_pos: np.ndarray,
    tgt_q: np.ndarray,
    tgt_vel: np.ndarray,
    tgt_omega: np.ndarray,
    tgt_cov: np.ndarray,
    trans_tool_cam,
    tgt_points: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: float,
    img_h: float,
    safe_margin_px: float,
    prev_cmd: np.ndarray,
    prev_prev_cmd: np.ndarray,
    dt_ctrl: float,
    predict_dt: float,
    distance_objective_mode: str,
    desired_dist: float,
    desired_view_cos: float,
    delta_pos_scale: float,
    delta_rot_scale: float,
    nominal_tool_pos: np.ndarray,
    nominal_pos_span: np.ndarray,
    nominal_pos_axis_weights: np.ndarray,
    rotation_penalty_scale: float,
    preferred_tool_view_axis: np.ndarray,
    min_safe_view_cos: float,
) -> float:
    tool_pos = curr_pos + x[:3]
    tool_rotvec = curr_rotvec + x[3:6]
    q_tool = ScipyR.from_rotvec(tool_rotvec).as_quat()

    q_tool_cam = np.array(
        [
            trans_tool_cam.transform.rotation.x,
            trans_tool_cam.transform.rotation.y,
            trans_tool_cam.transform.rotation.z,
            trans_tool_cam.transform.rotation.w,
        ],
        dtype=float,
    )
    p_tool_cam = np.array(
        [
            trans_tool_cam.transform.translation.x,
            trans_tool_cam.transform.translation.y,
            trans_tool_cam.transform.translation.z,
        ],
        dtype=float,
    )

    r_tool = ScipyR.from_quat(q_tool)
    r_tc = ScipyR.from_quat(q_tool_cam)
    cam_pos = r_tool.apply(p_tool_cam) + tool_pos
    cam_q = (r_tool * r_tc).as_quat()

    tgt_pos_d, tgt_q_d = _predict_target_state(tgt_pos, tgt_q, tgt_vel, tgt_omega, predict_dt)

    pos_cov_trace = float(np.trace(tgt_cov[:3, :3]))
    unc = float(np.clip(pos_cov_trace, 1e-6, 5e-2))
    w_track = 120.0 / (1.0 + 40.0 * unc)
    w_align = 80.0 / (1.0 + 40.0 * unc)
    w_view = 8.0 / (1.0 + 40.0 * unc)
    w_center = 90.0 / (1.0 + 40.0 * unc)
    w_vis = 40.0 * (1.0 + 20.0 * unc)
    w_rear = 160.0 * (1.0 + 20.0 * unc)
    w_nominal = 10.0
    w_smooth = 2.5 * (1.0 + 40.0 * unc)
    w_step = 1.5 * (1.0 + 40.0 * unc)

    vec_ct = tgt_pos_d - cam_pos
    dist = np.linalg.norm(vec_ct) + 1e-9
    view_dir, view_dir_ok = _safe_normalize(vec_ct)
    if distance_objective_mode == "camera_z":
        vec_ct_cam = ScipyR.from_quat(cam_q).inv().apply(vec_ct)
        track_dist = float(vec_ct_cam[2])
    else:
        track_dist = float(dist)
    loss_track = (track_dist - desired_dist) ** 2

    z_cam = ScipyR.from_quat(cam_q).as_matrix()[:, 2]
    loss_align = (1.0 - float(np.dot(view_dir, z_cam))) ** 2 if view_dir_ok else 1e2

    tool_view_axis = ScipyR.from_quat(tgt_q_d).apply(preferred_tool_view_axis)
    tool_view_axis, tool_axis_ok = _safe_normalize(tool_view_axis)
    if view_dir_ok and tool_axis_ok:
        view_cos = float(np.dot(view_dir, tool_view_axis))
        loss_view = (view_cos - desired_view_cos) ** 2
        loss_rear_safety = float(_softplus((min_safe_view_cos - view_cos) / 0.05))
    else:
        loss_view = 1e2
        loss_rear_safety = 1e3

    uv, valid = _project_target_points(
        cam_pos, cam_q, tgt_pos_d, tgt_q_d, tgt_points, fx, fy, cx, cy
    )
    if (not valid) or uv.shape[0] == 0:
        loss_center = 1e2
        loss_vis = 1e3
    else:
        uv_center = np.mean(uv, axis=0)
        uv_center_err = (uv_center - np.array([cx, cy], dtype=float)) / np.array(
            [max(img_w, 1.0), max(img_h, 1.0)],
            dtype=float,
        )
        loss_center = float(np.dot(uv_center_err, uv_center_err))
        margins = np.minimum.reduce(
            [
                uv[:, 0],
                (img_w - 1.0) - uv[:, 0],
                uv[:, 1],
                (img_h - 1.0) - uv[:, 1],
            ]
        )
        loss_vis = float(np.sum(_softplus((safe_margin_px - margins) / 6.0)))

    pos_scale = max(delta_pos_scale, 1e-6)
    rot_scale = max(delta_rot_scale, 1e-6)
    rot_weight = float(max(rotation_penalty_scale, 1e-3))

    x_pos_scaled = x[:3] / pos_scale
    x_rot_scaled = x[3:6] / rot_scale
    prev_pos_scaled = prev_cmd[:3] / pos_scale
    prev_rot_scaled = prev_cmd[3:6] / rot_scale
    prev_prev_pos_scaled = prev_prev_cmd[:3] / pos_scale
    prev_prev_rot_scaled = prev_prev_cmd[3:6] / rot_scale

    pos_rate = (x_pos_scaled - prev_pos_scaled) / max(dt_ctrl, 1e-6)
    rot_rate = (x_rot_scaled - prev_rot_scaled) / max(dt_ctrl, 1e-6)
    pos_jerk = (x_pos_scaled - 2.0 * prev_pos_scaled + prev_prev_pos_scaled) / max(
        dt_ctrl * dt_ctrl, 1e-6
    )
    rot_jerk = (x_rot_scaled - 2.0 * prev_rot_scaled + prev_prev_rot_scaled) / max(
        dt_ctrl * dt_ctrl, 1e-6
    )
    loss_smooth = float(
        np.dot(pos_rate, pos_rate)
        + 0.1 * np.dot(pos_jerk, pos_jerk)
        + rot_weight * (np.dot(rot_rate, rot_rate) + 0.1 * np.dot(rot_jerk, rot_jerk))
    )
    loss_step = float(
        np.dot(x_pos_scaled, x_pos_scaled)
        + rot_weight * np.dot(x_rot_scaled, x_rot_scaled)
    )

    nominal_err = (tool_pos - nominal_tool_pos) / nominal_pos_span
    loss_nominal = float(np.dot(nominal_pos_axis_weights * nominal_err, nominal_err))

    return (
        w_track * loss_track
        + w_align * loss_align
        + w_view * loss_view
        + w_center * loss_center
        + w_vis * loss_vis
        + w_rear * loss_rear_safety
        + w_nominal * loss_nominal
        + w_smooth * loss_smooth
        + w_step * loss_step
    )


class NaturalHandler(Node):
    def __init__(self):
        super().__init__("natural_handler")

        # Frames / topics
        self.declare_parameter("mode_topic", "/mode")
        self.declare_parameter("odom_seen_topic", "/distance_odom_seen")
        self.declare_parameter("target_pose_topic", "/ur7e/target_pose")
        self.declare_parameter("fused_pose_topic", "/ur7e/fused_tool_pose")
        self.declare_parameter("fused_twist_topic", "/ur7e/fused_tool_twist")
        self.declare_parameter("fused_odom_topic", "/vo")
        self.declare_parameter("probe_markers_topic", "/probe_tracker/markers")
        self.declare_parameter("camera_info_topic", "/probe_tracker/camera_info")
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("camera_frame", "head_camera")
        self.declare_parameter("target_frame", "odom")

        # Timing / freshness
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("vision_delay_sec", 0.045)
        self.declare_parameter("motor_delay_sec", 0.020)
        self.declare_parameter("hard_stale_sec", 1.5)
        self.declare_parameter("external_target_timeout_sec", 0.5)
        self.declare_parameter("use_tf_target_fallback", True)
        self.declare_parameter("pre_natural_z_threshold", 0.10)
        self.declare_parameter("pre_natural_stale_sec", 1.0)
        self.declare_parameter("pre_natural_search_enabled", True)
        self.declare_parameter("pre_natural_search_hold_sec", 0.35)
        self.declare_parameter("pre_natural_search_shrink", 0.6)
        self.declare_parameter("pre_natural_search_include_center", True)
        self.declare_parameter("pre_natural_search_include_current_start", False)
        self.declare_parameter("pre_natural_search_rotate", True)
        self.declare_parameter("pre_natural_search_yaw_sweep_deg", 12.0)
        self.declare_parameter("pre_natural_orbit_enabled", True)
        self.declare_parameter("pre_natural_orbit_distance", 0.30)
        self.declare_parameter("pre_natural_orbit_z_offset", 0.02)
        self.declare_parameter("natural_search_on_lost_target", True)
        self.declare_parameter("use_priority_tag_tracking", True)
        self.declare_parameter("track_only_in_natural_modes", True)
        self.declare_parameter("tracking_tag_ids", [2, 0, 7])
        self.declare_parameter("tracking_tag_timeout_sec", 0.35)
        self.declare_parameter("tracking_allow_nonfused_fallback", True)
        self.declare_parameter("min_priority_tags_for_multi_visibility", 2)
        self.declare_parameter("use_fused_cam_z_for_seen", True)
        self.declare_parameter("too_close_backoff_enabled", True)
        self.declare_parameter("too_close_distance_margin", 0.03)
        self.declare_parameter("too_close_backoff_step", 0.008)

        # Workspace and optimizer limits (safe working defaults)
        self.declare_parameter("workspace_side", "left")  # left | right | auto
        self.declare_parameter("workspace_side_hysteresis", 0.02)
        self.declare_parameter("x_min", 0.20)
        self.declare_parameter("x_max", 0.30)
        self.declare_parameter("y_min", -0.22)   # left side
        self.declare_parameter("y_max", -0.08)   # left side
        self.declare_parameter("y_right_min", 0.08)
        self.declare_parameter("y_right_max", 0.22)
        self.declare_parameter("z_min", 0.30)
        self.declare_parameter("z_max", 0.40)
        self.declare_parameter("delta_pos_max", 0.0006)
        self.declare_parameter("delta_rot_max", 0.015)
        self.declare_parameter("maxiter", 40)
        self.declare_parameter("max_angular_deviation", 0.9)

        # View objective
        self.declare_parameter("distance_objective_mode", "camera_z")
        self.declare_parameter("desired_distance", 0.30)
        self.declare_parameter("desired_view_cos", 0.85)
        self.declare_parameter("preferred_tool_view_axis", [0.0, 1.0, 0.0])
        self.declare_parameter("min_safe_view_cos", 0.15)
        self.declare_parameter("nominal_x_fraction", 0.12)
        self.declare_parameter("nominal_y_fraction", 0.50)
        self.declare_parameter("nominal_z_fraction", 0.15)
        self.declare_parameter("nominal_x_axis_weight", 1.5)
        self.declare_parameter("nominal_y_axis_weight", 0.25)
        self.declare_parameter("nominal_z_axis_weight", 1.5)
        self.declare_parameter("rotation_penalty_scale", 0.08)

        # Image model for visibility barrier
        self.declare_parameter("fx", 615.0)
        self.declare_parameter("fy", 615.0)
        self.declare_parameter("cx", 320.0)
        self.declare_parameter("cy", 240.0)
        self.declare_parameter("image_width", 640.0)
        self.declare_parameter("image_height", 480.0)
        self.declare_parameter("safe_margin_px", 40.0)

        # Target proxy points
        self.declare_parameter("target_half_extent_x", 0.03)
        self.declare_parameter("target_half_extent_y", 0.03)

        self.mode_topic = str(self.get_parameter("mode_topic").value)
        self.odom_seen_topic = str(self.get_parameter("odom_seen_topic").value)
        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self.fused_pose_topic = str(self.get_parameter("fused_pose_topic").value)
        self.fused_twist_topic = str(self.get_parameter("fused_twist_topic").value)
        self.fused_odom_topic = str(self.get_parameter("fused_odom_topic").value)
        self.probe_markers_topic = str(self.get_parameter("probe_markers_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tool_frame = str(self.get_parameter("tool_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.target_frame = str(self.get_parameter("target_frame").value)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.vision_delay_sec = float(self.get_parameter("vision_delay_sec").value)
        self.motor_delay_sec = float(self.get_parameter("motor_delay_sec").value)
        self.hard_stale_sec = float(self.get_parameter("hard_stale_sec").value)
        self.workspace_side = str(self.get_parameter("workspace_side").value).lower()
        self.workspace_side_hys = float(self.get_parameter("workspace_side_hysteresis").value)
        self.external_target_timeout_sec = float(
            self.get_parameter("external_target_timeout_sec").value
        )
        self.use_tf_target_fallback = bool(self.get_parameter("use_tf_target_fallback").value)
        self.pre_natural_z_threshold = float(
            self.get_parameter("pre_natural_z_threshold").value
        )
        self.pre_natural_stale_sec = float(
            self.get_parameter("pre_natural_stale_sec").value
        )
        self.pre_natural_search_enabled = bool(
            self.get_parameter("pre_natural_search_enabled").value
        )
        self.pre_natural_search_hold_sec = float(
            self.get_parameter("pre_natural_search_hold_sec").value
        )
        self.pre_natural_search_shrink = float(
            self.get_parameter("pre_natural_search_shrink").value
        )
        self.pre_natural_search_include_center = bool(
            self.get_parameter("pre_natural_search_include_center").value
        )
        self.pre_natural_search_include_current_start = bool(
            self.get_parameter("pre_natural_search_include_current_start").value
        )
        self.pre_natural_search_rotate = bool(
            self.get_parameter("pre_natural_search_rotate").value
        )
        self.pre_natural_search_yaw_sweep_deg = float(
            self.get_parameter("pre_natural_search_yaw_sweep_deg").value
        )
        self.pre_natural_orbit_enabled = bool(
            self.get_parameter("pre_natural_orbit_enabled").value
        )
        self.pre_natural_orbit_distance = float(
            self.get_parameter("pre_natural_orbit_distance").value
        )
        self.pre_natural_orbit_z_offset = float(
            self.get_parameter("pre_natural_orbit_z_offset").value
        )
        self.natural_search_on_lost_target = bool(
            self.get_parameter("natural_search_on_lost_target").value
        )
        self.use_priority_tag_tracking = bool(
            self.get_parameter("use_priority_tag_tracking").value
        )
        self.track_only_in_natural_modes = bool(
            self.get_parameter("track_only_in_natural_modes").value
        )
        self.tracking_tag_ids = [
            int(tag_id) for tag_id in self.get_parameter("tracking_tag_ids").value
        ]
        self.tracking_tag_timeout_sec = float(
            self.get_parameter("tracking_tag_timeout_sec").value
        )
        self.tracking_allow_nonfused_fallback = bool(
            self.get_parameter("tracking_allow_nonfused_fallback").value
        )
        self.min_priority_tags_for_multi_visibility = int(
            self.get_parameter("min_priority_tags_for_multi_visibility").value
        )
        self.use_fused_cam_z_for_seen = bool(
            self.get_parameter("use_fused_cam_z_for_seen").value
        )
        self.too_close_backoff_enabled = bool(
            self.get_parameter("too_close_backoff_enabled").value
        )
        self.too_close_distance_margin = float(
            self.get_parameter("too_close_distance_margin").value
        )
        self.too_close_backoff_step = float(
            self.get_parameter("too_close_backoff_step").value
        )
        self.distance_objective_mode = str(
            self.get_parameter("distance_objective_mode").value
        ).strip().lower()
        if self.distance_objective_mode not in ("camera_z", "euclidean"):
            self.distance_objective_mode = "euclidean"

        self.left_cart_mins = np.array(
            [
                float(self.get_parameter("x_min").value),
                float(self.get_parameter("y_min").value),
                float(self.get_parameter("z_min").value),
            ],
            dtype=float,
        )
        self.left_cart_maxs = np.array(
            [
                float(self.get_parameter("x_max").value),
                float(self.get_parameter("y_max").value),
                float(self.get_parameter("z_max").value),
            ],
            dtype=float,
        )
        self.right_cart_mins = np.array(
            [
                float(self.get_parameter("x_min").value),
                float(self.get_parameter("y_right_min").value),
                float(self.get_parameter("z_min").value),
            ],
            dtype=float,
        )
        self.right_cart_maxs = np.array(
            [
                float(self.get_parameter("x_max").value),
                float(self.get_parameter("y_right_max").value),
                float(self.get_parameter("z_max").value),
            ],
            dtype=float,
        )
        self.active_workspace_side = "left"
        self.delta_pos_max = float(self.get_parameter("delta_pos_max").value)
        self.delta_rot_max = float(self.get_parameter("delta_rot_max").value)
        self.maxiter = int(self.get_parameter("maxiter").value)
        self.max_angular_deviation = float(self.get_parameter("max_angular_deviation").value)

        self.desired_dist = float(self.get_parameter("desired_distance").value)
        self.desired_view_cos = float(self.get_parameter("desired_view_cos").value)
        preferred_view_axis = np.asarray(
            self.get_parameter("preferred_tool_view_axis").value,
            dtype=float,
        ).reshape(-1)
        if preferred_view_axis.size != 3:
            preferred_view_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        preferred_view_axis, axis_ok = _safe_normalize(preferred_view_axis)
        self.preferred_tool_view_axis = (
            preferred_view_axis
            if axis_ok
            else np.array([0.0, 1.0, 0.0], dtype=float)
        )
        self.min_safe_view_cos = float(
            self.get_parameter("min_safe_view_cos").value
        )
        self.nominal_fractions = np.array(
            [
                float(self.get_parameter("nominal_x_fraction").value),
                float(self.get_parameter("nominal_y_fraction").value),
                float(self.get_parameter("nominal_z_fraction").value),
            ],
            dtype=float,
        )
        self.nominal_axis_weights = np.array(
            [
                float(self.get_parameter("nominal_x_axis_weight").value),
                float(self.get_parameter("nominal_y_axis_weight").value),
                float(self.get_parameter("nominal_z_axis_weight").value),
            ],
            dtype=float,
        )
        self.rotation_penalty_scale = float(
            self.get_parameter("rotation_penalty_scale").value
        )

        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.cx = float(self.get_parameter("cx").value)
        self.cy = float(self.get_parameter("cy").value)
        self.img_w = float(self.get_parameter("image_width").value)
        self.img_h = float(self.get_parameter("image_height").value)
        self.safe_margin_px = float(self.get_parameter("safe_margin_px").value)

        hx = float(self.get_parameter("target_half_extent_x").value)
        hy = float(self.get_parameter("target_half_extent_y").value)
        self.tgt_points = np.array(
            [[hx, hy, 0.0], [hx, -hy, 0.0], [-hx, -hy, 0.0], [-hx, hy, 0.0]],
            dtype=float,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Mode/state
        self.mode = 0
        self.start_mode3_q = None
        self.start_mode3_pos = None

        # External fused target state
        self.tgt_pos = None
        self.tgt_q = None
        self.tgt_cov = np.eye(6, dtype=float) * 1e-3
        self.tgt_vel = np.zeros(3, dtype=float)
        self.tgt_omega = np.zeros(3, dtype=float)
        self.last_target_update_sec = None
        self.last_twist_update_sec = None
        self._last_pose_for_vel = None  # (pos, quat, sec)
        self._warned_frame_mismatch = False
        self._target_visible = False
        self.tag_tgt_pos = None
        self.tag_tgt_q = None
        self.tag_tgt_cov = np.eye(6, dtype=float) * 1e-3
        self.tag_tgt_vel = np.zeros(3, dtype=float)
        self.tag_tgt_omega = np.zeros(3, dtype=float)
        self.last_tag_update_sec = None
        self._last_tag_pose_for_vel = None  # (pos, quat, sec)
        self.last_tracking_tag_id = None
        self.last_tracking_tag_cam_z = None
        self.last_priority_tag_points_base = {}
        self.last_priority_tag_points_sec = None
        self.pre_nat_search_started = False
        self.pre_nat_search_waypoints = []
        self.pre_nat_search_quats = []
        self.pre_nat_search_idx = 0
        self.pre_nat_search_waypoint_start_sec = 0.0
        self.pre_nat_search_orientation = None

        # TF fallback state
        self._last_tf_pose = None  # (pos, quat, sec)
        self.last_vo_cov = None
        self.last_vo_cov_sec = None
        self.last_fused_cam_z = None
        self.last_fused_cam_z_sec = None

        # Command smoothing memory
        self.prev_cmd = np.zeros(6, dtype=float)
        self.prev_prev_cmd = np.zeros(6, dtype=float)
        self.ctrl_dt = 1.0 / max(self.rate_hz, 1e-6)

        self.odom_seen_pub = self.create_publisher(Bool, self.odom_seen_topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, self.target_pose_topic, 1)

        self.create_subscription(Int32, self.mode_topic, self.on_mode, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, self.fused_pose_topic, self.on_fused_pose, 10
        )
        self.create_subscription(TwistStamped, self.fused_twist_topic, self.on_fused_twist, 10)
        self.create_subscription(Odometry, self.fused_odom_topic, self.on_fused_odom, 10)
        self.create_subscription(MarkerArray, self.probe_markers_topic, self.on_probe_markers, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_camera_info, 10)

        self.timer = self.create_timer(self.ctrl_dt, self.tick)

        self.get_logger().info("NaturalHandler (Option B) active")
        self.get_logger().info(f"mode topic: {self.mode_topic}")
        self.get_logger().info(f"target pose topic: {self.target_pose_topic}")
        self.get_logger().info(
            f"fused topics: pose={self.fused_pose_topic}, twist={self.fused_twist_topic}, odom={self.fused_odom_topic}"
        )
        self.get_logger().info(f"probe markers topic: {self.probe_markers_topic}")
        self.get_logger().info(f"camera info topic: {self.camera_info_topic}")
        self.get_logger().info(
            f"priority tag tracking: {self.use_priority_tag_tracking}, preferred ids={self.tracking_tag_ids}"
        )
        self.get_logger().info(
            f"track tags only in mode 3/4: {self.track_only_in_natural_modes}"
        )
        self.get_logger().info(
            f"multi-tag visibility min count: {self.min_priority_tags_for_multi_visibility}"
        )
        self.get_logger().info(
            f"seen source: fused_cam_z={self.use_fused_cam_z_for_seen}, "
            f"distance objective={self.distance_objective_mode}"
        )
        self.get_logger().info(
            "too-close backoff: "
            f"enabled={self.too_close_backoff_enabled}, "
            f"margin={self.too_close_distance_margin:.3f}m, "
            f"step={self.too_close_backoff_step:.3f}m"
        )
        self.get_logger().info(
            "pre-natural search: "
            f"enabled={self.pre_natural_search_enabled}, "
            f"hold={self.pre_natural_search_hold_sec:.2f}s, "
            f"shrink={self.pre_natural_search_shrink:.2f}, "
            f"rotate={self.pre_natural_search_rotate}, "
            f"yaw_sweep={self.pre_natural_search_yaw_sweep_deg:.1f}deg"
        )
        self.get_logger().info(
            f"natural lost-target search: {self.natural_search_on_lost_target}"
        )
        self.get_logger().info(
            f"distance objective: desired_distance={self.desired_dist:.3f} m"
        )
        self.get_logger().info(
            "optimizer bias: "
            f"mode={self.distance_objective_mode}, "
            f"nominal_frac={np.round(self.nominal_fractions, 2).tolist()}, "
            f"rotation_penalty_scale={self.rotation_penalty_scale:.2f}, "
            f"tool_view_axis={np.round(self.preferred_tool_view_axis, 2).tolist()}, "
            f"desired_view_cos={self.desired_view_cos:.2f}, "
            f"min_safe_view_cos={self.min_safe_view_cos:.2f}"
        )
        self.get_logger().info(f"workspace_side mode: {self.workspace_side}")
        self.get_logger().info(
            f"left box  x[{self.left_cart_mins[0]:.3f},{self.left_cart_maxs[0]:.3f}] "
            f"y[{self.left_cart_mins[1]:.3f},{self.left_cart_maxs[1]:.3f}] "
            f"z[{self.left_cart_mins[2]:.3f},{self.left_cart_maxs[2]:.3f}]"
        )
        self.get_logger().info(
            f"right box x[{self.right_cart_mins[0]:.3f},{self.right_cart_maxs[0]:.3f}] "
            f"y[{self.right_cart_mins[1]:.3f},{self.right_cart_maxs[1]:.3f}] "
            f"z[{self.right_cart_mins[2]:.3f},{self.right_cart_maxs[2]:.3f}]"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_mode(self, msg: Int32):
        prev_mode = self.mode
        self.mode = int(msg.data)

        if prev_mode != 3 and self.mode == 3:
            self.prev_cmd[:] = 0.0
            self.prev_prev_cmd[:] = 0.0
            self.start_mode3_q = None
            self.start_mode3_pos = None
        if prev_mode != 4 and self.mode == 4:
            self.pre_nat_search_started = False
            self.pre_nat_search_waypoints = []
            self.pre_nat_search_quats = []
            self.pre_nat_search_idx = 0
            self.pre_nat_search_waypoint_start_sec = 0.0
            self.pre_nat_search_orientation = None
        elif self.mode != 3:
            self.start_mode3_q = None
            self.start_mode3_pos = None

        if self.mode not in (3, 4):
            self._reset_tag_tracking_state()
            self.pre_nat_search_started = False
            self.pre_nat_search_waypoints = []
            self.pre_nat_search_quats = []
            self.pre_nat_search_idx = 0
            self.pre_nat_search_waypoint_start_sec = 0.0
            self.pre_nat_search_orientation = None
            self.odom_seen_pub.publish(Bool(data=False))
            self._target_visible = False

    def _get_nominal_tool_pos(
        self,
        cart_mins: np.ndarray,
        cart_maxs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        frac = np.clip(self.nominal_fractions, 0.0, 1.0)
        nominal = cart_mins + frac * (cart_maxs - cart_mins)
        if self.start_mode3_pos is not None:
            start_clipped = np.clip(self.start_mode3_pos, cart_mins, cart_maxs)
            nominal = 0.65 * nominal + 0.35 * start_clipped
        span = np.maximum(cart_maxs - cart_mins, 1e-3)
        return nominal, span

    def _reset_tag_tracking_state(self):
        self.tag_tgt_pos = None
        self.tag_tgt_q = None
        self.tag_tgt_cov = np.eye(6, dtype=float) * 1e-3
        self.tag_tgt_vel = np.zeros(3, dtype=float)
        self.tag_tgt_omega = np.zeros(3, dtype=float)
        self.last_tag_update_sec = None
        self._last_tag_pose_for_vel = None
        self.last_tracking_tag_id = None
        self.last_tracking_tag_cam_z = None
        self.last_priority_tag_points_base = {}
        self.last_priority_tag_points_sec = None
        self.last_fused_cam_z = None
        self.last_fused_cam_z_sec = None

    def on_camera_info(self, msg: CameraInfo):
        if msg.width > 0:
            self.img_w = float(msg.width)
        if msg.height > 0:
            self.img_h = float(msg.height)
        if len(msg.k) >= 9:
            fx = float(msg.k[0])
            fy = float(msg.k[4])
            cx = float(msg.k[2])
            cy = float(msg.k[5])
            if all(np.isfinite(v) for v in (fx, fy, cx, cy)) and fx > 1e-6 and fy > 1e-6:
                self.fx = fx
                self.fy = fy
                self.cx = cx
                self.cy = cy

    def _publish_target_visible(self, visible: bool):
        visible = bool(visible)
        if visible != self._target_visible:
            self._target_visible = visible
            self.odom_seen_pub.publish(Bool(data=visible))
        elif visible:
            # Keep publishing True periodically so mode_handler can stay in sync.
            self.odom_seen_pub.publish(Bool(data=True))

    def _update_target_from_pose(
        self, pos: np.ndarray, quat: np.ndarray, cov: np.ndarray, stamp_sec: float
    ):
        now_sec = self._now_sec()
        t_sec = stamp_sec if stamp_sec > 0.0 else now_sec

        if self._last_pose_for_vel is not None:
            p_prev, q_prev, t_prev = self._last_pose_for_vel
            if t_sec > t_prev:
                dt = max(t_sec - t_prev, 1e-6)
                self.tgt_vel = (pos - p_prev) / dt
                r_delta = ScipyR.from_quat(quat) * ScipyR.from_quat(q_prev).inv()
                self.tgt_omega = r_delta.as_rotvec() / dt

        self._last_pose_for_vel = (pos.copy(), quat.copy(), t_sec)
        self.tgt_pos = pos.copy()
        self.tgt_q = quat.copy()
        self.tgt_cov = cov.copy()
        self.last_target_update_sec = t_sec

    def on_fused_pose(self, msg: PoseWithCovarianceStamped):
        frame = msg.header.frame_id
        if frame and frame != self.base_frame:
            if not self._warned_frame_mismatch:
                self.get_logger().warn(
                    f"fused pose frame '{frame}' != base_frame '{self.base_frame}'. Ignoring fused pose."
                )
                self._warned_frame_mismatch = True
            return

        pos = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            dtype=float,
        )
        quat = _quat_from_msg(msg.pose.pose.orientation)
        cov = np.array(msg.pose.covariance, dtype=float).reshape(6, 6)
        if np.allclose(cov, 0.0):
            cov = np.eye(6, dtype=float) * 1e-4
        else:
            cov = 0.5 * (cov + cov.T)

        self._update_target_from_pose(pos, quat, cov, _stamp_to_sec(msg.header.stamp))

    def on_fused_twist(self, msg: TwistStamped):
        self.tgt_vel = np.array(
            [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], dtype=float
        )
        self.tgt_omega = np.array(
            [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z], dtype=float
        )
        self.last_twist_update_sec = _stamp_to_sec(msg.header.stamp)
        if self.last_twist_update_sec <= 0.0:
            self.last_twist_update_sec = self._now_sec()

    def on_fused_odom(self, msg: Odometry):
        cov = np.array(msg.pose.covariance, dtype=float).reshape(6, 6)
        if np.allclose(cov, 0.0):
            cov = np.eye(6, dtype=float) * 1e-4
        else:
            cov = 0.5 * (cov + cov.T)
        self.last_vo_cov = cov
        stamp_sec = _stamp_to_sec(msg.header.stamp)
        self.last_vo_cov_sec = stamp_sec if stamp_sec > 0.0 else self._now_sec()
        frame_id = msg.header.frame_id if msg.header.frame_id else self.camera_frame

        # Probe tracker /vo is usually odom pose in camera frame; keep z for
        # pre-natural visibility checks.
        if frame_id == self.camera_frame:
            self.last_fused_cam_z = float(msg.pose.pose.position.z)
            self.last_fused_cam_z_sec = self.last_vo_cov_sec

        # If odometry already comes in base frame, accept it directly.
        if frame_id == self.base_frame:
            pos = np.array(
                [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
                dtype=float,
            )
            quat = _quat_from_msg(msg.pose.pose.orientation)
            self._update_target_from_pose(pos, quat, cov, _stamp_to_sec(msg.header.stamp))
            lin = np.array(
                [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
                dtype=float,
            )
            ang = np.array(
                [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z],
                dtype=float,
            )
            if _twist_is_informative(lin, ang):
                self.tgt_vel = lin
                self.tgt_omega = ang
                self.last_twist_update_sec = self._now_sec()
            return

        # Transform odom pose from source frame to base frame.
        try:
            trans = self.tf_buffer.lookup_transform(self.base_frame, frame_id, Time())
        except Exception:
            return

        p_bf = np.array(
            [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ],
            dtype=float,
        )
        q_bf = _quat_from_msg(trans.transform.rotation)
        r_bf = ScipyR.from_quat(q_bf)

        pos_f = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            dtype=float,
        )
        quat_f = _quat_from_msg(msg.pose.pose.orientation)
        pos_base = r_bf.apply(pos_f) + p_bf
        quat_base = (r_bf * ScipyR.from_quat(quat_f)).as_quat()
        self._update_target_from_pose(pos_base, quat_base, cov, stamp_sec)

        lin_f = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
            dtype=float,
        )
        ang_f = np.array(
            [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z],
            dtype=float,
        )
        lin_base = r_bf.apply(lin_f)
        ang_base = r_bf.apply(ang_f)
        if _twist_is_informative(lin_base, ang_base):
            self.tgt_vel = lin_base
            self.tgt_omega = ang_base
            self.last_twist_update_sec = self._now_sec()

    def _select_priority_marker(self, markers):
        candidates = []
        for marker in markers:
            if marker.ns != "april_tags":
                continue
            tag_id = int(marker.id)
            # probe_tracker colors fused tags as green-ish and non-fused as orange-ish.
            used_for_fusion = float(marker.color.g) >= float(marker.color.r)
            candidates.append((tag_id, marker, used_for_fusion))

        if not candidates:
            return None

        for preferred_id in self.tracking_tag_ids:
            for tag_id, marker, used_for_fusion in candidates:
                if tag_id == preferred_id and used_for_fusion:
                    return tag_id, marker, used_for_fusion

        for preferred_id in self.tracking_tag_ids:
            for tag_id, marker, _ in candidates:
                if tag_id == preferred_id and self.tracking_allow_nonfused_fallback:
                    return tag_id, marker, False

        for tag_id, marker, used_for_fusion in candidates:
            if used_for_fusion:
                return tag_id, marker, used_for_fusion

        if self.tracking_allow_nonfused_fallback:
            tag_id, marker, used_for_fusion = candidates[0]
            return tag_id, marker, used_for_fusion

        return None

    def on_probe_markers(self, msg: MarkerArray):
        if not self.use_priority_tag_tracking:
            return
        if self.track_only_in_natural_modes and self.mode not in (3, 4):
            return

        now_sec = self._now_sec()
        preferred_points_base = {}
        for marker in msg.markers:
            if marker.ns != "april_tags":
                continue
            tag_id = int(marker.id)
            if tag_id not in self.tracking_tag_ids:
                continue
            frame_id = marker.header.frame_id if marker.header.frame_id else self.camera_frame
            pos_frame = np.array(
                [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],
                dtype=float,
            )
            try:
                trans = self.tf_buffer.lookup_transform(self.base_frame, frame_id, Time())
            except Exception:
                continue
            p_bf = np.array(
                [
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z,
                ],
                dtype=float,
            )
            q_bf = _quat_from_msg(trans.transform.rotation)
            r_bf = ScipyR.from_quat(q_bf)
            preferred_points_base[tag_id] = r_bf.apply(pos_frame) + p_bf

        if preferred_points_base:
            self.last_priority_tag_points_base = preferred_points_base
            self.last_priority_tag_points_sec = now_sec

        selected = self._select_priority_marker(msg.markers)
        if selected is None:
            return

        tag_id, marker, _ = selected
        frame_id = marker.header.frame_id if marker.header.frame_id else self.camera_frame

        pos_frame = np.array(
            [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],
            dtype=float,
        )
        quat_frame = _quat_from_msg(marker.pose.orientation)
        if np.linalg.norm(quat_frame) < 1e-9:
            quat_frame = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.last_tracking_tag_cam_z = float(pos_frame[2])

        try:
            trans = self.tf_buffer.lookup_transform(self.base_frame, frame_id, Time())
        except Exception:
            return

        p_bf = np.array(
            [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ],
            dtype=float,
        )
        q_bf = _quat_from_msg(trans.transform.rotation)
        r_bf = ScipyR.from_quat(q_bf)
        r_ft = ScipyR.from_quat(quat_frame)

        pos_base = r_bf.apply(pos_frame) + p_bf
        quat_base = (r_bf * r_ft).as_quat()

        stamp_sec = _stamp_to_sec(marker.header.stamp)
        if stamp_sec <= 0.0:
            stamp_sec = self._now_sec()

        if self._last_tag_pose_for_vel is not None:
            p_prev, q_prev, t_prev = self._last_tag_pose_for_vel
            if stamp_sec > t_prev:
                dt = max(stamp_sec - t_prev, 1e-6)
                self.tag_tgt_vel = (pos_base - p_prev) / dt
                r_delta = ScipyR.from_quat(quat_base) * ScipyR.from_quat(q_prev).inv()
                self.tag_tgt_omega = r_delta.as_rotvec() / dt

        self._last_tag_pose_for_vel = (pos_base.copy(), quat_base.copy(), stamp_sec)
        self.tag_tgt_pos = pos_base
        self.tag_tgt_q = quat_base
        self.tag_tgt_cov = np.eye(6, dtype=float) * 1e-3
        self.last_tag_update_sec = stamp_sec

        if tag_id != self.last_tracking_tag_id:
            self.last_tracking_tag_id = tag_id
            self.get_logger().info(f"priority tracking tag -> {tag_id}")

    def _get_target_proxy_points(self, tgt_pos: np.ndarray, tgt_q: np.ndarray) -> np.ndarray:
        if (
            self.use_priority_tag_tracking
            and self.last_priority_tag_points_sec is not None
            and (self._now_sec() - self.last_priority_tag_points_sec) <= self.tracking_tag_timeout_sec
        ):
            r_tgt = ScipyR.from_quat(tgt_q)
            rel_pts = []
            for tag_id in self.tracking_tag_ids:
                if tag_id in self.last_priority_tag_points_base:
                    p_base = self.last_priority_tag_points_base[tag_id]
                    rel = r_tgt.inv().apply(p_base - tgt_pos)
                    rel_pts.append(rel)
            if len(rel_pts) >= max(1, self.min_priority_tags_for_multi_visibility):
                return np.array(rel_pts, dtype=float)
        return self.tgt_points

    def _publish_pre_natural_seen(self):
        if (
            self.use_fused_cam_z_for_seen
            and self.last_fused_cam_z is not None
            and self.last_fused_cam_z_sec is not None
        ):
            age = self._now_sec() - self.last_fused_cam_z_sec
            if age < self.pre_natural_stale_sec:
                seen = abs(self.last_fused_cam_z) > self.pre_natural_z_threshold
                self.odom_seen_pub.publish(Bool(data=bool(seen)))
                return bool(seen)

        if self.use_priority_tag_tracking and self.last_tag_update_sec is not None:
            age = self._now_sec() - self.last_tag_update_sec
            seen = (
                self.last_tracking_tag_cam_z is not None
                and abs(self.last_tracking_tag_cam_z) > self.pre_natural_z_threshold
                and age < self.pre_natural_stale_sec
            )
            self.odom_seen_pub.publish(Bool(data=bool(seen)))
            return bool(seen)

        try:
            trans = self.tf_buffer.lookup_transform(
                self.camera_frame, self.target_frame, Time()
            )
            stamp_sec = _stamp_to_sec(trans.header.stamp)
            age = self._now_sec() - stamp_sec if stamp_sec > 0.0 else 0.0
            seen = (
                abs(trans.transform.translation.z) > self.pre_natural_z_threshold
                and age < self.pre_natural_stale_sec
            )
            self.odom_seen_pub.publish(Bool(data=bool(seen)))
            return bool(seen)
        except Exception:
            self.odom_seen_pub.publish(Bool(data=False))
            return False

    def _build_pre_natural_search_waypoints(
        self,
        curr_pos: np.ndarray,
        curr_q: np.ndarray,
        enable_rotation_sweep: bool = False,
    ):
        tgt_hint = self.tag_tgt_pos if self.tag_tgt_pos is not None else curr_pos
        cart_mins, cart_maxs = self._select_workspace(curr_pos, tgt_hint)

        center = 0.5 * (cart_mins + cart_maxs)
        half = 0.5 * (cart_maxs - cart_mins)
        scale = float(np.clip(self.pre_natural_search_shrink, 0.1, 1.0))
        half_scaled = half * scale
        mins = center - half_scaled
        maxs = center + half_scaled

        waypoints = []
        if self.pre_natural_search_include_current_start:
            waypoints.append(curr_pos.copy())

        orbit_added = False
        if self.pre_natural_orbit_enabled and self.tag_tgt_pos is not None:
            orbit_r = float(self.pre_natural_orbit_distance)
            if orbit_r <= 1e-6:
                orbit_r = float(self.desired_dist)
            orbit_r = float(np.clip(orbit_r, 0.12, 0.60))
            z_off = float(self.pre_natural_orbit_z_offset)

            # Keep current orientation during search, so camera offset is fixed in base.
            try:
                trans_tool_cam = self.tf_buffer.lookup_transform(
                    self.tool_frame, self.camera_frame, Time()
                )
                p_tool_cam = np.array(
                    [
                        trans_tool_cam.transform.translation.x,
                        trans_tool_cam.transform.translation.y,
                        trans_tool_cam.transform.translation.z,
                    ],
                    dtype=float,
                )
                cam_offset_base = ScipyR.from_quat(curr_q).apply(p_tool_cam)
            except Exception:
                cam_offset_base = np.zeros(3, dtype=float)

            c = 0.70710678
            cam_offsets = [
                np.array([orbit_r, 0.0, z_off], dtype=float),
                np.array([-orbit_r, 0.0, z_off], dtype=float),
                np.array([0.0, orbit_r, z_off], dtype=float),
                np.array([0.0, -orbit_r, z_off], dtype=float),
                np.array([c * orbit_r, c * orbit_r, z_off], dtype=float),
                np.array([c * orbit_r, -c * orbit_r, z_off], dtype=float),
                np.array([-c * orbit_r, c * orbit_r, z_off], dtype=float),
                np.array([-c * orbit_r, -c * orbit_r, z_off], dtype=float),
            ]

            orbit_wps = []
            for cam_off in cam_offsets:
                cam_pos = self.tag_tgt_pos + cam_off
                tool_pos = np.clip(cam_pos - cam_offset_base, mins, maxs)
                if np.linalg.norm(tool_pos - curr_pos) < 0.003:
                    continue
                if any(np.linalg.norm(tool_pos - wp) < 0.003 for wp in orbit_wps):
                    continue
                orbit_wps.append(tool_pos)

            if orbit_wps:
                tgt_q_hint = self.tgt_q if self.tgt_q is not None else self.tag_tgt_q
                preferred_axis_base = None
                if tgt_q_hint is not None:
                    preferred_axis_base = ScipyR.from_quat(tgt_q_hint).apply(
                        self.preferred_tool_view_axis
                    )
                    preferred_axis_base, ok = _safe_normalize(preferred_axis_base)
                    if not ok:
                        preferred_axis_base = None

                def orbit_sort_key(wp):
                    wp_dist = float(np.linalg.norm(wp - curr_pos))
                    if preferred_axis_base is None:
                        return (wp_dist,)
                    cam_pos = wp + cam_offset_base
                    cam_view = self.tag_tgt_pos - cam_pos
                    cam_off_dir, ok = _safe_normalize(cam_view)
                    if not ok:
                        return (1.0, wp_dist)
                    view_score = float(np.dot(cam_off_dir, preferred_axis_base))
                    return (-view_score, wp_dist)

                orbit_wps.sort(key=orbit_sort_key)
                waypoints.extend(orbit_wps)
                orbit_added = True

        if not orbit_added:
            if self.pre_natural_search_include_center:
                waypoints.append(center.copy())
            waypoints.extend(
                [
                    np.array([mins[0], mins[1], mins[2]], dtype=float),
                    np.array([maxs[0], mins[1], mins[2]], dtype=float),
                    np.array([maxs[0], maxs[1], mins[2]], dtype=float),
                    np.array([mins[0], maxs[1], mins[2]], dtype=float),
                    np.array([mins[0], mins[1], maxs[2]], dtype=float),
                    np.array([maxs[0], mins[1], maxs[2]], dtype=float),
                    np.array([maxs[0], maxs[1], maxs[2]], dtype=float),
                    np.array([mins[0], maxs[1], maxs[2]], dtype=float),
                ]
            )

        if self.pre_natural_search_include_center:
            waypoints.append(center.copy())
        if not waypoints:
            waypoints = [center.copy()]

        quats = [curr_q.copy() for _ in waypoints]
        if (
            enable_rotation_sweep
            and self.pre_natural_search_rotate
            and len(waypoints) > 0
            and abs(self.pre_natural_search_yaw_sweep_deg) > 1e-6
        ):
            yaw_rad = np.deg2rad(self.pre_natural_search_yaw_sweep_deg)
            yaw_offsets = [0.0, yaw_rad, -yaw_rad]
            expanded_wps = []
            expanded_qs = []
            for wp, q_nom in zip(waypoints, quats):
                r_nom = ScipyR.from_quat(q_nom)
                for yaw in yaw_offsets:
                    # Rotate search view around world Z to scan laterally.
                    r_scan = ScipyR.from_euler("z", yaw) * r_nom
                    expanded_wps.append(wp.copy())
                    expanded_qs.append(r_scan.as_quat())
            waypoints = expanded_wps
            quats = expanded_qs

        self.pre_nat_search_started = True
        self.pre_nat_search_waypoints = waypoints
        self.pre_nat_search_quats = quats
        self.pre_nat_search_idx = 0
        self.pre_nat_search_waypoint_start_sec = self._now_sec()
        self.pre_nat_search_orientation = curr_q.copy()
        self.get_logger().info(
            f"pre-natural search started with {len(waypoints)} waypoints "
            f"(orbit={orbit_added})"
        )

    def _run_search_step(
        self,
        curr_pos: np.ndarray,
        curr_q: np.ndarray,
        enable_rotation_sweep: bool,
    ):
        if (
            not self.pre_nat_search_started
            or self.pre_nat_search_orientation is None
            or len(self.pre_nat_search_waypoints) == 0
        ):
            self._build_pre_natural_search_waypoints(
                curr_pos,
                curr_q,
                enable_rotation_sweep=enable_rotation_sweep,
            )

        if len(self.pre_nat_search_waypoints) == 0:
            return

        idx = self.pre_nat_search_idx % len(self.pre_nat_search_waypoints)
        waypoint = self.pre_nat_search_waypoints[idx]
        if len(self.pre_nat_search_quats) == len(self.pre_nat_search_waypoints):
            q_cmd = self.pre_nat_search_quats[idx]
        else:
            q_cmd = self.pre_nat_search_orientation
        self._publish_target_pose(waypoint, q_cmd)

        if (self._now_sec() - self.pre_nat_search_waypoint_start_sec) >= max(
            self.pre_natural_search_hold_sec, 0.05
        ):
            self.pre_nat_search_idx = (self.pre_nat_search_idx + 1) % len(
                self.pre_nat_search_waypoints
            )
            self.pre_nat_search_waypoint_start_sec = self._now_sec()

    def _camera_pose_from_tool(
        self,
        tool_pos: np.ndarray,
        tool_q: np.ndarray,
        trans_tool_cam,
    ):
        p_tool_cam = np.array(
            [
                trans_tool_cam.transform.translation.x,
                trans_tool_cam.transform.translation.y,
                trans_tool_cam.transform.translation.z,
            ],
            dtype=float,
        )
        q_tool_cam = _quat_from_msg(trans_tool_cam.transform.rotation)
        r_tool = ScipyR.from_quat(tool_q)
        r_tc = ScipyR.from_quat(q_tool_cam)
        cam_pos = r_tool.apply(p_tool_cam) + tool_pos
        cam_q = (r_tool * r_tc).as_quat()
        return cam_pos, cam_q

    def _compute_cam_target_z(
        self,
        tool_pos: np.ndarray,
        tool_q: np.ndarray,
        trans_tool_cam,
        tgt_pos: np.ndarray,
    ):
        cam_pos, cam_q = self._camera_pose_from_tool(tool_pos, tool_q, trans_tool_cam)
        vec_cam_tgt = ScipyR.from_quat(cam_q).inv().apply(tgt_pos - cam_pos)
        return float(vec_cam_tgt[2])

    def _maybe_backoff_if_too_close(
        self,
        curr_pos: np.ndarray,
        curr_q: np.ndarray,
        trans_tool_cam,
        cart_mins: np.ndarray,
        cart_maxs: np.ndarray,
        cam_target_z: float,
    ) -> bool:
        if not self.too_close_backoff_enabled:
            return False
        if not np.isfinite(cam_target_z):
            return False

        threshold = self.desired_dist - max(0.0, self.too_close_distance_margin)
        if cam_target_z >= threshold or cam_target_z <= 0.0:
            return False

        _, cam_q = self._camera_pose_from_tool(curr_pos, curr_q, trans_tool_cam)
        z_cam_base = ScipyR.from_quat(cam_q).as_matrix()[:, 2]
        backoff_step = float(max(1e-4, self.too_close_backoff_step))
        backoff_delta = -backoff_step * z_cam_base
        new_pos = np.clip(curr_pos + backoff_delta, cart_mins, cart_maxs)

        if np.linalg.norm(new_pos - curr_pos) < 1e-5:
            return False

        self._publish_target_pose(new_pos, curr_q)
        self.get_logger().debug(
            f"too-close backoff: cam_z={cam_target_z:.3f}m "
            f"(< {threshold:.3f}m), step={backoff_step:.3f}m"
        )
        return True

    def _publish_target_pose(self, pos: np.ndarray, quat: np.ndarray):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self.pose_pub.publish(msg)

    def _get_workspace_for_side(self, side: str):
        if side == "right":
            return self.right_cart_mins, self.right_cart_maxs
        return self.left_cart_mins, self.left_cart_maxs

    def _select_workspace(self, curr_pos: np.ndarray, tgt_pos: np.ndarray):
        prev_side = self.active_workspace_side
        if self.workspace_side == "left":
            self.active_workspace_side = "left"
        elif self.workspace_side == "right":
            self.active_workspace_side = "right"
        else:
            y_curr = float(curr_pos[1])
            y_tgt = float(tgt_pos[1]) if tgt_pos is not None else y_curr
            left_min, left_max = self.left_cart_mins[1], self.left_cart_maxs[1]
            right_min, right_max = self.right_cart_mins[1], self.right_cart_maxs[1]

            # Keep current side while within side range plus hysteresis.
            if self.active_workspace_side == "left":
                if y_curr > (left_max + self.workspace_side_hys):
                    self.active_workspace_side = "right"
            elif self.active_workspace_side == "right":
                if y_curr < (right_min - self.workspace_side_hys):
                    self.active_workspace_side = "left"
            else:
                self.active_workspace_side = "left"

            # If current y is clearly in one side, use it.
            if left_min <= y_curr <= left_max:
                self.active_workspace_side = "left"
            elif right_min <= y_curr <= right_max:
                self.active_workspace_side = "right"
            else:
                # Otherwise pick the side closest to target y.
                left_center = 0.5 * (left_min + left_max)
                right_center = 0.5 * (right_min + right_max)
                if abs(y_tgt - right_center) < abs(y_tgt - left_center):
                    self.active_workspace_side = "right"
                else:
                    self.active_workspace_side = "left"

        if self.active_workspace_side != prev_side:
            self.get_logger().info(f"workspace side -> {self.active_workspace_side}")

        return self._get_workspace_for_side(self.active_workspace_side)

    def _get_tf_target_state(self):
        try:
            trans = self.tf_buffer.lookup_transform(self.base_frame, self.target_frame, Time())
        except Exception:
            return None

        pos = np.array(
            [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ],
            dtype=float,
        )
        quat = np.array(
            [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ],
            dtype=float,
        )
        stamp_sec = _stamp_to_sec(trans.header.stamp)
        if stamp_sec <= 0.0:
            stamp_sec = self._now_sec()

        vel = np.zeros(3, dtype=float)
        omega = np.zeros(3, dtype=float)
        if self._last_tf_pose is not None:
            p_prev, q_prev, t_prev = self._last_tf_pose
            if stamp_sec > t_prev:
                dt = max(stamp_sec - t_prev, 1e-6)
                vel = (pos - p_prev) / dt
                r_delta = ScipyR.from_quat(quat) * ScipyR.from_quat(q_prev).inv()
                omega = r_delta.as_rotvec() / dt
        self._last_tf_pose = (pos.copy(), quat.copy(), stamp_sec)

        if (
            self.last_vo_cov is not None
            and self.last_vo_cov_sec is not None
            and (self._now_sec() - self.last_vo_cov_sec) < 1.0
        ):
            cov = self.last_vo_cov.copy()
        else:
            cov = np.eye(6, dtype=float) * 1e-3

        age = self._now_sec() - stamp_sec
        return pos, quat, cov, vel, omega, age

    def _get_active_target_state(self):
        now_sec = self._now_sec()

        # Prefer the more stable fused target state for camera optimization.
        # Priority tags still contribute visibility proxy points and can take
        # over only when fused tracking drops out.
        if self.tgt_pos is not None and self.tgt_q is not None and self.last_target_update_sec is not None:
            age = now_sec - self.last_target_update_sec
            if age <= self.external_target_timeout_sec:
                return (
                    self.tgt_pos.copy(),
                    self.tgt_q.copy(),
                    self.tgt_cov.copy(),
                    self.tgt_vel.copy(),
                    self.tgt_omega.copy(),
                    age,
                )

        if (
            self.use_priority_tag_tracking
            and self.tag_tgt_pos is not None
            and self.tag_tgt_q is not None
            and self.last_tag_update_sec is not None
        ):
            age = now_sec - self.last_tag_update_sec
            if age <= self.tracking_tag_timeout_sec:
                return (
                    self.tag_tgt_pos.copy(),
                    self.tag_tgt_q.copy(),
                    self.tag_tgt_cov.copy(),
                    self.tag_tgt_vel.copy(),
                    self.tag_tgt_omega.copy(),
                    age,
                )

        if self.use_tf_target_fallback:
            return self._get_tf_target_state()

        return None

    def tick(self):
        if self.mode == 4:
            seen = self._publish_pre_natural_seen()

            try:
                trans_tool = self.tf_buffer.lookup_transform(
                    self.base_frame, self.tool_frame, Time()
                )
                trans_tool_cam = self.tf_buffer.lookup_transform(
                    self.tool_frame, self.camera_frame, Time()
                )
            except Exception:
                return

            curr_pos = np.array(
                [
                    trans_tool.transform.translation.x,
                    trans_tool.transform.translation.y,
                    trans_tool.transform.translation.z,
                ],
                dtype=float,
            )
            curr_q = _quat_from_msg(trans_tool.transform.rotation)
            target_state = self._get_active_target_state()
            tgt_pos_hint = target_state[0] if target_state is not None else curr_pos
            cart_mins, cart_maxs = self._select_workspace(curr_pos, tgt_pos_hint)

            cam_target_z = None
            if target_state is not None and target_state[5] <= self.pre_natural_stale_sec:
                try:
                    cam_target_z = self._compute_cam_target_z(
                        curr_pos, curr_q, trans_tool_cam, target_state[0]
                    )
                except Exception:
                    cam_target_z = None
            if cam_target_z is None and self.last_fused_cam_z is not None:
                age = (
                    self._now_sec() - self.last_fused_cam_z_sec
                    if self.last_fused_cam_z_sec is not None
                    else np.inf
                )
                if age < self.pre_natural_stale_sec:
                    cam_target_z = float(self.last_fused_cam_z)

            if (
                cam_target_z is not None
                and self._maybe_backoff_if_too_close(
                    curr_pos,
                    curr_q,
                    trans_tool_cam,
                    cart_mins,
                    cart_maxs,
                    cam_target_z,
                )
            ):
                return

            if not self.pre_natural_search_enabled or seen:
                return

            self._run_search_step(
                curr_pos=curr_pos,
                curr_q=curr_q,
                enable_rotation_sweep=True,
            )
            return

        if self.mode != 3:
            return

        target_state = self._get_active_target_state()
        if target_state is None:
            self._publish_target_visible(False)
            if self.natural_search_on_lost_target and self.pre_natural_search_enabled:
                try:
                    trans_tool = self.tf_buffer.lookup_transform(
                        self.base_frame, self.tool_frame, Time()
                    )
                except Exception:
                    return
                curr_pos = np.array(
                    [
                        trans_tool.transform.translation.x,
                        trans_tool.transform.translation.y,
                        trans_tool.transform.translation.z,
                    ],
                    dtype=float,
                )
                curr_q = _quat_from_msg(trans_tool.transform.rotation)
                self._run_search_step(
                    curr_pos=curr_pos,
                    curr_q=curr_q,
                    enable_rotation_sweep=True,
                )
            return

        tgt_pos, tgt_q, tgt_cov, tgt_vel, tgt_omega, stale_age = target_state
        if stale_age > self.hard_stale_sec:
            self._publish_target_visible(False)
            self.get_logger().warn(
                f"target stale for {stale_age:.2f}s; skipping command",
                throttle_duration_sec=1.0,
            )
            if self.natural_search_on_lost_target and self.pre_natural_search_enabled:
                try:
                    trans_tool = self.tf_buffer.lookup_transform(
                        self.base_frame, self.tool_frame, Time()
                    )
                except Exception:
                    return
                curr_pos = np.array(
                    [
                        trans_tool.transform.translation.x,
                        trans_tool.transform.translation.y,
                        trans_tool.transform.translation.z,
                    ],
                    dtype=float,
                )
                curr_q = _quat_from_msg(trans_tool.transform.rotation)
                self._run_search_step(
                    curr_pos=curr_pos,
                    curr_q=curr_q,
                    enable_rotation_sweep=True,
                )
            return
        self._publish_target_visible(True)
        if self.pre_nat_search_started:
            self.pre_nat_search_started = False
            self.pre_nat_search_waypoints = []
            self.pre_nat_search_quats = []
            self.pre_nat_search_idx = 0
            self.pre_nat_search_waypoint_start_sec = 0.0
            self.pre_nat_search_orientation = None

        try:
            trans_tool = self.tf_buffer.lookup_transform(
                self.base_frame, self.tool_frame, Time()
            )
            trans_tool_cam = self.tf_buffer.lookup_transform(
                self.tool_frame, self.camera_frame, Time()
            )
        except Exception:
            return

        curr_pos = np.array(
            [
                trans_tool.transform.translation.x,
                trans_tool.transform.translation.y,
                trans_tool.transform.translation.z,
            ],
            dtype=float,
        )
        curr_q = _quat_from_msg(trans_tool.transform.rotation)
        curr_rotvec = ScipyR.from_quat(curr_q).as_rotvec()

        if self.start_mode3_q is None:
            self.start_mode3_q = curr_q.copy()
        if self.start_mode3_pos is None:
            self.start_mode3_pos = curr_pos.copy()

        cart_mins, cart_maxs = self._select_workspace(curr_pos, tgt_pos)
        cam_target_z = self._compute_cam_target_z(curr_pos, curr_q, trans_tool_cam, tgt_pos)
        if self._maybe_backoff_if_too_close(
            curr_pos,
            curr_q,
            trans_tool_cam,
            cart_mins,
            cart_maxs,
            cam_target_z,
        ):
            self.prev_prev_cmd[:] = 0.0
            self.prev_cmd[:] = 0.0
            return

        tgt_points = self._get_target_proxy_points(tgt_pos, tgt_q)

        new_pos, new_q, x_opt = self.optimize_pose(
            curr_pos=curr_pos,
            curr_rotvec=curr_rotvec,
            trans_tool_cam=trans_tool_cam,
            cart_mins=cart_mins,
            cart_maxs=cart_maxs,
            tgt_points=tgt_points,
            tgt_pos=tgt_pos,
            tgt_q=tgt_q,
            tgt_vel=tgt_vel,
            tgt_omega=tgt_omega,
            tgt_cov=tgt_cov,
            stale_age=stale_age,
        )

        self._publish_target_pose(new_pos, new_q)

        self.prev_prev_cmd = self.prev_cmd.copy()
        self.prev_cmd = x_opt.copy()

    def optimize_pose(
        self,
        curr_pos: np.ndarray,
        curr_rotvec: np.ndarray,
        trans_tool_cam,
        cart_mins: np.ndarray,
        cart_maxs: np.ndarray,
        tgt_points: np.ndarray,
        tgt_pos: np.ndarray,
        tgt_q: np.ndarray,
        tgt_vel: np.ndarray,
        tgt_omega: np.ndarray,
        tgt_cov: np.ndarray,
        stale_age: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bounds = [
            (
                max(-self.delta_pos_max, cart_mins[0] - curr_pos[0]),
                min(self.delta_pos_max, cart_maxs[0] - curr_pos[0]),
            ),
            (
                max(-self.delta_pos_max, cart_mins[1] - curr_pos[1]),
                min(self.delta_pos_max, cart_maxs[1] - curr_pos[1]),
            ),
            (
                max(-self.delta_pos_max, cart_mins[2] - curr_pos[2]),
                min(self.delta_pos_max, cart_maxs[2] - curr_pos[2]),
            ),
            (-self.delta_rot_max, self.delta_rot_max),
            (-self.delta_rot_max, self.delta_rot_max),
            (-self.delta_rot_max, self.delta_rot_max),
        ]

        # If currently outside workspace by more than one step, force correction.
        for axis in range(3):
            if bounds[axis][0] > bounds[axis][1]:
                target_delta = np.clip(
                    np.clip(curr_pos[axis], cart_mins[axis], cart_maxs[axis]) - curr_pos[axis],
                    -self.delta_pos_max,
                    self.delta_pos_max,
                )
                bounds[axis] = (target_delta, target_delta)

        predict_dt = self.vision_delay_sec + self.motor_delay_sec + max(stale_age, 0.0)
        x0 = np.zeros(6, dtype=float)
        nominal_tool_pos, nominal_pos_span = self._get_nominal_tool_pos(cart_mins, cart_maxs)

        try:
            res = minimize(
                _camera_pose_residual,
                x0,
                method="SLSQP",
                bounds=bounds,
                args=(
                    curr_pos,
                    curr_rotvec,
                    tgt_pos,
                    tgt_q,
                    tgt_vel,
                    tgt_omega,
                    tgt_cov,
                    trans_tool_cam,
                    tgt_points,
                    self.fx,
                    self.fy,
                    self.cx,
                    self.cy,
                    self.img_w,
                    self.img_h,
                    self.safe_margin_px,
                    self.prev_cmd,
                    self.prev_prev_cmd,
                    self.ctrl_dt,
                    predict_dt,
                    self.distance_objective_mode,
                    self.desired_dist,
                    self.desired_view_cos,
                    self.delta_pos_max,
                    self.delta_rot_max,
                    nominal_tool_pos,
                    nominal_pos_span,
                    self.nominal_axis_weights,
                    self.rotation_penalty_scale,
                    self.preferred_tool_view_axis,
                    self.min_safe_view_cos,
                ),
                options={"disp": False, "maxiter": self.maxiter, "ftol": 1e-4},
            )
            x_opt = res.x if res.success else x0
        except Exception:
            x_opt = x0

        new_pos = np.clip(curr_pos + x_opt[:3], cart_mins, cart_maxs)
        r_new = ScipyR.from_rotvec(curr_rotvec + x_opt[3:6])

        if self.start_mode3_q is not None and self.max_angular_deviation > 0.0:
            r_start = ScipyR.from_quat(self.start_mode3_q)
            r_delta = r_new * r_start.inv()
            delta_vec = r_delta.as_rotvec()
            delta_norm = np.linalg.norm(delta_vec)
            if delta_norm > self.max_angular_deviation:
                delta_vec = delta_vec * (self.max_angular_deviation / max(delta_norm, 1e-9))
                r_new = ScipyR.from_rotvec(delta_vec) * r_start

        new_q = r_new.as_quat()
        return new_pos, new_q, x_opt


def main(args=None):
    rclpy.init(args=args)
    node = NaturalHandler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
