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
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as ScipyR
from std_msgs.msg import Bool, Int32
from tf2_ros import Buffer, TransformListener


def _softplus(x: np.ndarray) -> np.ndarray:
    x_clamped = np.clip(x, -50.0, 50.0)
    return np.log1p(np.exp(x_clamped))


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _quat_from_msg(qmsg) -> np.ndarray:
    return np.array([qmsg.x, qmsg.y, qmsg.z, qmsg.w], dtype=float)


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
    desired_dist: float,
    desired_view_cos: float,
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
    w_view = 20.0 / (1.0 + 40.0 * unc)
    w_vis = 60.0 * (1.0 + 20.0 * unc)
    w_smooth = 4.0 * (1.0 + 40.0 * unc)
    w_step = 0.8 * (1.0 + 40.0 * unc)

    vec_ct = tgt_pos_d - cam_pos
    dist = np.linalg.norm(vec_ct) + 1e-9
    loss_track = (dist - desired_dist) ** 2

    z_cam = ScipyR.from_quat(cam_q).as_matrix()[:, 2]
    loss_align = (1.0 - float(np.dot(vec_ct / dist, z_cam))) ** 2

    z_tgt = ScipyR.from_quat(tgt_q_d).as_matrix()[:, 2]
    vec_tc = -vec_ct / dist
    loss_view = (float(np.dot(vec_tc, z_tgt)) - desired_view_cos) ** 2

    uv, valid = _project_target_points(
        cam_pos, cam_q, tgt_pos_d, tgt_q_d, tgt_points, fx, fy, cx, cy
    )
    if (not valid) or uv.shape[0] == 0:
        loss_vis = 1e3
    else:
        margins = np.minimum.reduce(
            [
                uv[:, 0],
                (img_w - 1.0) - uv[:, 0],
                uv[:, 1],
                (img_h - 1.0) - uv[:, 1],
            ]
        )
        loss_vis = float(np.sum(_softplus((safe_margin_px - margins) / 6.0)))

    cmd_rate = (x - prev_cmd) / max(dt_ctrl, 1e-6)
    cmd_jerk = (x - 2.0 * prev_cmd + prev_prev_cmd) / max(dt_ctrl * dt_ctrl, 1e-6)
    loss_smooth = float(np.dot(cmd_rate, cmd_rate) + 0.1 * np.dot(cmd_jerk, cmd_jerk))
    loss_step = float(np.dot(x, x))

    return (
        w_track * loss_track
        + w_align * loss_align
        + w_view * loss_view
        + w_vis * loss_vis
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
        self.declare_parameter("desired_distance", 0.35)
        self.declare_parameter("desired_view_cos", 0.7071)

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

        # TF fallback state
        self._last_tf_pose = None  # (pos, quat, sec)
        self.last_vo_cov = None
        self.last_vo_cov_sec = None

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

        self.timer = self.create_timer(self.ctrl_dt, self.tick)

        self.get_logger().info("NaturalHandler (Option B) active")
        self.get_logger().info(f"mode topic: {self.mode_topic}")
        self.get_logger().info(f"target pose topic: {self.target_pose_topic}")
        self.get_logger().info(
            f"fused topics: pose={self.fused_pose_topic}, twist={self.fused_twist_topic}, odom={self.fused_odom_topic}"
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
        elif self.mode != 3:
            self.start_mode3_q = None

        if self.mode not in (3, 4):
            self.odom_seen_pub.publish(Bool(data=False))

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

        # If odometry already comes in base frame, accept it directly.
        if msg.header.frame_id == self.base_frame:
            pos = np.array(
                [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
                dtype=float,
            )
            quat = _quat_from_msg(msg.pose.pose.orientation)
            self._update_target_from_pose(pos, quat, cov, _stamp_to_sec(msg.header.stamp))
            self.tgt_vel = np.array(
                [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
                dtype=float,
            )
            self.tgt_omega = np.array(
                [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z],
                dtype=float,
            )
            self.last_twist_update_sec = self._now_sec()

    def _publish_pre_natural_seen(self):
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
        except Exception:
            self.odom_seen_pub.publish(Bool(data=False))

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

        # Prefer external fused state when fresh.
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

        if self.use_tf_target_fallback:
            return self._get_tf_target_state()

        return None

    def tick(self):
        if self.mode == 4:
            self._publish_pre_natural_seen()
            return

        if self.mode != 3:
            return

        target_state = self._get_active_target_state()
        if target_state is None:
            return

        tgt_pos, tgt_q, tgt_cov, tgt_vel, tgt_omega, stale_age = target_state
        if stale_age > self.hard_stale_sec:
            self.get_logger().warn(
                f"target stale for {stale_age:.2f}s; skipping command",
                throttle_duration_sec=1.0,
            )
            return

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

        cart_mins, cart_maxs = self._select_workspace(curr_pos, tgt_pos)

        new_pos, new_q, x_opt = self.optimize_pose(
            curr_pos=curr_pos,
            curr_rotvec=curr_rotvec,
            trans_tool_cam=trans_tool_cam,
            cart_mins=cart_mins,
            cart_maxs=cart_maxs,
            tgt_pos=tgt_pos,
            tgt_q=tgt_q,
            tgt_vel=tgt_vel,
            tgt_omega=tgt_omega,
            tgt_cov=tgt_cov,
            stale_age=stale_age,
        )

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x = float(new_pos[0])
        msg.pose.position.y = float(new_pos[1])
        msg.pose.position.z = float(new_pos[2])
        msg.pose.orientation.x = float(new_q[0])
        msg.pose.orientation.y = float(new_q[1])
        msg.pose.orientation.z = float(new_q[2])
        msg.pose.orientation.w = float(new_q[3])
        self.pose_pub.publish(msg)

        self.prev_prev_cmd = self.prev_cmd.copy()
        self.prev_cmd = x_opt.copy()

    def optimize_pose(
        self,
        curr_pos: np.ndarray,
        curr_rotvec: np.ndarray,
        trans_tool_cam,
        cart_mins: np.ndarray,
        cart_maxs: np.ndarray,
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
                    self.tgt_points,
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
                    self.desired_dist,
                    self.desired_view_cos,
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
