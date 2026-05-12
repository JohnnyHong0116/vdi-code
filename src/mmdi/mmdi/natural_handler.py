#!/usr/bin/env python3
"""Natural demonstration camera tracker.

This node keeps the camera-to-arm transform from TF, records the arm pose when
mode 3 starts, and optimizes the tool pose so the camera points at the tracked
tool while staying inside global tool workspace bounds:

    (normalize(p_fused_tool - p_camera) dot z_camera - 1)^2

It can also bias the camera roll so the fused tool +Y direction appears along
camera +Y after projection into the image/look plane.

The command is constrained by global position bounds plus per-tick position and
rotation limits. Tool orientation is constrained to a detach-relative Euler
region and maximum total rotation so camera wiring is not driven through
excessive twists.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as ScipyR
from std_msgs.msg import Bool, Int32
from tf2_ros import Buffer, TransformListener

NATURAL_MODE = 3


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _quat_from_msg(qmsg) -> np.ndarray:
    return np.array([qmsg.x, qmsg.y, qmsg.z, qmsg.w], dtype=float)


def _normalize(vec: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if (not np.isfinite(norm)) or norm <= eps:
        return None
    return vec / norm


def _tool_camera_transform(trans_tool_cam) -> Tuple[np.ndarray, ScipyR]:
    p_tool_cam = np.array(
        [
            trans_tool_cam.transform.translation.x,
            trans_tool_cam.transform.translation.y,
            trans_tool_cam.transform.translation.z,
        ],
        dtype=float,
    )
    q_tool_cam = _quat_from_msg(trans_tool_cam.transform.rotation)
    return p_tool_cam, ScipyR.from_quat(q_tool_cam)


def _camera_pose_from_tool(
    tool_pos: np.ndarray,
    tool_rotvec: np.ndarray,
    trans_tool_cam,
) -> Tuple[np.ndarray, ScipyR]:
    return _camera_pose_from_tool_rotation(
        tool_pos,
        ScipyR.from_rotvec(tool_rotvec),
        trans_tool_cam,
    )


def _camera_pose_from_tool_rotation(
    tool_pos: np.ndarray,
    r_tool: ScipyR,
    trans_tool_cam,
) -> Tuple[np.ndarray, ScipyR]:
    p_tool_cam, r_tool_cam = _tool_camera_transform(trans_tool_cam)
    cam_pos = r_tool.apply(p_tool_cam) + tool_pos
    cam_rot = r_tool * r_tool_cam
    return cam_pos, cam_rot


def camera_z_alignment_residual(
    x: np.ndarray,
    curr_pos: np.ndarray,
    curr_rotvec: np.ndarray,
    target_pos: np.ndarray,
    trans_tool_cam,
) -> float:
    tool_pos = np.asarray(curr_pos, dtype=float) + np.asarray(x[:3], dtype=float)
    tool_rot = ScipyR.from_rotvec(x[3:6]) * ScipyR.from_rotvec(curr_rotvec)

    cam_pos, cam_rot = _camera_pose_from_tool_rotation(
        tool_pos,
        tool_rot,
        trans_tool_cam,
    )

    cam_to_target = _normalize(target_pos - cam_pos)
    if cam_to_target is None:
        return 1e3

    z_cam = cam_rot.as_matrix()[:, 2]
    return float((np.dot(cam_to_target, z_cam) - 1.0) ** 2)


def camera_fused_y_alignment_residual(
    cam_to_target: np.ndarray,
    cam_rot: ScipyR,
    target_quat: np.ndarray,
) -> float:
    r_target = ScipyR.from_quat(target_quat)
    fused_y = r_target.as_matrix()[:, 1]
    fused_y_proj = _normalize(
        fused_y - np.dot(fused_y, cam_to_target) * cam_to_target
    )
    if fused_y_proj is None:
        return 0.0

    y_cam = cam_rot.as_matrix()[:, 1]
    return float((np.dot(fused_y_proj, y_cam) - 1.0) ** 2)


def camera_alignment_residual(
    x: np.ndarray,
    curr_pos: np.ndarray,
    curr_rotvec: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    trans_tool_cam,
    look_weight: float,
    fused_y_weight: float,
    camera_distance_min_m: float,
    camera_distance_max_m: float,
    distance_weight: float,
    preferred_side_xy: Optional[np.ndarray],
    side_offset_m: float,
    side_weight: float,
    desired_tool_z_m: float,
    height_weight: float,
    motion_weight: float,
    rotation_motion_weight: float,
) -> float:
    tool_pos = np.asarray(curr_pos, dtype=float) + np.asarray(x[:3], dtype=float)
    tool_rot = ScipyR.from_rotvec(x[3:6]) * ScipyR.from_rotvec(curr_rotvec)
    cam_pos, cam_rot = _camera_pose_from_tool_rotation(
        tool_pos,
        tool_rot,
        trans_tool_cam,
    )

    cam_to_target = _normalize(target_pos - cam_pos)
    if cam_to_target is None:
        return 1e3

    z_cam = cam_rot.as_matrix()[:, 2]
    look_residual = float((np.dot(cam_to_target, z_cam) - 1.0) ** 2)

    dist_residual = 0.0
    if distance_weight > 0.0:
        dist = float(np.linalg.norm(target_pos - cam_pos))
        if dist < camera_distance_min_m:
            dist_residual = float((camera_distance_min_m - dist) ** 2)
        elif dist > camera_distance_max_m:
            dist_residual = float((dist - camera_distance_max_m) ** 2)

    side_residual = 0.0
    if (
        side_weight > 0.0
        and preferred_side_xy is not None
        and side_offset_m > 0.0
    ):
        side_target = np.asarray(target_pos, dtype=float).copy()
        side_target[:2] += side_offset_m * preferred_side_xy[:2]
        side_residual = float(np.linalg.norm(tool_pos[:2] - side_target[:2]) ** 2)

    height_residual = 0.0
    if height_weight > 0.0:
        height_residual = float((tool_pos[2] - desired_tool_z_m) ** 2)

    roll_residual = 0.0
    if fused_y_weight > 0.0:
        roll_residual = camera_fused_y_alignment_residual(
            cam_to_target,
            cam_rot,
            target_quat,
        )

    motion_residual = 0.0
    if motion_weight > 0.0:
        motion_residual = float(
            np.linalg.norm(x[:3]) ** 2
            + rotation_motion_weight * np.linalg.norm(x[3:6]) ** 2
        )

    return float(
        look_weight * look_residual
        + distance_weight * dist_residual
        + side_weight * side_residual
        + height_weight * height_residual
        + fused_y_weight * roll_residual
        + motion_weight * motion_residual
    )


class NaturalHandler(Node):
    def __init__(self):
        super().__init__("natural_handler")

        self.declare_parameter("mode_topic", "/mode")
        self.declare_parameter("target_pose_topic", "/ur7e/target_pose")
        self.declare_parameter(
            "optimizing_active_topic",
            "/natural_handler/optimizing_active",
        )
        self.declare_parameter("target_lost_topic", "/natural_handler/target_lost")
        self.declare_parameter("target_visible_topic", "/probe_tracker/target_visible")
        self.declare_parameter("fused_pose_topic", "/ur7e/fused_tool_pose")
        self.declare_parameter("fused_odom_topic", "/vo")

        self.declare_parameter("base_frame", "base")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("camera_frame", "head_camera")
        self.declare_parameter("target_frame", "odom")

        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("target_timeout_sec", 0.5)
        self.declare_parameter("hard_stale_sec", 1.0)
        self.declare_parameter("use_tf_target_fallback", True)

        self.declare_parameter("delta_pos_max_m", 0.015)
        self.declare_parameter("internal_bound_radius_m", 0.03)
        self.declare_parameter("delta_rot_max", 0.35)
        self.declare_parameter("max_relative_euler_deg", [25.0, 25.0, 45.0])
        self.declare_parameter("max_relative_rotation_deg", 45.0)
        self.declare_parameter("global_tool_pos_min", [0.09, -0.43, 0.24])
        self.declare_parameter("global_tool_pos_max", [0.64, 0.44, 0.49])
        self.declare_parameter("maxiter", 60)
        self.declare_parameter("look_alignment_weight", 100.0)
        self.declare_parameter("camera_distance_min_m", 0.25)
        self.declare_parameter("camera_distance_max_m", 0.35)
        self.declare_parameter("camera_distance_weight", 20.0)
        self.declare_parameter("side_offset_m", 0.20)
        self.declare_parameter("side_preference_weight", 10.0)
        self.declare_parameter("side_axis_sign", -1.0)
        self.declare_parameter("side_deadband", 0.15)
        self.declare_parameter("desired_tool_z_m", 0.35)
        self.declare_parameter("tool_height_weight", 2.0)
        self.declare_parameter("fused_y_alignment_weight", 0.1)
        self.declare_parameter("motion_weight", 1.0)
        self.declare_parameter("rotation_motion_weight", 0.1)
        self.declare_parameter("command_translation_deadband_m", 0.004)
        self.declare_parameter("command_rotation_deadband_rad", 0.035)
        self.declare_parameter("search_yaw_amplitude_rad", 0.35)
        self.declare_parameter("search_yaw_frequency_hz", 0.25)
        self.declare_parameter("search_yaw_axis_sign", 1.0)
        self.declare_parameter("disable_near_attached_target", True)
        self.declare_parameter("attached_target_camera_pos", [0.029, -0.025, 0.159])
        self.declare_parameter("attached_target_camera_radius_m", 0.04)

        self.mode_topic = str(self.get_parameter("mode_topic").value)
        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self.optimizing_active_topic = str(
            self.get_parameter("optimizing_active_topic").value
        )
        self.target_lost_topic = str(self.get_parameter("target_lost_topic").value)
        self.target_visible_topic = str(
            self.get_parameter("target_visible_topic").value
        )
        self.fused_pose_topic = str(self.get_parameter("fused_pose_topic").value)
        self.fused_odom_topic = str(self.get_parameter("fused_odom_topic").value)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tool_frame = str(self.get_parameter("tool_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.target_frame = str(self.get_parameter("target_frame").value)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.target_timeout_sec = float(
            self.get_parameter("target_timeout_sec").value
        )
        self.hard_stale_sec = float(self.get_parameter("hard_stale_sec").value)
        self.use_tf_target_fallback = bool(
            self.get_parameter("use_tf_target_fallback").value
        )

        self.delta_pos_max_m = max(
            0.0,
            float(self.get_parameter("delta_pos_max_m").value),
        )
        self.internal_bound_radius_m = max(
            0.0,
            float(self.get_parameter("internal_bound_radius_m").value),
        )
        self.delta_rot_max = float(self.get_parameter("delta_rot_max").value)
        self.max_relative_euler_rad = self._get_euler_limit_parameter()
        self.max_relative_rotation_rad = np.deg2rad(
            max(0.0, float(self.get_parameter("max_relative_rotation_deg").value))
        )
        self.global_tool_pos_min = self._get_vector_parameter(
            "global_tool_pos_min",
            [0.09, -0.43, 0.24],
        )
        self.global_tool_pos_max = self._get_vector_parameter(
            "global_tool_pos_max",
            [0.64, 0.44, 0.49],
        )
        self.maxiter = int(self.get_parameter("maxiter").value)
        self.look_alignment_weight = max(
            0.0,
            float(self.get_parameter("look_alignment_weight").value),
        )
        self.camera_distance_min_m = max(
            0.0,
            float(self.get_parameter("camera_distance_min_m").value),
        )
        self.camera_distance_max_m = max(
            self.camera_distance_min_m,
            float(self.get_parameter("camera_distance_max_m").value),
        )
        self.camera_distance_weight = max(
            0.0,
            float(self.get_parameter("camera_distance_weight").value),
        )
        self.side_offset_m = max(
            0.0,
            float(self.get_parameter("side_offset_m").value),
        )
        self.side_preference_weight = max(
            0.0,
            float(self.get_parameter("side_preference_weight").value),
        )
        self.side_axis_sign = 1.0 if float(
            self.get_parameter("side_axis_sign").value
        ) >= 0.0 else -1.0
        self.side_deadband = max(
            0.0,
            float(self.get_parameter("side_deadband").value),
        )
        self.desired_tool_z_m = float(self.get_parameter("desired_tool_z_m").value)
        self.tool_height_weight = max(
            0.0,
            float(self.get_parameter("tool_height_weight").value),
        )
        self.fused_y_alignment_weight = float(
            self.get_parameter("fused_y_alignment_weight").value
        )
        self.motion_weight = max(
            0.0,
            float(self.get_parameter("motion_weight").value),
        )
        self.rotation_motion_weight = max(
            0.0,
            float(self.get_parameter("rotation_motion_weight").value),
        )
        self.command_translation_deadband_m = max(
            0.0,
            float(self.get_parameter("command_translation_deadband_m").value),
        )
        self.command_rotation_deadband_rad = max(
            0.0,
            float(self.get_parameter("command_rotation_deadband_rad").value),
        )
        self.search_yaw_amplitude_rad = max(
            0.0,
            float(self.get_parameter("search_yaw_amplitude_rad").value),
        )
        self.search_yaw_frequency_hz = max(
            0.0,
            float(self.get_parameter("search_yaw_frequency_hz").value),
        )
        self.search_yaw_axis_sign = 1.0 if float(
            self.get_parameter("search_yaw_axis_sign").value
        ) >= 0.0 else -1.0
        self.disable_near_attached_target = bool(
            self.get_parameter("disable_near_attached_target").value
        )
        self.attached_target_camera_pos = np.asarray(
            self.get_parameter("attached_target_camera_pos").value,
            dtype=float,
        ).reshape(-1)
        if self.attached_target_camera_pos.size != 3:
            self.get_logger().warn(
                "attached_target_camera_pos must have 3 values; using default"
            )
            self.attached_target_camera_pos = np.array(
                [0.029, -0.025, 0.159],
                dtype=float,
            )
        self.attached_target_camera_radius_m = float(
            self.get_parameter("attached_target_camera_radius_m").value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.mode = 0
        self.detach_pos = None
        self.detach_rotvec = None
        self.detach_camera_pos = None
        self.preferred_side_xy = None
        self.last_side_log_sec = None
        self.last_command_pos = None
        self.last_command_quat = None
        self.search_start_sec = None
        self.search_center_pos = None
        self.search_center_rotvec = None
        self.search_initial_direction = 1.0

        self.target_pos = None
        self.target_quat = None
        self.last_target_update_sec = None
        self.target_visible = False
        self.last_target_visible_sec = None
        self.target_camera_pos = None
        self.last_target_camera_update_sec = None

        self.pose_pub = self.create_publisher(PoseStamped, self.target_pose_topic, 1)
        self.optimizing_pub = self.create_publisher(
            Bool,
            self.optimizing_active_topic,
            10,
        )
        self.target_lost_pub = self.create_publisher(
            Bool,
            self.target_lost_topic,
            10,
        )

        self.create_subscription(Int32, self.mode_topic, self.on_mode, 10)
        self.create_subscription(Bool, self.target_visible_topic, self.on_target_visible, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            self.fused_pose_topic,
            self.on_fused_pose,
            10,
        )
        self.create_subscription(Odometry, self.fused_odom_topic, self.on_fused_odom, 10)

        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1e-6), self.tick)

        self.get_logger().info("NaturalHandler camera optimizer active")
        self.get_logger().info(
            f"frames: base={self.base_frame}, tool={self.tool_frame}, "
            f"camera={self.camera_frame}, target={self.target_frame}"
        )
        self.get_logger().info(
            f"objective: align camera +z to fused tool, "
            f"bias global side using signed fused +y, "
            f"bias camera +y to projected fused +y; "
            f"look_weight={self.look_alignment_weight:.3f}, "
            f"distance_band=[{self.camera_distance_min_m:.3f}, "
            f"{self.camera_distance_max_m:.3f}]m, "
            f"distance_weight={self.camera_distance_weight:.3f}, "
            f"side_weight={self.side_preference_weight:.3f}, "
            f"height_weight={self.tool_height_weight:.3f}, "
            f"fused_y_weight={self.fused_y_alignment_weight:.3f}, "
            f"delta_pos={self.delta_pos_max_m:.3f}m, "
            f"command_deadband={self.command_translation_deadband_m:.3f}m/"
            f"{self.command_rotation_deadband_rad:.3f}rad, "
            f"search_yaw={self.search_yaw_amplitude_rad:.3f}rad@"
            f"{self.search_yaw_frequency_hz:.2f}Hz, "
            f"internal_radius={self.internal_bound_radius_m:.3f}m, "
            f"delta_rot={self.delta_rot_max:.3f}rad, "
            f"relative_euler_limit="
            f"{np.round(np.rad2deg(self.max_relative_euler_rad), 1).tolist()}deg, "
            f"relative_rotation_limit="
            f"{np.rad2deg(self.max_relative_rotation_rad):.1f}deg"
        )
        self.get_logger().info(
            f"side preference: offset={self.side_offset_m:.3f}m, "
            f"axis_sign={self.side_axis_sign:.0f}, "
            f"deadband={self.side_deadband:.3f}, "
            f"desired_z={self.desired_tool_z_m:.3f}m"
        )
        self.get_logger().info(
            f"global tool bounds: "
            f"{np.round(self.global_tool_pos_min, 3).tolist()} -> "
            f"{np.round(self.global_tool_pos_max, 3).tolist()}"
        )
        self.get_logger().info(
            f"attached-target guard: enabled={self.disable_near_attached_target}, "
            f"camera_pos={np.round(self.attached_target_camera_pos, 4).tolist()}, "
            f"radius={self.attached_target_camera_radius_m:.3f}m"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _get_euler_limit_parameter(self) -> np.ndarray:
        raw = np.asarray(
            self.get_parameter("max_relative_euler_deg").value,
            dtype=float,
        ).reshape(-1)
        if raw.size != 3:
            self.get_logger().warn(
                "max_relative_euler_deg must have 3 values; using default"
            )
            raw = np.array([25.0, 25.0, 45.0], dtype=float)
        return np.deg2rad(np.maximum(raw, 0.0))

    def _get_vector_parameter(self, name: str, default) -> np.ndarray:
        raw = np.asarray(self.get_parameter(name).value, dtype=float).reshape(-1)
        if raw.size != 3:
            self.get_logger().warn(f"{name} must have 3 values; using default")
            raw = np.asarray(default, dtype=float)
        return raw

    def on_mode(self, msg: Int32):
        prev_mode = self.mode
        self.mode = int(msg.data)

        if prev_mode != NATURAL_MODE and self.mode == NATURAL_MODE:
            self._reset_natural_state()
            self.get_logger().info("natural mode entered; waiting to latch detach pose")
            self._ensure_detach_pose_latched()

        if self.mode != NATURAL_MODE:
            tool_state = self._lookup_tool_pose() if prev_mode == NATURAL_MODE else None
            if tool_state is not None:
                curr_pos, curr_rotvec, _ = tool_state
                curr_quat = ScipyR.from_rotvec(curr_rotvec).as_quat()
                self._publish_target_pose(curr_pos, curr_quat, clamp=False)
            self._publish_optimizing_active(False)
            self._publish_target_lost(False)
            self._reset_natural_state()

    def _reset_natural_state(self):
        self.detach_pos = None
        self.detach_rotvec = None
        self.detach_camera_pos = None
        self.preferred_side_xy = None
        self.last_side_log_sec = None
        self.last_command_pos = None
        self.last_command_quat = None
        self.search_start_sec = None
        self.search_center_pos = None
        self.search_center_rotvec = None
        self.search_initial_direction = 1.0

    def on_target_visible(self, msg: Bool):
        self.target_visible = bool(msg.data)
        self.last_target_visible_sec = self._now_sec()
        if self.target_visible:
            self.search_start_sec = None
            self.search_center_pos = None
            self.search_center_rotvec = None

    def on_fused_pose(self, msg: PoseWithCovarianceStamped):
        frame_id = msg.header.frame_id if msg.header.frame_id else self.base_frame
        pos = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=float,
        )
        quat = _quat_from_msg(msg.pose.pose.orientation)
        stamp_sec = _stamp_to_sec(msg.header.stamp)
        if stamp_sec <= 0.0:
            stamp_sec = self._now_sec()

        if frame_id != self.base_frame:
            transformed = self._transform_pose_to_base(pos, quat, frame_id)
            if transformed is None:
                return
            pos, quat = transformed

        self._update_target(pos, quat, stamp_sec)

    def on_fused_odom(self, msg: Odometry):
        frame_id = msg.header.frame_id if msg.header.frame_id else self.camera_frame
        stamp_sec = _stamp_to_sec(msg.header.stamp)
        if stamp_sec <= 0.0:
            stamp_sec = self._now_sec()

        pos = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=float,
        )
        quat = _quat_from_msg(msg.pose.pose.orientation)

        if frame_id == self.camera_frame:
            self._update_target_camera_pos(pos, stamp_sec)

        if frame_id != self.base_frame:
            transformed = self._transform_pose_to_base(pos, quat, frame_id)
            if transformed is None:
                return
            pos, quat = transformed

        self._update_target(pos, quat, stamp_sec)

    def _update_target(self, pos: np.ndarray, quat: np.ndarray, stamp_sec: float):
        if not np.isfinite(pos).all():
            return
        if (not np.isfinite(quat).all()) or np.linalg.norm(quat) < 1e-9:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        else:
            quat = quat / np.linalg.norm(quat)
        self.target_pos = pos.copy()
        self.target_quat = quat.copy()
        self.last_target_update_sec = stamp_sec

    def _target_visible_active(self) -> bool:
        if not self.target_visible or self.last_target_visible_sec is None:
            return False
        return (self._now_sec() - self.last_target_visible_sec) <= self.target_timeout_sec

    def _update_target_camera_pos(self, pos: np.ndarray, stamp_sec: float):
        if not np.isfinite(pos).all():
            return
        self.target_camera_pos = pos.copy()
        self.last_target_camera_update_sec = stamp_sec

    def _target_near_attached_pose(self) -> bool:
        if not self.disable_near_attached_target:
            return False
        if (
            self.target_camera_pos is None
            or self.last_target_camera_update_sec is None
        ):
            return False

        age = self._now_sec() - self.last_target_camera_update_sec
        if age > self.target_timeout_sec:
            return False

        dist = float(np.linalg.norm(
            self.target_camera_pos - self.attached_target_camera_pos
        ))
        near_attached = dist <= self.attached_target_camera_radius_m
        if near_attached:
            self.get_logger().info(
                f"fused target near attached pose; holding natural command "
                f"({dist:.3f}m <= {self.attached_target_camera_radius_m:.3f}m)",
                throttle_duration_sec=1.0,
            )
        return near_attached

    def _transform_pose_to_base(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        frame_id: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            trans = self.tf_buffer.lookup_transform(self.base_frame, frame_id, Time())
        except Exception:
            return None

        p_bf = np.array(
            [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ],
            dtype=float,
        )
        r_bf = ScipyR.from_quat(_quat_from_msg(trans.transform.rotation))
        pos_base = r_bf.apply(pos) + p_bf
        quat_base = (r_bf * ScipyR.from_quat(quat)).as_quat()
        return pos_base, quat_base

    def _lookup_tool_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        try:
            trans_tool = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tool_frame,
                Time(),
            )
            trans_tool_cam = self.tf_buffer.lookup_transform(
                self.tool_frame,
                self.camera_frame,
                Time(),
            )
        except Exception:
            return None

        pos = np.array(
            [
                trans_tool.transform.translation.x,
                trans_tool.transform.translation.y,
                trans_tool.transform.translation.z,
            ],
            dtype=float,
        )
        quat = _quat_from_msg(trans_tool.transform.rotation)
        rotvec = ScipyR.from_quat(quat).as_rotvec()
        return pos, rotvec, trans_tool_cam

    def _get_active_target_pose(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        now_sec = self._now_sec()
        if (
            self.target_pos is not None
            and self.target_quat is not None
            and self.last_target_update_sec is not None
        ):
            age = now_sec - self.last_target_update_sec
            if age <= self.target_timeout_sec:
                return self.target_pos.copy(), self.target_quat.copy(), age

        if not self.use_tf_target_fallback:
            return None

        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.target_frame,
                Time(),
            )
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
        stamp_sec = _stamp_to_sec(trans.header.stamp)
        if stamp_sec <= 0.0:
            stamp_sec = now_sec
        quat = _quat_from_msg(trans.transform.rotation)
        return pos, quat, now_sec - stamp_sec

    def _latch_detach_pose(
        self,
        curr_pos: np.ndarray,
        curr_rotvec: np.ndarray,
        trans_tool_cam,
    ):
        self.detach_pos = curr_pos.copy()
        self.detach_rotvec = curr_rotvec.copy()
        cam_pos, _ = _camera_pose_from_tool(
            curr_pos,
            curr_rotvec,
            trans_tool_cam,
        )
        self.detach_camera_pos = cam_pos.copy()
        self.get_logger().info(
            "latched detach pose: "
            f"tool={np.round(self.detach_pos, 4).tolist()}, "
            f"camera={np.round(self.detach_camera_pos, 4).tolist()}"
        )

    def _ensure_detach_pose_latched(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        tool_state = self._lookup_tool_pose()
        if tool_state is None:
            return None

        if self.detach_pos is None:
            curr_pos, curr_rotvec, trans_tool_cam = tool_state
            self._latch_detach_pose(curr_pos, curr_rotvec, trans_tool_cam)

        return tool_state

    def _build_bounds(
        self,
        curr_pos: np.ndarray,
        curr_rotvec: np.ndarray,
    ) -> Tuple[np.ndarray, list]:
        bounds = []

        workspace_center = 0.5 * (
            self.global_tool_pos_min + self.global_tool_pos_max
        )
        for axis in range(3):
            low = self.global_tool_pos_min[axis] - curr_pos[axis]
            high = self.global_tool_pos_max[axis] - curr_pos[axis]
            if self.delta_pos_max_m > 1e-9:
                low = max(low, -self.delta_pos_max_m)
                high = min(high, self.delta_pos_max_m)
            if low > high:
                step = workspace_center[axis] - curr_pos[axis]
                if self.delta_pos_max_m > 1e-9:
                    step = np.clip(
                        step,
                        -self.delta_pos_max_m,
                        self.delta_pos_max_m,
                    )
                low = high = float(step)
            bounds.append((float(low), float(high)))

        for axis in range(3):
            bounds.append(
                (
                    float(-self.delta_rot_max),
                    float(self.delta_rot_max),
                )
            )

        x0 = np.zeros(6, dtype=float)
        x0[:3] = self._clamp_to_global_tool_bounds(curr_pos) - curr_pos
        x0[3:6] = self._initial_rotation_step(curr_rotvec)
        for idx, (low, high) in enumerate(bounds):
            x0[idx] = np.clip(x0[idx], low, high)
        return x0, bounds

    def _internal_position_constraint(self, x: np.ndarray) -> float:
        if self.internal_bound_radius_m <= 1e-9:
            return 1.0
        return float(self.internal_bound_radius_m - np.linalg.norm(x[:3]))

    def _preferred_side_from_target(
        self,
        target_quat: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self.side_preference_weight <= 0.0 or self.side_offset_m <= 0.0:
            return None

        r_target = ScipyR.from_quat(target_quat)
        tool_y = r_target.as_matrix()[:, 1]
        tool_y_xy = np.array([tool_y[0], tool_y[1], 0.0], dtype=float)
        norm = float(np.linalg.norm(tool_y_xy[:2]))

        if norm < self.side_deadband:
            if self.preferred_side_xy is None:
                return None
            return self.preferred_side_xy.copy()

        preferred = -self.side_axis_sign * (tool_y_xy / norm)
        if (
            self.preferred_side_xy is not None
            and np.dot(preferred, self.preferred_side_xy) < -0.25
        ):
            self.get_logger().info(
                f"natural side preference flipped: "
                f"{np.round(self.preferred_side_xy[:2], 3).tolist()} -> "
                f"{np.round(preferred[:2], 3).tolist()}",
                throttle_duration_sec=0.5,
            )
        self.preferred_side_xy = preferred
        return preferred.copy()

    def _log_side_preference(
        self,
        preferred_side_xy: Optional[np.ndarray],
        target_pos: np.ndarray,
    ):
        if preferred_side_xy is None:
            return
        now = self._now_sec()
        if (
            self.last_side_log_sec is not None
            and now - self.last_side_log_sec < 1.0
        ):
            return
        self.last_side_log_sec = now
        side_target = np.asarray(target_pos, dtype=float).copy()
        side_target[:2] += self.side_offset_m * preferred_side_xy[:2]
        self.get_logger().info(
            f"natural side target: "
            f"side={np.round(preferred_side_xy[:2], 3).tolist()}, "
            f"target_xy={np.round(side_target[:2], 3).tolist()}"
        )

    def _candidate_tool_rotation(
        self,
        x: np.ndarray,
        curr_rotvec: np.ndarray,
    ) -> ScipyR:
        return ScipyR.from_rotvec(x[3:6]) * ScipyR.from_rotvec(curr_rotvec)

    def _relative_rotation_from_detach(self, tool_rot: ScipyR) -> Optional[ScipyR]:
        if self.detach_rotvec is None:
            return None
        return ScipyR.from_rotvec(self.detach_rotvec).inv() * tool_rot

    def _relative_euler_xyz(self, tool_rot: ScipyR) -> Optional[np.ndarray]:
        relative_rot = self._relative_rotation_from_detach(tool_rot)
        if relative_rot is None:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return relative_rot.as_euler("xyz", degrees=False)

    def _relative_euler_constraint(
        self,
        x: np.ndarray,
        curr_rotvec: np.ndarray,
    ) -> np.ndarray:
        euler = self._relative_euler_xyz(
            self._candidate_tool_rotation(x, curr_rotvec)
        )
        if euler is None:
            return np.ones(3, dtype=float)
        return self.max_relative_euler_rad - np.abs(euler)

    def _relative_rotation_constraint(
        self,
        x: np.ndarray,
        curr_rotvec: np.ndarray,
    ) -> float:
        relative_rot = self._relative_rotation_from_detach(
            self._candidate_tool_rotation(x, curr_rotvec)
        )
        if relative_rot is None:
            return 1.0
        return float(
            self.max_relative_rotation_rad
            - np.linalg.norm(relative_rot.as_rotvec())
        )

    def _rotation_constraint_margins(self, tool_rot: ScipyR) -> Tuple[np.ndarray, float]:
        euler = self._relative_euler_xyz(tool_rot)
        relative_rot = self._relative_rotation_from_detach(tool_rot)
        if euler is None or relative_rot is None:
            return np.ones(3, dtype=float), 1.0
        euler_margin = self.max_relative_euler_rad - np.abs(euler)
        total_margin = float(
            self.max_relative_rotation_rad
            - np.linalg.norm(relative_rot.as_rotvec())
        )
        return euler_margin, total_margin

    def _make_rotation_feasible(self, tool_rot: ScipyR) -> ScipyR:
        relative_rot = self._relative_rotation_from_detach(tool_rot)
        if relative_rot is None:
            return tool_rot

        limited_rel = relative_rot
        for _ in range(2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                euler = limited_rel.as_euler("xyz", degrees=False)
            euler = np.clip(
                euler,
                -self.max_relative_euler_rad,
                self.max_relative_euler_rad,
            )
            limited_rel = ScipyR.from_euler("xyz", euler)

            rotvec = limited_rel.as_rotvec()
            angle = float(np.linalg.norm(rotvec))
            if self.max_relative_rotation_rad <= 1e-9:
                limited_rel = ScipyR.identity()
            elif angle > self.max_relative_rotation_rad:
                limited_rel = ScipyR.from_rotvec(
                    rotvec * (self.max_relative_rotation_rad / angle)
                )

        detach_rot = ScipyR.from_rotvec(self.detach_rotvec)
        return detach_rot * limited_rel

    def _limit_rotation_step(
        self,
        desired_rot: ScipyR,
        curr_rot: ScipyR,
    ) -> ScipyR:
        step_rot = desired_rot * curr_rot.inv()
        step_rotvec = step_rot.as_rotvec()
        step_angle = float(np.linalg.norm(step_rotvec))
        if step_angle <= self.delta_rot_max or self.delta_rot_max <= 1e-9:
            return desired_rot
        return (
            ScipyR.from_rotvec(step_rotvec * (self.delta_rot_max / step_angle))
            * curr_rot
        )

    def _initial_rotation_step(self, curr_rotvec: np.ndarray) -> np.ndarray:
        curr_rot = ScipyR.from_rotvec(curr_rotvec)
        feasible_rot = self._make_rotation_feasible(curr_rot)
        limited_rot = self._limit_rotation_step(feasible_rot, curr_rot)
        return (limited_rot * curr_rot.inv()).as_rotvec()

    def _limit_tool_rotation(
        self,
        tool_rot: ScipyR,
        curr_rotvec: np.ndarray,
    ) -> ScipyR:
        feasible_rot = self._make_rotation_feasible(tool_rot)
        euler_margin, total_margin = self._rotation_constraint_margins(tool_rot)
        if np.any(euler_margin < -1e-6) or total_margin < -1e-6:
            self.get_logger().warn(
                "natural command orientation clamped to detach-relative limits",
                throttle_duration_sec=1.0,
            )
            curr_rot = ScipyR.from_rotvec(curr_rotvec)
            return self._limit_rotation_step(feasible_rot, curr_rot)
        return feasible_rot

    def _clamp_to_global_tool_bounds(self, pos: np.ndarray) -> np.ndarray:
        return np.minimum(
            np.maximum(np.asarray(pos, dtype=float), self.global_tool_pos_min),
            self.global_tool_pos_max,
        )

    def _clamp_command_position(self, pos: np.ndarray) -> np.ndarray:
        clamped = self._clamp_to_global_tool_bounds(pos)
        if np.linalg.norm(clamped - pos) > 1e-9:
            self.get_logger().warn(
                "natural command clamped to global tool bounds",
                throttle_duration_sec=1.0,
            )
        return clamped

    def _command_change_exceeds_deadband(
        self,
        curr_pos: np.ndarray,
        curr_rotvec: np.ndarray,
        new_pos: np.ndarray,
        new_quat: np.ndarray,
    ) -> bool:
        ref_pos = curr_pos if self.last_command_pos is None else self.last_command_pos
        ref_quat = (
            ScipyR.from_rotvec(curr_rotvec).as_quat()
            if self.last_command_quat is None
            else self.last_command_quat
        )
        linear_step = float(np.linalg.norm(np.asarray(new_pos) - np.asarray(ref_pos)))
        ref_rot = ScipyR.from_quat(ref_quat)
        new_rot = ScipyR.from_quat(new_quat)
        angular_step = float(np.linalg.norm((new_rot * ref_rot.inv()).as_rotvec()))
        return (
            linear_step >= self.command_translation_deadband_m
            or angular_step >= self.command_rotation_deadband_rad
        )

    def _publish_optimizing_active(self, active: bool):
        msg = Bool()
        msg.data = bool(active)
        self.optimizing_pub.publish(msg)

    def _publish_target_lost(self, lost: bool):
        msg = Bool()
        msg.data = bool(lost)
        self.target_lost_pub.publish(msg)

    def _search_direction_from_position(self, curr_pos: np.ndarray) -> float:
        y_center = 0.5 * (self.global_tool_pos_min[1] + self.global_tool_pos_max[1])
        if curr_pos[1] > y_center:
            return -1.0
        return 1.0

    def _publish_search_pose(self, curr_pos: np.ndarray, curr_rotvec: np.ndarray):
        now_sec = self._now_sec()
        if (
            self.search_start_sec is None
            or self.search_center_pos is None
            or self.search_center_rotvec is None
        ):
            self.search_start_sec = now_sec
            self.search_center_pos = curr_pos.copy()
            self.search_center_rotvec = curr_rotvec.copy()
            self.search_initial_direction = self._search_direction_from_position(curr_pos)

        elapsed = max(0.0, now_sec - self.search_start_sec)
        phase = 2.0 * np.pi * self.search_yaw_frequency_hz * elapsed
        yaw = (
            self.search_yaw_axis_sign
            * self.search_initial_direction
            * self.search_yaw_amplitude_rad
            * np.sin(phase)
        )
        center_rot = ScipyR.from_rotvec(self.search_center_rotvec)
        search_rot = ScipyR.from_rotvec(np.array([0.0, 0.0, yaw])) * center_rot
        search_rot = self._limit_tool_rotation(search_rot, curr_rotvec)
        self._publish_target_pose(self.search_center_pos, search_rot.as_quat())
        self._publish_optimizing_active(False)
        self._publish_target_lost(True)

    def optimize_pose(
        self,
        curr_pos: np.ndarray,
        curr_rotvec: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        trans_tool_cam,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x0, bounds = self._build_bounds(curr_pos, curr_rotvec)
        preferred_side_xy = self._preferred_side_from_target(target_quat)
        self._log_side_preference(preferred_side_xy, target_pos)

        constraints = [
            {
                "type": "ineq",
                "fun": self._internal_position_constraint,
            },
            {
                "type": "ineq",
                "fun": self._relative_euler_constraint,
                "args": (curr_rotvec,),
            },
            {
                "type": "ineq",
                "fun": self._relative_rotation_constraint,
                "args": (curr_rotvec,),
            },
        ]

        try:
            res = minimize(
                camera_alignment_residual,
                x0,
                args=(
                    curr_pos,
                    curr_rotvec,
                    target_pos,
                    target_quat,
                    trans_tool_cam,
                    self.look_alignment_weight,
                    self.fused_y_alignment_weight,
                    self.camera_distance_min_m,
                    self.camera_distance_max_m,
                    self.camera_distance_weight,
                    preferred_side_xy,
                    self.side_offset_m,
                    self.side_preference_weight,
                    self.desired_tool_z_m,
                    self.tool_height_weight,
                    self.motion_weight,
                    self.rotation_motion_weight,
                ),
                method="SLSQP",
                bounds=bounds,
                constraints=tuple(constraints),
                options={"disp": False, "maxiter": self.maxiter, "ftol": 1e-6},
            )
            x_opt = res.x if res.success else x0
        except Exception as exc:
            self.get_logger().warn(
                f"camera optimization failed: {exc}",
                throttle_duration_sec=1.0,
            )
            x_opt = x0

        pos = curr_pos + np.asarray(x_opt[:3], dtype=float)
        rot = ScipyR.from_rotvec(x_opt[3:6]) * ScipyR.from_rotvec(curr_rotvec)
        rot = self._limit_tool_rotation(rot, curr_rotvec)
        quat = rot.as_quat()
        pos = self._clamp_command_position(pos)
        return pos, quat

    def _publish_target_pose(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        clamp: bool = True,
    ):
        if clamp:
            pos = self._clamp_command_position(pos)

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
        self.last_command_pos = np.asarray(pos, dtype=float).copy()
        self.last_command_quat = np.asarray(quat, dtype=float).copy()

    def _publish_hold_pose(self, curr_pos: np.ndarray, curr_rotvec: np.ndarray):
        curr_rot = ScipyR.from_rotvec(curr_rotvec)
        curr_quat = self._limit_tool_rotation(curr_rot, curr_rotvec).as_quat()
        self._publish_target_pose(curr_pos, curr_quat)
        self._publish_optimizing_active(False)

    def tick(self):
        if self.mode != NATURAL_MODE:
            return

        tool_state = self._ensure_detach_pose_latched()
        if tool_state is None:
            return
        curr_pos, curr_rotvec, trans_tool_cam = tool_state

        target_state = self._get_active_target_pose()
        if target_state is None or not self._target_visible_active():
            self._publish_search_pose(curr_pos, curr_rotvec)
            return

        if self._target_near_attached_pose():
            self._publish_target_lost(False)
            self._publish_hold_pose(curr_pos, curr_rotvec)
            return

        target_pos, target_quat, stale_age = target_state
        if stale_age > self.hard_stale_sec:
            self._publish_search_pose(curr_pos, curr_rotvec)
            self.get_logger().warn(
                f"target stale for {stale_age:.2f}s; skipping command",
                throttle_duration_sec=1.0,
            )
            return

        self._publish_target_lost(False)
        self.search_start_sec = None
        self.search_center_pos = None
        self.search_center_rotvec = None

        new_pos, new_quat = self.optimize_pose(
            curr_pos,
            curr_rotvec,
            target_pos,
            target_quat,
            trans_tool_cam,
        )
        if self._command_change_exceeds_deadband(curr_pos, curr_rotvec, new_pos, new_quat):
            self._publish_target_pose(new_pos, new_quat)
            self._publish_optimizing_active(True)
        else:
            self._publish_optimizing_active(False)


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
