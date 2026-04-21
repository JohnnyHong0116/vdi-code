#!/usr/bin/env python3
"""Natural demonstration camera tracker.

This node keeps the camera-to-arm transform from TF, records the arm pose when
mode 3 starts, and optimizes the tool pose so the camera points at the tracked
tool while staying near the detach pose:

    (normalize(p_fused_tool - p_camera) dot z_camera - 1)^2

The command is constrained to a spherical radius around the detach pose.
"""

from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as ScipyR
from std_msgs.msg import Int32
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


class NaturalHandler(Node):
    def __init__(self):
        super().__init__("natural_handler")

        self.declare_parameter("mode_topic", "/mode")
        self.declare_parameter("target_pose_topic", "/ur7e/target_pose")
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

        self.declare_parameter("detach_radius_m", 0.03)
        self.declare_parameter("delta_rot_max", 0.35)
        self.declare_parameter("maxiter", 60)
        self.declare_parameter("disable_near_attached_target", True)
        self.declare_parameter("attached_target_camera_pos", [0.029, -0.025, 0.159])
        self.declare_parameter("attached_target_camera_radius_m", 0.04)

        self.mode_topic = str(self.get_parameter("mode_topic").value)
        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)
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

        self.detach_radius_m = float(self.get_parameter("detach_radius_m").value)
        self.delta_rot_max = float(self.get_parameter("delta_rot_max").value)
        self.maxiter = int(self.get_parameter("maxiter").value)
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

        self.target_pos = None
        self.target_quat = None
        self.last_target_update_sec = None
        self.target_camera_pos = None
        self.last_target_camera_update_sec = None

        self.pose_pub = self.create_publisher(PoseStamped, self.target_pose_topic, 1)

        self.create_subscription(Int32, self.mode_topic, self.on_mode, 10)
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
            f"objective: align camera +z to fused tool; "
            f"detach_radius={self.detach_radius_m:.3f}m, "
            f"delta_rot={self.delta_rot_max:.3f}rad"
        )
        self.get_logger().info(
            f"attached-target guard: enabled={self.disable_near_attached_target}, "
            f"camera_pos={np.round(self.attached_target_camera_pos, 4).tolist()}, "
            f"radius={self.attached_target_camera_radius_m:.3f}m"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_mode(self, msg: Int32):
        prev_mode = self.mode
        self.mode = int(msg.data)

        if prev_mode != NATURAL_MODE and self.mode == NATURAL_MODE:
            self.detach_pos = None
            self.detach_rotvec = None
            self.detach_camera_pos = None
            self.get_logger().info("natural mode entered; waiting to latch detach pose")
            self._ensure_detach_pose_latched()

        if self.mode != NATURAL_MODE:
            tool_state = self._lookup_tool_pose() if prev_mode == NATURAL_MODE else None
            if tool_state is not None:
                curr_pos, curr_rotvec, _ = tool_state
                curr_quat = ScipyR.from_rotvec(curr_rotvec).as_quat()
                self._publish_target_pose(curr_pos, curr_quat, clamp=False)
            self.detach_pos = None
            self.detach_rotvec = None
            self.detach_camera_pos = None

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
        if np.linalg.norm(quat) < 1e-9:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.target_pos = pos.copy()
        self.target_quat = quat.copy()
        self.last_target_update_sec = stamp_sec

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

    def _get_active_target_pos(self) -> Optional[Tuple[np.ndarray, float]]:
        now_sec = self._now_sec()
        if self.target_pos is not None and self.last_target_update_sec is not None:
            age = now_sec - self.last_target_update_sec
            if age <= self.target_timeout_sec:
                return self.target_pos.copy(), age

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
        return pos, now_sec - stamp_sec

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
    ) -> Tuple[np.ndarray, list]:
        anchor_min = self.detach_pos - self.detach_radius_m
        anchor_max = self.detach_pos + self.detach_radius_m
        bounds = []

        for axis in range(3):
            low = anchor_min[axis] - curr_pos[axis]
            high = anchor_max[axis] - curr_pos[axis]
            bounds.append((float(low), float(high)))

        for axis in range(3):
            bounds.append(
                (
                    float(-self.delta_rot_max),
                    float(self.delta_rot_max),
                )
            )

        x0 = np.zeros(6, dtype=float)
        if self.detach_pos is not None:
            x0[:3] = self._clamp_to_detach_radius(curr_pos) - curr_pos
        for idx, (low, high) in enumerate(bounds):
            x0[idx] = np.clip(x0[idx], low, high)
        return x0, bounds

    def _detach_radius_constraint(self, x: np.ndarray, curr_pos: np.ndarray) -> float:
        if self.detach_pos is None:
            return 0.0
        cmd_pos = curr_pos + x[:3]
        return float(self.detach_radius_m - np.linalg.norm(cmd_pos - self.detach_pos))

    def _clamp_to_detach_radius(self, pos: np.ndarray) -> np.ndarray:
        if self.detach_pos is None or self.detach_radius_m <= 1e-9:
            return pos

        offset = pos - self.detach_pos
        dist = float(np.linalg.norm(offset))
        if dist <= self.detach_radius_m:
            return pos

        clamped = self.detach_pos + (offset / dist) * self.detach_radius_m
        self.get_logger().warn(
            f"natural command clamped to detach radius "
            f"({dist:.4f}m > {self.detach_radius_m:.4f}m)",
            throttle_duration_sec=1.0,
        )
        return clamped

    def optimize_pose(
        self,
        curr_pos: np.ndarray,
        curr_rotvec: np.ndarray,
        target_pos: np.ndarray,
        trans_tool_cam,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x0, bounds = self._build_bounds(curr_pos)

        try:
            res = minimize(
                camera_z_alignment_residual,
                x0,
                args=(
                    curr_pos,
                    curr_rotvec,
                    target_pos,
                    trans_tool_cam,
                ),
                method="SLSQP",
                bounds=bounds,
                constraints=(
                    {
                        "type": "ineq",
                        "fun": self._detach_radius_constraint,
                        "args": (curr_pos,),
                    },
                ),
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
        quat = rot.as_quat()
        pos = self._clamp_to_detach_radius(pos)
        return pos, quat

    def _publish_target_pose(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        clamp: bool = True,
    ):
        if clamp:
            pos = self._clamp_to_detach_radius(pos)

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

    def _publish_hold_pose(self, curr_pos: np.ndarray, curr_rotvec: np.ndarray):
        curr_quat = ScipyR.from_rotvec(curr_rotvec).as_quat()
        self._publish_target_pose(curr_pos, curr_quat)

    def tick(self):
        if self.mode != NATURAL_MODE:
            return

        tool_state = self._ensure_detach_pose_latched()
        if tool_state is None:
            return
        curr_pos, curr_rotvec, trans_tool_cam = tool_state

        target_state = self._get_active_target_pos()
        if target_state is None:
            self._publish_hold_pose(curr_pos, curr_rotvec)
            return

        if self._target_near_attached_pose():
            self._publish_hold_pose(curr_pos, curr_rotvec)
            return

        target_pos, stale_age = target_state
        if stale_age > self.hard_stale_sec:
            self._publish_hold_pose(curr_pos, curr_rotvec)
            self.get_logger().warn(
                f"target stale for {stale_age:.2f}s; skipping command",
                throttle_duration_sec=1.0,
            )
            return

        new_pos, new_quat = self.optimize_pose(
            curr_pos,
            curr_rotvec,
            target_pos,
            trans_tool_cam,
        )
        self._publish_target_pose(new_pos, new_quat)


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
