#!/usr/bin/env python3
"""Minimal natural demo camera tracker.

This node keeps the camera-to-arm transform from TF, records the arm pose when
natural mode starts, and optimizes only one objective:

    (normalize(p_fused_tool - p_camera) dot z_camera - 1)^2

The command is constrained to a small box around the detach pose.
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
from std_msgs.msg import Bool, Int32
from tf2_ros import Buffer, TransformListener


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
    tool_euler: np.ndarray,
    trans_tool_cam,
    euler_order: str,
) -> Tuple[np.ndarray, ScipyR]:
    p_tool_cam, r_tool_cam = _tool_camera_transform(trans_tool_cam)
    r_tool = ScipyR.from_euler(euler_order, tool_euler)
    cam_pos = r_tool.apply(p_tool_cam) + tool_pos
    cam_rot = r_tool * r_tool_cam
    return cam_pos, cam_rot


def camera_z_alignment_residual(
    x: np.ndarray,
    target_pos: np.ndarray,
    trans_tool_cam,
    euler_order: str,
) -> float:
    tool_pos = np.asarray(x[:3], dtype=float)
    tool_euler = np.asarray(x[3:6], dtype=float)

    cam_pos, cam_rot = _camera_pose_from_tool(
        tool_pos,
        tool_euler,
        trans_tool_cam,
        euler_order,
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
        self.declare_parameter("odom_seen_topic", "/distance_odom_seen")
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
        self.declare_parameter("pre_natural_z_threshold", 0.10)
        self.declare_parameter("pre_natural_stale_sec", 1.0)
        self.declare_parameter("use_tf_target_fallback", True)

        self.declare_parameter("detach_radius_m", 0.05)
        self.declare_parameter("delta_pos_max", 0.001)
        self.declare_parameter("delta_euler_max", 0.01)
        self.declare_parameter("euler_order", "xyz")
        self.declare_parameter("maxiter", 30)

        self.mode_topic = str(self.get_parameter("mode_topic").value)
        self.odom_seen_topic = str(self.get_parameter("odom_seen_topic").value)
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
        self.pre_natural_z_threshold = float(
            self.get_parameter("pre_natural_z_threshold").value
        )
        self.pre_natural_stale_sec = float(
            self.get_parameter("pre_natural_stale_sec").value
        )
        self.use_tf_target_fallback = bool(
            self.get_parameter("use_tf_target_fallback").value
        )

        self.detach_radius_m = float(self.get_parameter("detach_radius_m").value)
        self.delta_pos_max = float(self.get_parameter("delta_pos_max").value)
        self.delta_euler_max = float(self.get_parameter("delta_euler_max").value)
        self.euler_order = str(self.get_parameter("euler_order").value).lower()
        if self.euler_order not in ("xyz", "zyx", "yxz", "zxy", "xzy", "yzx"):
            self.get_logger().warn(
                f"unsupported euler_order '{self.euler_order}', using xyz"
            )
            self.euler_order = "xyz"
        self.maxiter = int(self.get_parameter("maxiter").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.mode = 0
        self.detach_pos = None
        self.detach_euler = None
        self.detach_camera_pos = None

        self.target_pos = None
        self.target_quat = None
        self.last_target_update_sec = None
        self.last_fused_cam_z = None
        self.last_fused_cam_z_sec = None

        self.odom_seen_pub = self.create_publisher(Bool, self.odom_seen_topic, 10)
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

        self.get_logger().info("NaturalHandler minimal camera optimizer active")
        self.get_logger().info(
            f"frames: base={self.base_frame}, tool={self.tool_frame}, "
            f"camera={self.camera_frame}, target={self.target_frame}"
        )
        self.get_logger().info(
            f"objective: align camera +z to fused tool; "
            f"detach_radius={self.detach_radius_m:.3f}m, "
            f"euler_order={self.euler_order}"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_mode(self, msg: Int32):
        prev_mode = self.mode
        self.mode = int(msg.data)

        if prev_mode != 3 and self.mode == 3:
            self.detach_pos = None
            self.detach_euler = None
            self.detach_camera_pos = None
            self.get_logger().info("natural mode entered; waiting to latch detach pose")

        if self.mode not in (3, 4):
            self.detach_pos = None
            self.detach_euler = None
            self.detach_camera_pos = None
            self.odom_seen_pub.publish(Bool(data=False))

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
            self.last_fused_cam_z = float(pos[2])
            self.last_fused_cam_z_sec = stamp_sec

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
        euler = ScipyR.from_quat(quat).as_euler(self.euler_order)
        return pos, euler, trans_tool_cam

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

    def _publish_pre_natural_seen(self):
        now_sec = self._now_sec()
        if self.last_fused_cam_z is not None and self.last_fused_cam_z_sec is not None:
            age = now_sec - self.last_fused_cam_z_sec
            if age <= self.pre_natural_stale_sec:
                seen = abs(self.last_fused_cam_z) > self.pre_natural_z_threshold
                self.odom_seen_pub.publish(Bool(data=bool(seen)))
                return

        try:
            trans = self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.target_frame,
                Time(),
            )
        except Exception:
            self.odom_seen_pub.publish(Bool(data=False))
            return

        stamp_sec = _stamp_to_sec(trans.header.stamp)
        age = now_sec - stamp_sec if stamp_sec > 0.0 else 0.0
        seen = (
            abs(trans.transform.translation.z) > self.pre_natural_z_threshold
            and age <= self.pre_natural_stale_sec
        )
        self.odom_seen_pub.publish(Bool(data=bool(seen)))

    def _latch_detach_pose(
        self,
        curr_pos: np.ndarray,
        curr_euler: np.ndarray,
        trans_tool_cam,
    ):
        self.detach_pos = curr_pos.copy()
        self.detach_euler = curr_euler.copy()
        cam_pos, _ = _camera_pose_from_tool(
            curr_pos,
            curr_euler,
            trans_tool_cam,
            self.euler_order,
        )
        self.detach_camera_pos = cam_pos.copy()
        self.get_logger().info(
            "latched detach pose: "
            f"tool={np.round(self.detach_pos, 4).tolist()}, "
            f"camera={np.round(self.detach_camera_pos, 4).tolist()}"
        )

    def _build_bounds(
        self,
        curr_pos: np.ndarray,
        curr_euler: np.ndarray,
    ) -> Tuple[np.ndarray, list]:
        anchor_min = self.detach_pos - self.detach_radius_m
        anchor_max = self.detach_pos + self.detach_radius_m
        bounds = []

        for axis in range(3):
            step_min = curr_pos[axis] - self.delta_pos_max
            step_max = curr_pos[axis] + self.delta_pos_max
            low = max(anchor_min[axis], step_min)
            high = min(anchor_max[axis], step_max)
            if low > high:
                target = np.clip(curr_pos[axis], anchor_min[axis], anchor_max[axis])
                target = np.clip(target, step_min, step_max)
                low = high = float(target)
            bounds.append((float(low), float(high)))

        for axis in range(3):
            bounds.append(
                (
                    float(curr_euler[axis] - self.delta_euler_max),
                    float(curr_euler[axis] + self.delta_euler_max),
                )
            )

        x0 = np.concatenate((curr_pos, curr_euler)).astype(float)
        for idx, (low, high) in enumerate(bounds):
            x0[idx] = np.clip(x0[idx], low, high)
        return x0, bounds

    def optimize_pose(
        self,
        curr_pos: np.ndarray,
        curr_euler: np.ndarray,
        target_pos: np.ndarray,
        trans_tool_cam,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x0, bounds = self._build_bounds(curr_pos, curr_euler)

        try:
            res = minimize(
                camera_z_alignment_residual,
                x0,
                args=(target_pos, trans_tool_cam, self.euler_order),
                method="SLSQP",
                bounds=bounds,
                options={"disp": False, "maxiter": self.maxiter, "ftol": 1e-6},
            )
            x_opt = res.x if res.success else x0
        except Exception as exc:
            self.get_logger().warn(
                f"camera optimization failed: {exc}",
                throttle_duration_sec=1.0,
            )
            x_opt = x0

        pos = np.asarray(x_opt[:3], dtype=float)
        euler = np.asarray(x_opt[3:6], dtype=float)
        quat = ScipyR.from_euler(self.euler_order, euler).as_quat()
        return pos, quat

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

    def tick(self):
        if self.mode == 4:
            self._publish_pre_natural_seen()
            return

        if self.mode != 3:
            return

        tool_state = self._lookup_tool_pose()
        if tool_state is None:
            return
        curr_pos, curr_euler, trans_tool_cam = tool_state

        if self.detach_pos is None:
            self._latch_detach_pose(curr_pos, curr_euler, trans_tool_cam)

        target_state = self._get_active_target_pos()
        if target_state is None:
            self.odom_seen_pub.publish(Bool(data=False))
            return

        target_pos, stale_age = target_state
        if stale_age > self.hard_stale_sec:
            self.odom_seen_pub.publish(Bool(data=False))
            self.get_logger().warn(
                f"target stale for {stale_age:.2f}s; skipping command",
                throttle_duration_sec=1.0,
            )
            return

        self.odom_seen_pub.publish(Bool(data=True))
        new_pos, new_quat = self.optimize_pose(
            curr_pos,
            curr_euler,
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
