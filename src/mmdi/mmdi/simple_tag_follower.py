#!/usr/bin/env python3
"""
Simple tag follower for debugging natural demonstrations.

Tracks target_frame in camera_frame and publishes a tool pose on /ur7e/target_pose.
Compared to natural_handler, this avoids nonlinear optimization and uses direct P
steps in camera coordinates with safety clamps.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as ScipyR
from tf2_ros import Buffer, TransformListener


def _quat_from_msg(qmsg) -> np.ndarray:
    return np.array([qmsg.x, qmsg.y, qmsg.z, qmsg.w], dtype=float)


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class SimpleTagFollower(Node):
    def __init__(self):
        super().__init__("simple_tag_follower")

        # Frames/topics
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("camera_frame", "head_camera")
        self.declare_parameter("target_frame", "odom")
        self.declare_parameter("target_pose_topic", "/ur7e/target_pose")

        # Control behavior
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("desired_distance", 0.40)
        self.declare_parameter("k_xy", 0.8)
        self.declare_parameter("k_z", 0.8)
        self.declare_parameter("k_ang", 0.8)
        self.declare_parameter("enable_orientation_control", True)
        self.declare_parameter("xy_deadband", 0.005)
        self.declare_parameter("z_deadband", 0.010)
        self.declare_parameter("max_step_xy", 0.0025)
        self.declare_parameter("max_step_z", 0.0030)
        self.declare_parameter("max_step_ang", 0.035)  # rad per tick
        self.declare_parameter("stale_timeout_sec", 0.40)
        self.declare_parameter("target_filter_alpha", 0.25)

        # Safety workspace
        self.declare_parameter("x_min", 0.15)
        self.declare_parameter("x_max", 0.34)
        self.declare_parameter("y_min", -0.25)
        self.declare_parameter("y_max", 0.25)
        self.declare_parameter("z_min", 0.27)
        self.declare_parameter("z_max", 0.43)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tool_frame = str(self.get_parameter("tool_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.desired_distance = float(self.get_parameter("desired_distance").value)
        self.k_xy = float(self.get_parameter("k_xy").value)
        self.k_z = float(self.get_parameter("k_z").value)
        self.k_ang = float(self.get_parameter("k_ang").value)
        self.enable_orientation = bool(self.get_parameter("enable_orientation_control").value)
        self.xy_deadband = float(self.get_parameter("xy_deadband").value)
        self.z_deadband = float(self.get_parameter("z_deadband").value)
        self.max_step_xy = float(self.get_parameter("max_step_xy").value)
        self.max_step_z = float(self.get_parameter("max_step_z").value)
        self.max_step_ang = float(self.get_parameter("max_step_ang").value)
        self.stale_timeout = float(self.get_parameter("stale_timeout_sec").value)
        self.alpha = float(self.get_parameter("target_filter_alpha").value)

        self.cart_mins = np.array(
            [
                float(self.get_parameter("x_min").value),
                float(self.get_parameter("y_min").value),
                float(self.get_parameter("z_min").value),
            ],
            dtype=float,
        )
        self.cart_maxs = np.array(
            [
                float(self.get_parameter("x_max").value),
                float(self.get_parameter("y_max").value),
                float(self.get_parameter("z_max").value),
            ],
            dtype=float,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose_pub = self.create_publisher(PoseStamped, self.target_pose_topic, 1)

        self.target_cam_filt = None
        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1.0), self.tick)

        self.get_logger().info("simple_tag_follower active")
        self.get_logger().info(
            f"Tracking {self.camera_frame} -> {self.target_frame}, publish {self.target_pose_topic}"
        )
        self.get_logger().info(
            f"distance={self.desired_distance:.3f}m, k_xy={self.k_xy:.2f}, k_z={self.k_z:.2f}, k_ang={self.k_ang:.2f}"
        )
        self.get_logger().info(
            f"workspace x[{self.cart_mins[0]:.3f},{self.cart_maxs[0]:.3f}] "
            f"y[{self.cart_mins[1]:.3f},{self.cart_maxs[1]:.3f}] "
            f"z[{self.cart_mins[2]:.3f},{self.cart_maxs[2]:.3f}]"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def tick(self):
        try:
            t_bt = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, Time())
            t_tc = self.tf_buffer.lookup_transform(self.tool_frame, self.camera_frame, Time())
            t_ct = self.tf_buffer.lookup_transform(self.camera_frame, self.target_frame, Time())
        except Exception:
            return

        target_stamp = _stamp_to_sec(t_ct.header.stamp)
        if target_stamp > 0.0 and (self._now_sec() - target_stamp) > self.stale_timeout:
            return

        p_tool = np.array(
            [
                t_bt.transform.translation.x,
                t_bt.transform.translation.y,
                t_bt.transform.translation.z,
            ],
            dtype=float,
        )
        q_tool = _quat_from_msg(t_bt.transform.rotation)
        r_tool = ScipyR.from_quat(q_tool)

        p_tc = np.array(
            [
                t_tc.transform.translation.x,
                t_tc.transform.translation.y,
                t_tc.transform.translation.z,
            ],
            dtype=float,
        )
        q_tc = _quat_from_msg(t_tc.transform.rotation)
        r_tc = ScipyR.from_quat(q_tc)

        p_target_cam_raw = np.array(
            [
                t_ct.transform.translation.x,
                t_ct.transform.translation.y,
                t_ct.transform.translation.z,
            ],
            dtype=float,
        )

        if self.target_cam_filt is None:
            self.target_cam_filt = p_target_cam_raw.copy()
        else:
            a = np.clip(self.alpha, 0.0, 1.0)
            self.target_cam_filt = a * p_target_cam_raw + (1.0 - a) * self.target_cam_filt

        p_target_cam = self.target_cam_filt
        if p_target_cam[2] <= 1e-4:
            return

        # Camera-frame tracking error: want [0, 0, desired_distance].
        ex = p_target_cam[0]
        ey = p_target_cam[1]
        ez = p_target_cam[2] - self.desired_distance

        if abs(ex) < self.xy_deadband:
            ex = 0.0
        if abs(ey) < self.xy_deadband:
            ey = 0.0
        if abs(ez) < self.z_deadband:
            ez = 0.0

        d_cam = np.array(
            [
                np.clip(self.k_xy * ex, -self.max_step_xy, self.max_step_xy),
                np.clip(self.k_xy * ey, -self.max_step_xy, self.max_step_xy),
                np.clip(self.k_z * ez, -self.max_step_z, self.max_step_z),
            ],
            dtype=float,
        )

        r_cam = r_tool * r_tc
        d_base = r_cam.apply(d_cam)
        p_tool_des = np.clip(p_tool + d_base, self.cart_mins, self.cart_maxs)

        # Optional orientation alignment: rotate camera z-axis toward target vector.
        if self.enable_orientation:
            v = p_target_cam / (np.linalg.norm(p_target_cam) + 1e-9)
            z = np.array([0.0, 0.0, 1.0], dtype=float)
            axis = np.cross(z, v)
            axis_n = np.linalg.norm(axis)
            if axis_n > 1e-6:
                axis_u = axis / axis_n
                angle = np.arctan2(axis_n, float(np.dot(z, v)))
                step = np.clip(self.k_ang * angle, -self.max_step_ang, self.max_step_ang)
                r_delta_local = ScipyR.from_rotvec(axis_u * step)
                r_cam_des = r_cam * r_delta_local
                r_tool_des = r_cam_des * r_tc.inv()
                q_tool_des = r_tool_des.as_quat()
            else:
                q_tool_des = q_tool
        else:
            q_tool_des = q_tool

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x = float(p_tool_des[0])
        msg.pose.position.y = float(p_tool_des[1])
        msg.pose.position.z = float(p_tool_des[2])
        msg.pose.orientation.x = float(q_tool_des[0])
        msg.pose.orientation.y = float(q_tool_des[1])
        msg.pose.orientation.z = float(q_tool_des[2])
        msg.pose.orientation.w = float(q_tool_des[3])
        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleTagFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
