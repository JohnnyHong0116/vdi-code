#!/usr/bin/env python3
"""
Publishes box waypoints for quickly validating natural-mode workspace limits.

This node keeps the current tool orientation and only moves position.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener


class BBoxRunner(Node):
    def __init__(self):
        super().__init__("bbox_runner")

        self.declare_parameter("base_frame", "base")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("target_pose_topic", "/ur7e/target_pose")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("hold_sec", 3.0)
        self.declare_parameter("start_delay_sec", 1.0)
        self.declare_parameter("include_center", True)
        self.declare_parameter("include_current_start", True)
        self.declare_parameter("auto_shutdown", True)
        self.declare_parameter("workspace_side", "left")  # left | right | both

        # Safe working box defaults (same as natural_handler).
        self.declare_parameter("x_min", 0.20)
        self.declare_parameter("x_max", 0.30)
        self.declare_parameter("y_min", -0.22)   # left side
        self.declare_parameter("y_max", -0.08)   # left side
        self.declare_parameter("y_right_min", 0.08)
        self.declare_parameter("y_right_max", 0.22)
        self.declare_parameter("z_min", 0.30)
        self.declare_parameter("z_max", 0.40)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tool_frame = str(self.get_parameter("tool_frame").value)
        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.hold_sec = float(self.get_parameter("hold_sec").value)
        self.start_delay_sec = float(self.get_parameter("start_delay_sec").value)
        self.include_center = bool(self.get_parameter("include_center").value)
        self.include_current_start = bool(self.get_parameter("include_current_start").value)
        self.auto_shutdown = bool(self.get_parameter("auto_shutdown").value)
        self.workspace_side = str(self.get_parameter("workspace_side").value).lower()

        self.x_min = float(self.get_parameter("x_min").value)
        self.x_max = float(self.get_parameter("x_max").value)
        self.y_min = float(self.get_parameter("y_min").value)
        self.y_max = float(self.get_parameter("y_max").value)
        self.y_right_min = float(self.get_parameter("y_right_min").value)
        self.y_right_max = float(self.get_parameter("y_right_max").value)
        self.z_min = float(self.get_parameter("z_min").value)
        self.z_max = float(self.get_parameter("z_max").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose_pub = self.create_publisher(PoseStamped, self.target_pose_topic, 1)

        self.started = False
        self.start_ns = self.get_clock().now().nanoseconds
        self.orientation = None
        self.waypoints = []
        self.waypoint_idx = 0
        self.waypoint_start_sec = 0.0
        self.finished = False

        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1.0), self.tick)
        self.get_logger().info(f"bbox_runner publishing to {self.target_pose_topic}")
        self.get_logger().info(f"workspace_side={self.workspace_side}")

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _lookup_current_tool_pose(self):
        trans = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, Time())
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
        return pos, quat

    def _build_box_waypoints(self, y_min: float, y_max: float):
        center = np.array(
            [
                0.5 * (self.x_min + self.x_max),
                0.5 * (y_min + y_max),
                0.5 * (self.z_min + self.z_max),
            ],
            dtype=float,
        )

        corners = [
            np.array([self.x_min, y_min, self.z_min], dtype=float),
            np.array([self.x_max, y_min, self.z_min], dtype=float),
            np.array([self.x_max, y_max, self.z_min], dtype=float),
            np.array([self.x_min, y_max, self.z_min], dtype=float),
            np.array([self.x_min, y_min, self.z_max], dtype=float),
            np.array([self.x_max, y_min, self.z_max], dtype=float),
            np.array([self.x_max, y_max, self.z_max], dtype=float),
            np.array([self.x_min, y_max, self.z_max], dtype=float),
        ]

        waypoints = []
        if self.include_center:
            waypoints.append(center)
        waypoints.extend(corners)
        if self.include_center:
            waypoints.append(center)
        return waypoints

    def _build_waypoints(self, curr_pos: np.ndarray):
        waypoints = []
        if self.include_current_start:
            waypoints.append(curr_pos.copy())

        if self.workspace_side == "right":
            waypoints.extend(self._build_box_waypoints(self.y_right_min, self.y_right_max))
        elif self.workspace_side == "both":
            waypoints.extend(self._build_box_waypoints(self.y_min, self.y_max))
            waypoints.extend(self._build_box_waypoints(self.y_right_min, self.y_right_max))
        else:
            waypoints.extend(self._build_box_waypoints(self.y_min, self.y_max))

        return waypoints

    def _publish_pose(self, pos: np.ndarray):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(self.orientation[0])
        msg.pose.orientation.y = float(self.orientation[1])
        msg.pose.orientation.z = float(self.orientation[2])
        msg.pose.orientation.w = float(self.orientation[3])
        self.pose_pub.publish(msg)

    def tick(self):
        if self.finished:
            return

        if (self._now_sec() - self.start_ns * 1e-9) < self.start_delay_sec:
            return

        if not self.started:
            try:
                curr_pos, curr_q = self._lookup_current_tool_pose()
            except Exception:
                return

            self.orientation = curr_q
            self.waypoints = self._build_waypoints(curr_pos)
            self.waypoint_idx = 0
            self.waypoint_start_sec = self._now_sec()
            self.started = True

            self.get_logger().info("Starting bbox waypoint run:")
            for i, wp in enumerate(self.waypoints):
                self.get_logger().info(
                    f"  [{i}] x={wp[0]:.3f}, y={wp[1]:.3f}, z={wp[2]:.3f}"
                )

        wp = self.waypoints[self.waypoint_idx]
        self._publish_pose(wp)

        if (self._now_sec() - self.waypoint_start_sec) >= self.hold_sec:
            self.waypoint_idx += 1
            self.waypoint_start_sec = self._now_sec()

            if self.waypoint_idx >= len(self.waypoints):
                self.finished = True
                self.timer.cancel()
                self.get_logger().info("bbox_runner complete.")
                if self.auto_shutdown:
                    rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = BBoxRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
