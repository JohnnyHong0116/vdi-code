#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node

import tf2_ros
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int32

from scipy.spatial.transform import Rotation as R
import pyspacemouse

class SMTeleop(Node):

    def __init__(self):
        super().__init__('sm_teleop')

        # parameters
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('tool_frame', 'tool0')
        self.declare_parameter('desired_pose_topic', '/ur7e/target_pose')
        self.declare_parameter('teleop_active_topic', '/ur7e/teleop_active')
        self.declare_parameter('mode_topic', '/mode')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('teleop_active_hold_s', 0.25)

        self.declare_parameter('w_lin', 0.02)   # meters per tick * spacemouse unit
        self.declare_parameter('w_ang', 0.05)   # rad per tick * spacemouse unit

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.desired_pose_topic = self.get_parameter('desired_pose_topic').value
        self.teleop_active_topic = self.get_parameter('teleop_active_topic').value
        self.mode_topic = self.get_parameter('mode_topic').value
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.w_lin = float(self.get_parameter('w_lin').value)
        self.w_ang = float(self.get_parameter('w_ang').value)
        self.active_hold_s = float(self.get_parameter('teleop_active_hold_s').value)
        self.active_hold_ns = int(max(0.0, self.active_hold_s) * 1e9)

        # Deadband to prevent drift
        self.declare_parameter('deadband', 0.15)
        self.deadband = float(self.get_parameter('deadband').value)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # publishers
        self.pose_pub = self.create_publisher(PoseStamped, self.desired_pose_topic, 1)
        self.active_pub = self.create_publisher(Bool, self.teleop_active_topic, 1)
        self.mode_sub = self.create_subscription(Int32, self.mode_topic, self.on_mode, 10)

        # SpaceMouse
        self.sm_device = None
        self.use_v1_api = False

        try:
            res = pyspacemouse.open()
            if isinstance(res, bool):
                # v1.x API: open() returns Success(bool). Functions are module-level.
                if not res:
                    raise RuntimeError("Failed to open SpaceMouse (v1 API returned False)")
                self.use_v1_api = True
                self.get_logger().info("Detected PySpaceMouse v1.x API")
            else:
                # v2.x API: open() returns a device object.
                if res is None:
                    raise RuntimeError("Failed to open SpaceMouse (v2 API returned None)")
                self.sm_device = res
                self.use_v1_api = False
                self.get_logger().info("Detected PySpaceMouse v2.x API")

        except Exception as e:
            self.get_logger().warn(f"SpaceMouse open error: {e}")

        self.got_robot_pose = False
        self.curr_pos = None
        self.curr_q = None
        self.last_active_ns = 0
        self.mode = -1

        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)
        self.get_logger().info(f"sm_teleop publishing to {self.desired_pose_topic}")
        self.get_logger().info(f"sm_teleop active topic {self.teleop_active_topic}")
        self.get_logger().info(f"sm_teleop mode topic {self.mode_topic}")
        self.get_logger().info(f"TF: {self.base_frame} -> {self.tool_frame}")

    def apply_deadband(self, val):
        if abs(val) < self.deadband:
            return 0.0
        return val

    def on_mode(self, msg: Int32):
        self.mode = int(msg.data)

    def _publish_active(self, active: bool):
        active_msg = Bool()
        active_msg.data = bool(active)
        self.active_pub.publish(active_msg)

    def _sync_to_actual_pose(self, actual_pos, actual_q):
        if actual_pos is None or actual_q is None:
            return
        self.curr_pos = actual_pos.copy()
        self.curr_q = actual_q.copy()
        self.got_robot_pose = True

    def tick(self):
        # 1) Get Actual Robot Pose (latest available)
        actual_pos = None
        actual_q = None
        try:
            trans = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, rclpy.time.Time())
            actual_pos = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ], dtype=float)
            actual_q = np.array([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ], dtype=float)
        except Exception:
            pass

        # Initialize if not done yet
        if not self.got_robot_pose:
            if actual_pos is not None:
                self.curr_pos = actual_pos.copy()
                self.curr_q = actual_q.copy()
                self.got_robot_pose = True
                self.get_logger().info("Initialized SpaceMouse teleop pose from TF.")
            else:
                return  # Wait for TF

        if self.mode in (3, 4):
            self.last_active_ns = 0
            self._publish_active(False)
            self._sync_to_actual_pose(actual_pos, actual_q)
            return

        # 2) Read SpaceMouse
        sm_state = None
        for _ in range(50):  # Clear buffer
            try:
                if self.use_v1_api:
                    tmp = pyspacemouse.read()
                else:
                    # v2.x: read() is a method of the device object
                    tmp = self.sm_device.read()

                if tmp is None:
                    break
                sm_state = tmp
            except Exception:
                break

        # Default to 0 input
        lx, ly, lz = 0.0, 0.0, 0.0
        rx, ry, rz = 0.0, 0.0, 0.0

        if sm_state is not None:
            # Apply deadband
            lx = self.apply_deadband(sm_state.x)
            ly = self.apply_deadband(sm_state.y)
            lz = self.apply_deadband(sm_state.z)
            rx = self.apply_deadband(sm_state.roll)
            ry = self.apply_deadband(sm_state.pitch)
            rz = self.apply_deadband(sm_state.yaw)

        # Check for Idle (All zeros)
        is_idle = (lx == 0.0 and ly == 0.0 and lz == 0.0 and
                   rx == 0.0 and ry == 0.0 and rz == 0.0)

        now_ns = self.get_clock().now().nanoseconds
        if not is_idle:
            self.last_active_ns = now_ns

        active = (not is_idle)
        if self.active_hold_ns > 0 and self.last_active_ns > 0:
            active = active or ((now_ns - self.last_active_ns) <= self.active_hold_ns)

        self._publish_active(active)

        # 3) Update Poses
        if is_idle and actual_pos is not None:
            # SYNC ON IDLE: Snap desired pose to actual robot pose.
            # This kills any accumulated error/drift immediately.
            self.curr_pos = actual_pos.copy()
            self.curr_q = actual_q.copy()
        else:
            # INTEGRATE INPUT
            self.curr_pos[0] += self.w_lin * ly
            self.curr_pos[1] -= self.w_lin * lx  # Inverted Y
            self.curr_pos[2] += self.w_lin * lz

            # Rotation
            rot_x = rx
            rot_y = ry
            rot_z = -rz  # Inverted Yaw

            if not (rot_x == 0 and rot_y == 0 and rot_z == 0):
                R_sm = R.from_rotvec([
                    self.w_ang * rot_x,
                    self.w_ang * rot_y,
                    self.w_ang * rot_z
                ])
                R_old = R.from_quat(self.curr_q)
                q_new = (R_sm * R_old).as_quat()
                self.curr_q = q_new.copy()

        # 4) Publish desired pose
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x = float(self.curr_pos[0])
        msg.pose.position.y = float(self.curr_pos[1])
        msg.pose.position.z = float(self.curr_pos[2])
        msg.pose.orientation.x = float(self.curr_q[0])
        msg.pose.orientation.y = float(self.curr_q[1])
        msg.pose.orientation.z = float(self.curr_q[2])
        msg.pose.orientation.w = float(self.curr_q[3])

        self.pose_pub.publish(msg)


def main():
    rclpy.init()
    node = SMTeleop()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
