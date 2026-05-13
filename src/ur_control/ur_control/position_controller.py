#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Int32
from scipy.spatial.transform import Rotation as R


class PositionController(Node):
    def __init__(self):
        super().__init__('position_controller')

        # parameters
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('tool_frame', 'tool0')
        self.declare_parameter('desired_pose_topic', '/ur7e/desired_pose')
        self.declare_parameter('script_topic', '/urscript_interface/script_command')
        self.declare_parameter('mode_topic', '/mode')

        self.declare_parameter('rate_hz', 450.0)
        self.declare_parameter('script_rate_hz', 50.0)
        self.declare_parameter('speedl_time_s', 0.08)
        self.declare_parameter('lin_P', 2.0)
        self.declare_parameter('ang_P', 1.5)
        self.declare_parameter('max_lin_vel', 0.25)   # m/s clamp
        self.declare_parameter('max_ang_vel', 1.0)    # rad/s clamp
        self.declare_parameter('linear_zero_deadband', 5e-4)
        self.declare_parameter('angular_zero_deadband', 1e-3)
        self.declare_parameter('freedrive_exit_hold_s', 1.0)

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.desired_pose_topic = self.get_parameter('desired_pose_topic').value
        self.script_topic = self.get_parameter('script_topic').value
        self.mode_topic = self.get_parameter('mode_topic').value

        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.script_rate_hz = max(1.0, float(self.get_parameter('script_rate_hz').value))
        self.speedl_time_s = max(0.02, float(self.get_parameter('speedl_time_s').value))
        self.lin_P = float(self.get_parameter('lin_P').value)
        self.ang_P = float(self.get_parameter('ang_P').value)
        self.max_lin_vel = float(self.get_parameter('max_lin_vel').value)
        self.max_ang_vel = float(self.get_parameter('max_ang_vel').value)
        self.linear_zero_deadband = max(
            0.0,
            float(self.get_parameter('linear_zero_deadband').value),
        )
        self.angular_zero_deadband = max(
            0.0,
            float(self.get_parameter('angular_zero_deadband').value),
        )
        self.freedrive_exit_hold_s = max(
            0.0,
            float(self.get_parameter('freedrive_exit_hold_s').value),
        )

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # comms
        self.urscript_pub = self.create_publisher(String, self.script_topic, 1)
        self.des_sub = self.create_subscription(PoseStamped, self.desired_pose_topic, self.on_desired_pose, 1)
        self.mode_sub = self.create_subscription(Int32, self.mode_topic, self.on_mode, 10)

        self.des_p = None
        self.des_q = None
        self.mode = -1
        self.was_mode_2 = False
        self.resume_speedl_after_s = None
        self.sent_freedrive_exit_stop = False
        self.last_script_publish_s = None
        self.speedl_stopped = True
        self.warned_tf_unavailable = False

        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

        self.get_logger().info(f"position_controller listening: {self.desired_pose_topic}")
        self.get_logger().info(f"position_controller publishing URScript: {self.script_topic}")
        self.get_logger().info(
            f"URScript speedl publish rate: {self.script_rate_hz:.1f} Hz, "
            f"t={self.speedl_time_s:.3f}s"
        )
        self.get_logger().info(f"TF: {self.base_frame} -> {self.tool_frame}")
        self.get_logger().info(f"Mode topic: {self.mode_topic}")

    def on_desired_pose(self, msg: PoseStamped):
        self.des_p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        self.des_q = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                               msg.pose.orientation.z, msg.pose.orientation.w], dtype=float)

    def on_mode(self, msg: Int32):
        new_mode = int(msg.data)
        if self.mode == 2 and new_mode != 2:
            now_s = self.get_clock().now().nanoseconds / 1e9
            self.resume_speedl_after_s = now_s + self.freedrive_exit_hold_s
            self.sent_freedrive_exit_stop = False
        self.mode = new_mode

    def publish_speedl(self, v, w, force=False):
        now_s = self.get_clock().now().nanoseconds / 1e9
        is_zero = (
            np.linalg.norm(v) <= self.linear_zero_deadband
            and np.linalg.norm(w) <= self.angular_zero_deadband
        )

        if is_zero:
            if self.speedl_stopped and not force:
                return
            v = np.zeros(3, dtype=float)
            w = np.zeros(3, dtype=float)
        elif not force and self.last_script_publish_s is not None:
            min_period_s = 1.0 / self.script_rate_hz
            if (now_s - self.last_script_publish_s) < min_period_s:
                return

        cmd = (
            f"speedl([{v[0]:.6f},{v[1]:.6f},{v[2]:.6f},"
            f"{w[0]:.6f},{w[1]:.6f},{w[2]:.6f}], 0.5, "
            f"t={self.speedl_time_s:.3f})\n"
        )
        self.urscript_pub.publish(String(data=cmd))
        self.last_script_publish_s = now_s
        self.speedl_stopped = is_zero

    def tick(self):
        # Get current EE pose from TF
        try:
            tf_time = rclpy.time.Time()
            if not self.tf_buffer.can_transform(
                self.base_frame,
                self.tool_frame,
                tf_time,
                timeout=Duration(seconds=0.02),
            ):
                if not self.warned_tf_unavailable:
                    self.get_logger().warn(
                        f"Waiting for TF {self.base_frame} -> {self.tool_frame}"
                    )
                    self.warned_tf_unavailable = True
                return
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tool_frame,
                tf_time,
            )
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed: {e}",
                throttle_duration_sec=1.0,
            )
            return
        self.warned_tf_unavailable = False

        curr_p = np.array([
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z
        ], dtype=float)
        curr_q = np.array([
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w
        ], dtype=float)

        # Give freedrive exclusive control in kinesthetic mode.
        if self.mode == 2:
            self.des_p = curr_p.copy()
            self.des_q = curr_q.copy()
            if not self.was_mode_2:
                self.publish_speedl(np.zeros(3), np.zeros(3), force=True)
            self.was_mode_2 = True
            return
        self.was_mode_2 = False

        if self.resume_speedl_after_s is not None:
            now_s = self.get_clock().now().nanoseconds / 1e9
            if now_s < self.resume_speedl_after_s:
                self.des_p = curr_p.copy()
                self.des_q = curr_q.copy()
                return
            self.resume_speedl_after_s = None
            self.sent_freedrive_exit_stop = False

        if self.des_p is None or self.des_q is None:
            return

        # Linear velocity (P)
        dp = self.des_p - curr_p
        v = self.lin_P * dp

        # Clamp linear velocity
        v = np.clip(v, -self.max_lin_vel, self.max_lin_vel)

        # Angular velocity from quaternion error
        R_curr = R.from_quat(curr_q)
        R_des = R.from_quat(self.des_q)
        R_err = R_des * R_curr.inv()
        rotvec = R_err.as_rotvec()
        w = self.ang_P * rotvec
        w = np.clip(w, -self.max_ang_vel, self.max_ang_vel)

        # URScript speedl expects [vx,vy,vz,wx,wy,wz].
        self.publish_speedl(v, w)


def main():
    rclpy.init()
    node = PositionController()
    try:
        rclpy.spin(node)
    finally:
        # Stop the robot when shutting down
        try:
            node.urscript_pub.publish(String(data="speedl([0,0,0,0,0,0], 0.5, t=1.0)\n"))
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
