#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node

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
        self.declare_parameter('lin_P', 2.0)
        self.declare_parameter('ang_P', 1.5)
        self.declare_parameter('max_lin_vel', 0.25)   # m/s clamp
        self.declare_parameter('max_ang_vel', 1.0)    # rad/s clamp

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.desired_pose_topic = self.get_parameter('desired_pose_topic').value
        self.script_topic = self.get_parameter('script_topic').value
        self.mode_topic = self.get_parameter('mode_topic').value

        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.lin_P = float(self.get_parameter('lin_P').value)
        self.ang_P = float(self.get_parameter('ang_P').value)
        self.max_lin_vel = float(self.get_parameter('max_lin_vel').value)
        self.max_ang_vel = float(self.get_parameter('max_ang_vel').value)

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

        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

        self.get_logger().info(f"position_controller listening: {self.desired_pose_topic}")
        self.get_logger().info(f"position_controller publishing URScript: {self.script_topic}")
        self.get_logger().info(f"TF: {self.base_frame} -> {self.tool_frame}")
        self.get_logger().info(f"Mode topic: {self.mode_topic}")

    def on_desired_pose(self, msg: PoseStamped):
        self.des_p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        self.des_q = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                               msg.pose.orientation.z, msg.pose.orientation.w], dtype=float)

    def on_mode(self, msg: Int32):
        self.mode = int(msg.data)

    def tick(self):
        # Get current EE pose from TF
        try:
            trans = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

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
                self.urscript_pub.publish(String(data="speedl([0,0,0,0,0,0], 0.5, t=0.02)\n"))
            self.was_mode_2 = True
            return
        self.was_mode_2 = False

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

        # URScript speedl expects [vx,vy,vz,wx,wy,wz]
        cmd = f"speedl([{v[0]:.6f},{v[1]:.6f},{v[2]:.6f},{w[0]:.6f},{w[1]:.6f},{w[2]:.6f}], 0.5, t=0.02)\n"
        self.urscript_pub.publish(String(data=cmd))


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
