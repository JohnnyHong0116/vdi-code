#!/usr/bin/env python3
"""
F = kx compliance controller for UR7e (ROS 2).

Behavior:
- Pass-through of SpaceMouse/teleop target pose.
- Apply compliance offsets from measured wrench: x = F / k.
- Uses bias (tare) and low-pass filtering for stable behavior.

Notes:
- Payload and CoG set on the teach pendant affect the sensor bias; this node
  still tares on startup to remove any residual offset.
"""

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Bool, Int32
from scipy.spatial.transform import Rotation as R
import tf2_ros


class ComplianceController(Node):
    def __init__(self):
        super().__init__('compliance_controller')

        # Parameters
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('tool_frame', 'tool0')
        self.declare_parameter('target_pose_topic', '/ur7e/target_pose')
        self.declare_parameter('teleop_active_topic', '/ur7e/teleop_active')
        self.declare_parameter('mode_topic', '/mode')
        self.declare_parameter('desired_pose_topic', '/ur7e/desired_pose')
        self.declare_parameter('wrench_topic', '/force_torque_sensor_broadcaster/wrench')

        self.declare_parameter('stiffness_linear', 500.0)      # N/m
        self.declare_parameter('stiffness_angular', 100.0)      # Nm/rad
        self.declare_parameter('rate_hz', 450.0)

        self.declare_parameter('force_deadband', 0.1)          # N
        self.declare_parameter('torque_deadband', 0.05)        # Nm
        self.declare_parameter('engage_force_threshold', 2.5)  # N (start compliance)
        self.declare_parameter('disengage_force_threshold', 0.2)  # N (stop compliance)
        self.declare_parameter('max_linear_offset', 0.50)      # m
        self.declare_parameter('max_angular_offset', 0.5)      # rad
        self.declare_parameter('invert_force_sign', False)     # False => yield with force
        self.declare_parameter('invert_torque_sign', True)     # True matches legacy yaw sign
        self.declare_parameter('torque_z_only', True)          # Only yaw compliance
        self.declare_parameter('tare_samples', 50)
        self.declare_parameter('wait_for_tf', False)
        # Latch target on contact so arm returns to the pre-contact target after force is released
        self.declare_parameter('freeze_target_on_contact', True)
        # Ignore target updates while engaged; apply the latest pending target after disengage
        self.declare_parameter('ignore_target_updates_while_engaged', True)
        self.declare_parameter('force_filter_alpha', 0.015)      # low-pass on wrench
        self.declare_parameter('disable_compliance_in_mode2', True)

        # Resolve parameters
        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.target_topic = self.get_parameter('target_pose_topic').value
        self.teleop_active_topic = self.get_parameter('teleop_active_topic').value
        self.mode_topic = self.get_parameter('mode_topic').value
        self.output_topic = self.get_parameter('desired_pose_topic').value
        self.wrench_topic = self.get_parameter('wrench_topic').value

        self.K_lin = float(self.get_parameter('stiffness_linear').value)
        self.K_ang = float(self.get_parameter('stiffness_angular').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)

        self.f_deadband = float(self.get_parameter('force_deadband').value)
        self.t_deadband = float(self.get_parameter('torque_deadband').value)
        self.engage_f = float(self.get_parameter('engage_force_threshold').value)
        self.disengage_f = float(self.get_parameter('disengage_force_threshold').value)
        self.max_lin_offset = float(self.get_parameter('max_linear_offset').value)
        self.max_ang_offset = float(self.get_parameter('max_angular_offset').value)
        self.invert_force = bool(self.get_parameter('invert_force_sign').value)
        self.invert_torque = bool(self.get_parameter('invert_torque_sign').value)
        self.torque_z_only = bool(self.get_parameter('torque_z_only').value)
        self.tare_samples = int(self.get_parameter('tare_samples').value)
        self.wait_for_tf = bool(self.get_parameter('wait_for_tf').value)
        self.freeze_on_contact = bool(self.get_parameter('freeze_target_on_contact').value)
        self.ignore_updates_engaged = bool(self.get_parameter('ignore_target_updates_while_engaged').value)
        self.f_alpha = float(self.get_parameter('force_filter_alpha').value)
        self.disable_compliance_in_mode2 = bool(
            self.get_parameter('disable_compliance_in_mode2').value
        )

        # State
        self.desired_pos = None
        self.desired_q = None
        self.anchor_pos = None
        self.anchor_q = None
        self.engaged = False
        self.teleop_active = False
        self.mode = -1

        self.forces = np.zeros(3, dtype=float)
        self.torques = np.zeros(3, dtype=float)
        self.filt_forces = np.zeros(3, dtype=float)
        self.filt_torques = np.zeros(3, dtype=float)

        # Pending target (used if we decide to sync after disengage)
        self.pending_pos = None
        self.pending_q = None

        # Bias / tare
        self.bias_samples = []
        self.bias_force = np.zeros(3, dtype=float)
        self.bias_torque = np.zeros(3, dtype=float)
        self.is_tared = False
        self.warned_tf = False
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS for sensor data (Best Effort to match drivers)
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subs/Pubs
        self.create_subscription(PoseStamped, self.target_topic, self.on_target_pose, 1)
        self.create_subscription(Bool, self.teleop_active_topic, self.on_teleop_active, 1)
        self.create_subscription(Int32, self.mode_topic, self.on_mode, 10)
        self.create_subscription(WrenchStamped, self.wrench_topic, self.on_wrench, qos_sensor)
        self.compliant_pub = self.create_publisher(PoseStamped, self.output_topic, 1)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

        self.get_logger().info("Compliance Controller active (F=kx).")
        self.get_logger().info(f"Stiffness: Lin={self.K_lin} Ang={self.K_ang}")
        self.get_logger().info(f"Chain: {self.target_topic} -> [compliance] -> {self.output_topic}")
        self.get_logger().info(f"Teleop active topic: {self.teleop_active_topic}")
        self.get_logger().info(f"Mode topic: {self.mode_topic}")
        self.get_logger().info(f"Wrench: {self.wrench_topic} (tool_frame={self.tool_frame})")
        self.get_logger().info(f"Disable compliance in mode 2: {self.disable_compliance_in_mode2}")

    # ------------------------------------------------------------------ #
    # Callbacks
    def on_target_pose(self, msg: PoseStamped):
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ], dtype=float)
        quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ], dtype=float)

        if (self.engaged and self.freeze_on_contact and self.ignore_updates_engaged and not self.teleop_active):
            self.pending_pos = pos
            self.pending_q = quat
            return

        self.desired_pos = pos
        self.desired_q = quat
        # If teleop is moving while engaged, advance the freeze anchor with teleop targets.
        if self.engaged and self.freeze_on_contact and self.teleop_active:
            self.anchor_pos = pos.copy()
            self.anchor_q = quat.copy()

    def on_teleop_active(self, msg: Bool):
        self.teleop_active = bool(msg.data)

    def on_mode(self, msg: Int32):
        self.mode = int(msg.data)

    def on_wrench(self, msg: WrenchStamped):
        raw_f = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z
        ], dtype=float)
        raw_t = np.array([
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ], dtype=float)

        # Rotate wrench into base frame (ignore moment arm)
        source_frame = msg.header.frame_id if msg.header.frame_id else self.tool_frame
        if source_frame != self.base_frame:
            try:
                tf_time = rclpy.time.Time()
                if self.wait_for_tf:
                    if not self.tf_buffer.can_transform(
                        self.base_frame, source_frame, tf_time, timeout=Duration(seconds=0.2)
                    ):
                        return
                tf = self.tf_buffer.lookup_transform(self.base_frame, source_frame, tf_time)
                q = tf.transform.rotation
                r = R.from_quat([q.x, q.y, q.z, q.w])
                raw_f = r.apply(raw_f)
                raw_t = r.apply(raw_t)
            except Exception as ex:
                if not self.warned_tf:
                    self.get_logger().warn(
                        f"TF lookup failed from '{source_frame}' to '{self.base_frame}': {ex}. "
                        "Using raw wrench frame."
                    )
                    self.warned_tf = True
                if self.wait_for_tf:
                    return

        # Tare the first N samples
        if self.tare_samples > 0 and not self.is_tared:
            self.bias_samples.append((raw_f, raw_t))
            if len(self.bias_samples) >= self.tare_samples:
                f_sum = np.zeros(3, dtype=float)
                t_sum = np.zeros(3, dtype=float)
                for f, t in self.bias_samples:
                    f_sum += f
                    t_sum += t
                self.bias_force = f_sum / len(self.bias_samples)
                self.bias_torque = t_sum / len(self.bias_samples)
                self.is_tared = True
                self.get_logger().info(f"Calibration complete. Bias force: {self.bias_force}")
            return

        # Apply bias
        self.forces = raw_f - self.bias_force
        self.torques = raw_t - self.bias_torque

        # Low-pass filter to reduce jitter (use configurable alpha)
        alpha = self.f_alpha
        self.filt_forces = alpha * self.forces + (1 - alpha) * self.filt_forces
        self.filt_torques = alpha * self.torques + (1 - alpha) * self.filt_torques

    # ------------------------------------------------------------------ #
    # Helpers
    @staticmethod
    def apply_deadband(vec, limit):
        if np.linalg.norm(vec) < limit:
            return np.zeros_like(vec)
        return vec

    # ------------------------------------------------------------------ #
    # Control loop
    def tick(self):
        if self.desired_pos is None or self.desired_q is None:
            return

        # If we are still taring, pass through the target pose
        if self.tare_samples > 0 and not self.is_tared:
            self.publish_pose(self.desired_pos, self.desired_q)
            return

        if self.disable_compliance_in_mode2 and self.mode == 2:
            # Give freedrive exclusive control in kinesthetic mode.
            self.engaged = False
            self.anchor_pos = None
            self.anchor_q = None
            self.pending_pos = None
            self.pending_q = None
            self.publish_pose(self.desired_pos, self.desired_q)
            return

        f_active = self.apply_deadband(self.filt_forces, self.f_deadband)
        t_active = self.apply_deadband(self.filt_torques, self.t_deadband)
        if self.torque_z_only:
            t_active = np.array([0.0, 0.0, t_active[2]], dtype=float)
        f_norm = np.linalg.norm(f_active)

        # Engagement logic with hysteresis
        if not self.engaged and f_norm >= self.engage_f and self.is_tared:
            self.engaged = True
            if self.freeze_on_contact:
                self.anchor_pos = self.desired_pos.copy()
                self.anchor_q = self.desired_q.copy()

        if self.engaged and f_norm <= self.disengage_f:
            self.engaged = False
            if self.freeze_on_contact:
                self.anchor_pos = None
                self.anchor_q = None

            if self.ignore_updates_engaged and self.pending_pos is not None:
                self.desired_pos = self.pending_pos
                self.desired_q = self.pending_q
                self.pending_pos = None
                self.pending_q = None

        # Select reference pose
        if self.freeze_on_contact and self.anchor_pos is not None and not self.teleop_active:
            ref_pos = self.anchor_pos
            ref_q = self.anchor_q
        else:
            ref_pos = self.desired_pos
            ref_q = self.desired_q

        # If not engaged, publish ref pose directly
        if not self.engaged or not self.is_tared:
            self.publish_pose(ref_pos, ref_q)
            return

        # Compliance offsets: x = F / k
        if self.K_lin > 0.0:
            sign_f = -1.0 if self.invert_force else 1.0
            lin_offset = sign_f * (f_active / self.K_lin)
        else:
            lin_offset = np.zeros(3, dtype=float)
        lin_offset = np.clip(lin_offset, -self.max_lin_offset, self.max_lin_offset)

        if self.K_ang > 0.0:
            sign_t = -1.0 if self.invert_torque else 1.0
            rot_vec = sign_t * (t_active / self.K_ang)
        else:
            rot_vec = np.zeros(3, dtype=float)
        rot_vec = np.clip(rot_vec, -self.max_ang_offset, self.max_ang_offset)

        pos_out = ref_pos + lin_offset
        q_out = (R.from_rotvec(rot_vec) * R.from_quat(ref_q)).as_quat()

        self.publish_pose(pos_out, q_out)

    def publish_pose(self, pos, quat):
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
        self.compliant_pub.publish(msg)


def main():
    rclpy.init()
    node = ComplianceController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
