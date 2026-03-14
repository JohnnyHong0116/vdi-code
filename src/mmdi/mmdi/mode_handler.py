#!/usr/bin/env python3

"""Mode (state machine) handling and LED publishing
Converted to ROS 2 from ROS 1
Original Author: Mike Hagenow
Last Updated: 6/11/2024

MODE DESCRIPTION:
-1: initialization
 0: idle (red)
 1: teleop (green)
 2: kinesthetic (yellow)
 3: natural (blue)
 4: pre_natural (blue)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Bool, Float32
from geometry_msgs.msg import WrenchStamped
from scipy.spatial.transform import Rotation as ScipyR
import time

from tf2_ros import Buffer, TransformListener
from rclpy.time import Time


class ModeHandler(Node):
    def __init__(self):
        super().__init__('mode_handler')
        self.declare_parameter('spacemouse_input_topic', '/ur7e/teleop_active')
        self.declare_parameter('external_force_topic', '/uniforce/force')
        self.declare_parameter('wrench_topic', '/force_torque_sensor_broadcaster/wrench')
        self.declare_parameter('tool_frame', 'tool0')
        self.declare_parameter('force_discrepancy_axis_sign', -1.0)
        self.declare_parameter('kinesthetic_trigger_direction', -1.0)
        self.declare_parameter('force_discrepancy_threshold_z', 4.0)
        self.declare_parameter('force_discrepancy_release_threshold_z', 2.0)
        self.declare_parameter('kinesthetic_request_hold_s', 2.0)
        self.declare_parameter('tool_contact_debounce_s', 0.5)
        self.declare_parameter('external_force_default', 0.0)
        self.declare_parameter('wrench_tare_samples', 50)
        self.declare_parameter('enable_freedrive_controller_topic', False)
        self.spacemouse_input_topic = self.get_parameter('spacemouse_input_topic').value
        self.external_force_topic = self.get_parameter('external_force_topic').value
        self.wrench_topic = self.get_parameter('wrench_topic').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.force_discrepancy_axis_sign = float(
            self.get_parameter('force_discrepancy_axis_sign').value
        )
        self.kinesthetic_trigger_direction = float(
            self.get_parameter('kinesthetic_trigger_direction').value
        )
        if self.kinesthetic_trigger_direction == 0.0:
            self.kinesthetic_trigger_direction = 1.0
        else:
            self.kinesthetic_trigger_direction = (
                1.0 if self.kinesthetic_trigger_direction > 0.0 else -1.0
            )
        self.force_discrepancy_threshold_z = float(
            self.get_parameter('force_discrepancy_threshold_z').value
        )
        self.force_discrepancy_release_threshold_z = float(
            self.get_parameter('force_discrepancy_release_threshold_z').value
        )
        self.kinesthetic_request_hold_s = float(
            self.get_parameter('kinesthetic_request_hold_s').value
        )
        self.tool_contact_debounce_s = float(
            self.get_parameter('tool_contact_debounce_s').value
        )
        self.external_force_default = float(
            self.get_parameter('external_force_default').value
        )
        self.wrench_tare_samples = max(
            0, int(self.get_parameter('wrench_tare_samples').value)
        )
        self.enable_freedrive_controller_topic = bool(
            self.get_parameter('enable_freedrive_controller_topic').value
        )

        # Initial state
        self.tool_contact = 1
        self.curr_sm = False
        self.external_force = self.external_force_default
        self.odom_seen = False
        self.tool_force_z = 0.0
        self.kinesthetic_armed = True
        self.last_wrench = None
        self.last_no_contact = None
        self.wrench_bias_z = 0.0
        self.wrench_tare_accum_z = 0.0
        self.wrench_tare_count = 0
        self.wrench_is_tared = (self.wrench_tare_samples == 0)
        self.kinesthetic_request_start = None
        self.kinesthetic_entry_armed_after_hold = False

        # Publishers
        self.led_pub = self.create_publisher(String, '/led_state', 10)
        self.mode_pub = self.create_publisher(Int32, '/mode', 10)
        self.freedrive_pub = self.create_publisher(Bool, '/freedrive_mode_controller/enable_freedrive_mode', 10)

        # Subscribers
        self.contact_sub = self.create_subscription(
            Int32, '/tool_contact', self.store_tool_contact, 10)
        self.sm_sub = self.create_subscription(
            Bool, self.spacemouse_input_topic, self.store_curr_sm, 10)
        self.odom_sub = self.create_subscription(
            Bool, '/distance_odom_seen', self.store_odom_seen, 10)
        self.uniforce_sub = self.create_subscription(
            Float32, self.external_force_topic, self.store_uni_force, 10)
        self.wrench_sub = self.create_subscription(
            WrenchStamped, self.wrench_topic, self.store_wrench_global, 10)
        # Direct mode command subscriber
        self.mode_cmd_sub = self.create_subscription(
            Int32, '/mode_cmd', self.handle_mode_cmd, 10)

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timing
        self.start_mode_zero = time.time()
        self.start_mode_four = None

        # Mode state
        self.mode = -1  # Start with initialization
        self.target_mode = None  # For handling direct mode commands
        self.mode_colors = {0: 'r', 1: 'g', 2: 'y', 3: 'b', 4: 'b'}

        self.get_logger().info('ModeHandler initialized, waiting for startup...')
        self.get_logger().info(f'SpaceMouse input topic: {self.spacemouse_input_topic}')
        self.get_logger().info(f'Wrench topic: {self.wrench_topic}')
        self.get_logger().info(f'External force topic: {self.external_force_topic}')
        self.get_logger().info(
            f'Kinesthetic trigger direction: {self.kinesthetic_trigger_direction:+.0f} '
            '(+1 uses positive discrepancy, -1 uses negative discrepancy)'
        )
        if self.wrench_tare_samples > 0:
            self.get_logger().info(
                f'Taring wrench Z with first {self.wrench_tare_samples} samples'
            )
        self.get_logger().info(
            f'Kinesthetic force hold before mode 2: {self.kinesthetic_request_hold_s:.2f}s'
        )
        self.get_logger().info(
            f'Publish /freedrive_mode_controller/enable_freedrive_mode: '
            f'{self.enable_freedrive_controller_topic}'
        )

        # Defer start so TF and other nodes can initialize
        self._init_timer = self.create_timer(2.0, self._start_processing)

    def _start_processing(self):
        self._init_timer.cancel()
        # Main processing timer at 5 Hz
        self.timer = self.create_timer(0.2, self.mode_processor)

    def store_curr_sm(self, msg):
        self.curr_sm = msg.data

    def store_uni_force(self, msg):
        self.external_force = msg.data

    def store_wrench_global(self, msg):
        self.last_wrench = msg
        force_tool = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
            ],
            dtype=float,
        )

        source_frame = msg.header.frame_id if msg.header.frame_id else self.tool_frame
        if source_frame != self.tool_frame:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.tool_frame, source_frame, Time()
                )
                q = tf.transform.rotation
                rot = np.array([q.x, q.y, q.z, q.w], dtype=float)
                force_tool = ScipyR.from_quat(rot).apply(force_tool)
            except Exception as exc:
                self.get_logger().debug(f'Wrench TF lookup failed: {exc}')

        raw_tool_force_z = float(force_tool[2])
        if not self.wrench_is_tared:
            self.wrench_tare_accum_z += raw_tool_force_z
            self.wrench_tare_count += 1
            self.tool_force_z = 0.0
            if self.wrench_tare_count >= self.wrench_tare_samples:
                self.wrench_bias_z = self.wrench_tare_accum_z / self.wrench_tare_count
                self.wrench_is_tared = True
                self.get_logger().info(
                    f'Wrench tare complete. Z bias: {self.wrench_bias_z:.3f} N'
                )
            return

        self.tool_force_z = raw_tool_force_z - self.wrench_bias_z

    def store_tool_contact(self, msg):
        self.tool_contact = msg.data

    def store_odom_seen(self, msg):
        self.odom_seen = msg.data

    def handle_mode_cmd(self, msg):
        """Handle direct mode command - request mode change via tick loop"""
        cmd_mode = msg.data
        if cmd_mode in [0, 1, 2, 3, 4]:
            self.target_mode = cmd_mode
            mode_names = {0: 'idle', 1: 'teleop', 2: 'freedrive', 3: 'natural', 4: 'pre-natural'}
            self.get_logger().info(f'Mode command received: requesting mode {cmd_mode} ({mode_names.get(cmd_mode, "unknown")})')
        else:
            self.get_logger().warn(f'Invalid mode command: {cmd_mode}')

    def force_discrepancy(self):
        sensed_pull = self.force_discrepancy_axis_sign * self.tool_force_z
        return sensed_pull - self.external_force

    def attached(self):
        return int(self.tool_contact) != 0

    def detached_stably(self):
        if self.attached():
            self.last_no_contact = None
            return False
        if self.last_no_contact is None:
            self.last_no_contact = time.time()
            return False
        return (time.time() - self.last_no_contact) >= self.tool_contact_debounce_s

    def mode_processor(self):
        """Main mode processing loop"""
        # Mode transitions
        new_mode = self.mode

        # Check for direct mode command override
        manual_override = False
        if self.target_mode is not None:
            new_mode = self.target_mode
            self.target_mode = None  # Clear after using
            manual_override = True
            self.kinesthetic_entry_armed_after_hold = False

        discrepancy = self.force_discrepancy()
        directional_discrepancy = self.kinesthetic_trigger_direction * discrepancy
        kinesthetic_requested = directional_discrepancy >= self.force_discrepancy_threshold_z
        kinesthetic_released = directional_discrepancy <= self.force_discrepancy_release_threshold_z
        if not self.wrench_is_tared:
            kinesthetic_requested = False
            kinesthetic_released = True
        if kinesthetic_requested:
            if self.kinesthetic_request_start is None:
                self.kinesthetic_request_start = time.time()
            kinesthetic_requested_held = (
                time.time() - self.kinesthetic_request_start
            ) >= self.kinesthetic_request_hold_s
        else:
            self.kinesthetic_request_start = None
            kinesthetic_requested_held = False
        detached = self.detached_stably()

        if kinesthetic_released:
            self.kinesthetic_armed = True

        if not manual_override:
            if self.mode == -1:
                new_mode = 4 if detached else 1
            elif detached:
                self.kinesthetic_entry_armed_after_hold = False
                new_mode = 3 if self.odom_seen else 4
                if self.mode not in (3, 4):
                    self.start_mode_four = time.time()
            else:
                # Reattaching the tool should always return to teleop.
                self.start_mode_zero = time.time()
                if self.mode in (3, 4):
                    new_mode = 1
                elif self.mode == 2:
                    # Keep freedrive latched until SpaceMouse activity requests teleop.
                    new_mode = 1 if self.curr_sm else 2
                elif self.curr_sm:
                    self.kinesthetic_entry_armed_after_hold = False
                    new_mode = 1
                elif self.kinesthetic_entry_armed_after_hold:
                    # Enter freedrive only after force is released to avoid entering under push.
                    if kinesthetic_released:
                        new_mode = 2
                        self.kinesthetic_armed = False
                        self.kinesthetic_entry_armed_after_hold = False
                    else:
                        new_mode = 1
                elif kinesthetic_requested_held and self.kinesthetic_armed:
                    # Arm entry after force-hold; actual switch happens on release.
                    self.kinesthetic_entry_armed_after_hold = True
                    new_mode = 1
                else:
                    new_mode = 1

        # Set mode and command LEDs
        if new_mode != self.mode:
            self.mode = new_mode
            self.get_logger().info(f'Mode changed to: {self.mode}')
            self.led_pub.publish(String(data=self.mode_colors[self.mode]))
        else:
            self.led_pub.publish(String(data=self.mode_colors[self.mode]))

        if self.enable_freedrive_controller_topic:
            self.freedrive_pub.publish(Bool(data=(self.mode == 2)))

        # Always publish current mode
        mode_msg = Int32()
        mode_msg.data = self.mode
        self.mode_pub.publish(mode_msg)


def main(args=None):
    rclpy.init(args=args)

    node = ModeHandler()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
