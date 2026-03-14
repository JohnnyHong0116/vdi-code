#!/usr/bin/env python3

"""Mode (state machine) handling and LED publishing
Converted to ROS 2 from ROS 1
Original Author: Mike Hagenow
Last Updated: 6/11/2024

MODE DESCRIPTION:
-1: initialization
 0: no input (red)
 1: teleop (green)
 2: kinesthetic (yellow)
 3: natural (blue)
 4: pre_natural (strobe blue)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Bool, Float32
from geometry_msgs.msg import WrenchStamped
from scipy.spatial.transform import Rotation as ScipyR
import subprocess
import time

from tf2_ros import Buffer, TransformListener
from rclpy.time import Time


class ModeHandler(Node):
    def __init__(self):
        super().__init__('mode_handler')
        self.declare_parameter('spacemouse_input_topic', '/ur7e/teleop_active')
        self.spacemouse_input_topic = self.get_parameter('spacemouse_input_topic').value

        # Initial state
        self.tool_contact = 1
        self.tool_q = None
        self.curr_sm = False
        self.uniforce = None
        self.uniforce_time = None
        self.odom_seen = False
        self.wrench_global = None
        self.discrepancy_samps = 0

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
            Float32, '/uniforce/force', self.store_uni_force, 10)
        self.wrench_sub = self.create_subscription(
            WrenchStamped, '/wrench_global', self.store_wrench_global, 10)
        # Direct mode command subscriber
        self.mode_cmd_sub = self.create_subscription(
            Int32, '/mode_cmd', self.handle_mode_cmd, 10)

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timing
        self.start_mode_zero = time.time()
        self.last_no_contact = None
        self.start_mode_four = None

        # Mode state
        self.mode = -1  # Start with initialization
        self.target_mode = None  # For handling direct mode commands
        # Index maps to mode: 0=red, 1=green, 2=yellow, 3=blue, 4=strobe
        self.mode_colors = {0: 'r', 1: 'g', 2: 'y', 3: 'b', 4: 's'}

        self.get_logger().info('ModeHandler initialized, waiting for startup...')
        self.get_logger().info(f'SpaceMouse input topic: {self.spacemouse_input_topic}')

        # Defer start so TF and other nodes can initialize
        self._init_timer = self.create_timer(2.0, self._start_processing)

    def _start_processing(self):
        self._init_timer.cancel()
        # Main processing timer at 5 Hz
        self.timer = self.create_timer(0.2, self.mode_processor)

    def store_curr_sm(self, msg):
        self.curr_sm = msg.data

    def store_uni_force(self, msg):
        self.uniforce = msg.data
        self.uniforce_time = self.get_clock().now()

    def store_wrench_global(self, msg):
        if self.uniforce_time is not None:
            time_diff = (self.get_clock().now() - self.uniforce_time).nanoseconds / 1e9
            if time_diff < 0.01:
                self.wrench_global = msg.wrench

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
        """Check for force discrepancy between uniforce and F/T sensor"""
        if self.tool_q is not None and self.uniforce is not None and self.wrench_global is not None:
            R_tool_global = ScipyR.from_quat(self.tool_q)
            F_global = np.array([
                self.wrench_global.force.x,
                self.wrench_global.force.y,
                self.wrench_global.force.z
            ])
            F_local = R_tool_global.inv().apply(F_global)

            # Use sample counting to get around filter induced debouncing
            if (F_local[2] - self.uniforce) > 10:  # Only when pulling down
                self.discrepancy_samps += 1
            else:
                self.discrepancy_samps = 0

            if self.discrepancy_samps >= 4:  # 4/5 seconds
                return True

        return False

    def mode_processor(self):
        """Main mode processing loop"""
        # Check tool location via TF
        try:
            trans = self.tf_buffer.lookup_transform('base', 'tool0', Time())
            self.tool_q = np.array([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ])
        except Exception as e:
            self.get_logger().debug(f'TF lookup failed: {e}')

        # Mode transitions
        new_mode = self.mode

        # Check for direct mode command override
        if self.target_mode is not None:
            new_mode = self.target_mode
            self.target_mode = None  # Clear after using

        self.get_logger().debug(f'Current MODE: {self.mode}')

        if self.mode == -1:
            # Initialization -> idle
            new_mode = 0

        elif self.mode == 0:
            # Idle mode - can transition to 1, 2, or 4
            if (time.time() - self.start_mode_zero) > 2.0:  # Debouncing
                # Switch to Teleoperation (1)
                if self.curr_sm:
                    new_mode = 1

                # Switch to Kinesthetic (2)
                if self.force_discrepancy():
                    new_mode = 2
                    self.freedrive_pub.publish(Bool(data=True))

                # Switch to pre-Natural (4)
                if self.tool_contact == 0:
                    if self.last_no_contact is None:
                        self.last_no_contact = time.time()
                    if (time.time() - self.last_no_contact) > 1:
                        new_mode = 4
                        self.start_mode_four = time.time()
                else:
                    self.last_no_contact = None

        elif self.mode == 1:
            # Teleoperation mode
            # Change LED color based on force intensity
            if self.wrench_global is not None:
                F_global = np.array([
                    self.wrench_global.force.x,
                    self.wrench_global.force.y,
                    self.wrench_global.force.z
                ])
                if np.linalg.norm(F_global) > 10:
                    led_val = str(int((25 - np.linalg.norm(F_global)) / 3))
                    self.led_pub.publish(String(data=led_val))
                else:
                    self.led_pub.publish(String(data=self.mode_colors[self.mode]))

                # Play sound for force sensor
                if self.uniforce is not None and abs(self.uniforce) > 0.75 * 9.8:
                    freq = 66 * (abs(self.uniforce) - 0.7 * 9.8)
                    subprocess.Popen(['play', '-nq', '-t', 'alsa', 'synth', '0.2', 'sine', str(freq)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Switch to Kinesthetic (2)
                if self.force_discrepancy():
                    new_mode = 2
                    self.freedrive_pub.publish(Bool(data=True))

                # Too much force also switches to kinesthetic
                if self.uniforce is not None:
                    if abs(self.uniforce) > 25 or np.linalg.norm(F_global) > 25:
                        new_mode = 2
                        self.freedrive_pub.publish(Bool(data=True))

            # Switch to pre-Natural (4)
            if self.tool_contact == 0:
                if self.last_no_contact is None:
                    self.last_no_contact = time.time()
                if (time.time() - self.last_no_contact) > 1:
                    new_mode = 4
                    self.start_mode_four = time.time()
            else:
                self.last_no_contact = None

        elif self.mode == 2:
            # Kinesthetic mode - continuously publish freedrive enable
            self.freedrive_pub.publish(Bool(data=True))

            # NOTE: Automatic switching to teleop disabled - use button to switch modes
            # Switch to Teleoperation (1)
            # if self.curr_sm:
            #     new_mode = 1
            #     self.freedrive_pub.publish(Bool(data=False))

            # Switch to pre-Natural (4)
            if self.tool_contact == 0:
                if self.last_no_contact is None:
                    self.last_no_contact = time.time()
                if (time.time() - self.last_no_contact) > 2:
                    new_mode = 4
                    self.freedrive_pub.publish(Bool(data=False))
                    self.start_mode_four = time.time()
            else:
                self.last_no_contact = None

        elif self.mode == 3:
            # Natural mode
            # Switch to idle (0)
            if self.tool_contact == 1:
                new_mode = 0
                self.start_mode_zero = time.time()

        elif self.mode == 4:
            # Pre-natural mode
            if self.odom_seen:
                new_mode = 3

            # Switch to idle (0)
            if self.start_mode_four is not None:
                if self.tool_contact == 1 and (time.time() - self.start_mode_four) > 5:
                    new_mode = 0
                    self.start_mode_zero = time.time()

        # Set mode and command LEDs
        if new_mode != self.mode:
            self.mode = new_mode
            self.get_logger().info(f'Mode changed to: {self.mode}')
            self.led_pub.publish(String(data=self.mode_colors[self.mode]))

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
