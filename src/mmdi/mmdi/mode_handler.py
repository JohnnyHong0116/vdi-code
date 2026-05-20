#!/usr/bin/env python3

"""Mode state machine and LED publisher.

MODE DESCRIPTION:
-1: initialization
 0: idle (red)
 1: teleop (green)
 2: kinesthetic (yellow)
 3: natural demonstration (blue, pulsing blue while optimizing, red when lost)
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

MODE_INIT = -1
MODE_IDLE = 0
MODE_TELEOP = 1
MODE_KINESTHETIC = 2
MODE_NATURAL = 3


class ModeHandler(Node):
    def __init__(self):
        super().__init__('mode_handler')
        self.declare_parameter('spacemouse_input_topic', '/ur7e/teleop_active')
        self.declare_parameter('calibration_ready_topic', '/ft_calibration/ready')
        self.declare_parameter('calibration_required', True)
        self.declare_parameter(
            'natural_optimizing_topic',
            '/natural_handler/optimizing_active',
        )
        self.declare_parameter(
            'natural_target_lost_topic',
            '/natural_handler/target_lost',
        )
        self.declare_parameter('natural_optimizing_timeout_s', 0.6)
        self.declare_parameter('external_force_topic', '/uniforce/force')
        self.declare_parameter('wrench_topic', '/ur7e/ft_internal_calibrated')
        self.declare_parameter('env_wrench_topic', '/ur7e/ft_env_sensor')
        self.declare_parameter('tool_frame', 'tool0')
        self.declare_parameter('force_discrepancy_axis_sign', -1.0)
        self.declare_parameter('kinesthetic_trigger_direction', -1.0)
        self.declare_parameter('force_discrepancy_threshold_z', 4.0)
        self.declare_parameter('force_discrepancy_release_threshold_z', 2.0)
        self.declare_parameter('force_discrepancy_use_abs_with_env', True)
        self.declare_parameter('env_wrench_timeout_s', 0.5)
        self.declare_parameter('kinesthetic_request_hold_s', 2.0)
        self.declare_parameter('tool_contact_debounce_s', 0.5)
        self.declare_parameter('external_force_default', 0.0)
        self.declare_parameter('wrench_tare_samples', 0)
        self.declare_parameter('enable_freedrive_controller_topic', False)
        self.spacemouse_input_topic = self.get_parameter('spacemouse_input_topic').value
        self.calibration_ready_topic = self.get_parameter('calibration_ready_topic').value
        self.calibration_required = bool(
            self.get_parameter('calibration_required').value
        )
        self.natural_optimizing_topic = self.get_parameter(
            'natural_optimizing_topic'
        ).value
        self.natural_target_lost_topic = self.get_parameter(
            'natural_target_lost_topic'
        ).value
        self.natural_optimizing_timeout_s = float(
            self.get_parameter('natural_optimizing_timeout_s').value
        )
        self.external_force_topic = self.get_parameter('external_force_topic').value
        self.wrench_topic = self.get_parameter('wrench_topic').value
        self.env_wrench_topic = self.get_parameter('env_wrench_topic').value
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
        self.force_discrepancy_use_abs_with_env = bool(
            self.get_parameter('force_discrepancy_use_abs_with_env').value
        )
        self.env_wrench_timeout_s = float(
            self.get_parameter('env_wrench_timeout_s').value
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
        self.tool_force_z = 0.0
        self.env_force_z = 0.0
        self.last_env_wrench_sec = None
        self.kinesthetic_armed = True
        self.last_wrench = None
        self.last_no_contact = None
        self.last_contact = None
        self.wrench_bias_z = 0.0
        self.wrench_tare_accum_z = 0.0
        self.wrench_tare_count = 0
        self.wrench_is_tared = (self.wrench_tare_samples == 0)
        self.kinesthetic_request_start = None
        self.kinesthetic_entry_armed_after_hold = False
        self.calibration_ready = not self.calibration_required
        self.natural_optimizing = False
        self.last_natural_optimizing_sec = None
        self.natural_target_lost = False
        self.last_natural_target_lost_sec = None

        # Publishers
        self.led_pub = self.create_publisher(String, '/led_state', 10)
        self.mode_pub = self.create_publisher(Int32, '/mode', 10)
        self.freedrive_pub = self.create_publisher(Bool, '/freedrive_mode_controller/enable_freedrive_mode', 10)

        # Subscribers
        self.contact_sub = self.create_subscription(
            Int32, '/tool_contact', self.store_tool_contact, 10)
        self.sm_sub = self.create_subscription(
            Bool, self.spacemouse_input_topic, self.store_curr_sm, 10)
        self.calibration_ready_sub = self.create_subscription(
            Bool, self.calibration_ready_topic, self.store_calibration_ready, 10)
        self.natural_optimizing_sub = self.create_subscription(
            Bool,
            self.natural_optimizing_topic,
            self.store_natural_optimizing,
            10,
        )
        self.natural_target_lost_sub = self.create_subscription(
            Bool,
            self.natural_target_lost_topic,
            self.store_natural_target_lost,
            10,
        )
        self.uniforce_sub = self.create_subscription(
            Float32, self.external_force_topic, self.store_uni_force, 10)
        self.wrench_sub = self.create_subscription(
            WrenchStamped, self.wrench_topic, self.store_wrench_global, 10)
        self.env_wrench_sub = self.create_subscription(
            WrenchStamped, self.env_wrench_topic, self.store_env_wrench, 10)
        # Direct mode command subscriber
        self.mode_cmd_sub = self.create_subscription(
            Int32, '/mode_cmd', self.handle_mode_cmd, 10)

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Mode state
        self.mode = MODE_INIT
        self.target_mode = None  # For handling direct mode commands
        self.mode_colors = {
            MODE_IDLE: 'r',
            MODE_TELEOP: 'g',
            MODE_KINESTHETIC: 'y',
            MODE_NATURAL: 'b',
        }

        self.get_logger().info('ModeHandler initialized, waiting for startup...')
        self.get_logger().info(
            f'Calibration gate: required={self.calibration_required}, '
            f'topic={self.calibration_ready_topic}'
        )
        self.get_logger().info(f'SpaceMouse input topic: {self.spacemouse_input_topic}')
        self.get_logger().info(
            f'Natural optimizing topic: {self.natural_optimizing_topic}'
        )
        self.get_logger().info(
            f'Natural target lost topic: {self.natural_target_lost_topic}'
        )
        self.get_logger().info(f'Wrench topic: {self.wrench_topic}')
        self.get_logger().info(f'Environment wrench topic: {self.env_wrench_topic}')
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

    def store_calibration_ready(self, msg):
        was_ready = self.calibration_ready
        self.calibration_ready = bool(msg.data)
        if self.calibration_ready and not was_ready:
            self.get_logger().info('FT calibration ready; enabling mode logic.')

    def store_natural_optimizing(self, msg):
        self.natural_optimizing = bool(msg.data)
        self.last_natural_optimizing_sec = self._now_sec()

    def store_natural_target_lost(self, msg):
        self.natural_target_lost = bool(msg.data)
        self.last_natural_target_lost_sec = self._now_sec()

    def store_uni_force(self, msg):
        self.external_force = msg.data

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _wrench_force_z_in_tool(self, msg):
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
                return None

        return float(force_tool[2])

    def store_wrench_global(self, msg):
        self.last_wrench = msg
        raw_tool_force_z = self._wrench_force_z_in_tool(msg)
        if raw_tool_force_z is None:
            return

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

    def store_env_wrench(self, msg):
        env_force_z = self._wrench_force_z_in_tool(msg)
        if env_force_z is None:
            return

        self.env_force_z = env_force_z
        self.last_env_wrench_sec = self._now_sec()

    def store_tool_contact(self, msg):
        self.tool_contact = msg.data

    def handle_mode_cmd(self, msg):
        """Handle direct mode command - request mode change via tick loop"""
        if self.calibration_required and not self.calibration_ready:
            self.get_logger().warn(
                'Ignoring mode command until FT calibration is ready.'
            )
            return
        cmd_mode = msg.data
        if cmd_mode in [MODE_IDLE, MODE_TELEOP, MODE_KINESTHETIC, MODE_NATURAL]:
            self.target_mode = cmd_mode
            mode_names = {
                MODE_IDLE: 'idle',
                MODE_TELEOP: 'teleop',
                MODE_KINESTHETIC: 'freedrive',
                MODE_NATURAL: 'natural',
            }
            self.get_logger().info(f'Mode command received: requesting mode {cmd_mode} ({mode_names.get(cmd_mode, "unknown")})')
        else:
            self.get_logger().warn(f'Invalid mode command: {cmd_mode}')

    def force_discrepancy(self):
        if self.env_wrench_active():
            return self.force_discrepancy_axis_sign * (
                self.tool_force_z - self.env_force_z
            )

        sensed_pull = self.force_discrepancy_axis_sign * self.tool_force_z
        return sensed_pull - self.external_force

    def env_wrench_active(self):
        if self.last_env_wrench_sec is None:
            return False
        return (self._now_sec() - self.last_env_wrench_sec) <= self.env_wrench_timeout_s

    def natural_optimizing_active(self):
        if not self.natural_optimizing:
            return False
        if self.last_natural_optimizing_sec is None:
            return False
        return (
            self._now_sec() - self.last_natural_optimizing_sec
        ) <= self.natural_optimizing_timeout_s

    def natural_target_lost_active(self):
        if not self.natural_target_lost:
            return False
        if self.last_natural_target_lost_sec is None:
            return False
        return (
            self._now_sec() - self.last_natural_target_lost_sec
        ) <= self.natural_optimizing_timeout_s

    def led_command_for_mode(self):
        if self.mode == MODE_NATURAL and self.natural_target_lost_active():
            return 'r'
        if self.mode == MODE_NATURAL and self.natural_optimizing_active():
            return 'p'
        return self.mode_colors.get(self.mode, 'r')

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

    def attached_stably(self):
        if not self.attached():
            self.last_contact = None
            return False
        if self.last_contact is None:
            self.last_contact = time.time()
            return False
        return (time.time() - self.last_contact) >= self.tool_contact_debounce_s

    def _kinesthetic_request_state(self):
        discrepancy = self.force_discrepancy()
        if self.env_wrench_active() and self.force_discrepancy_use_abs_with_env:
            directional_discrepancy = abs(discrepancy)
        else:
            directional_discrepancy = self.kinesthetic_trigger_direction * discrepancy
        requested = directional_discrepancy >= self.force_discrepancy_threshold_z
        released = directional_discrepancy <= self.force_discrepancy_release_threshold_z
        if not self.wrench_is_tared:
            requested = False
            released = True

        if requested:
            if self.kinesthetic_request_start is None:
                self.kinesthetic_request_start = time.time()
            held = (time.time() - self.kinesthetic_request_start) >= self.kinesthetic_request_hold_s
        else:
            self.kinesthetic_request_start = None
            held = False

        return held, released

    def _automatic_mode(self):
        detached = self.detached_stably()

        if self.mode == MODE_INIT:
            return MODE_NATURAL if detached else MODE_TELEOP

        if self.mode == MODE_NATURAL:
            return MODE_TELEOP if self.attached_stably() else MODE_NATURAL

        if detached:
            self.kinesthetic_entry_armed_after_hold = False
            return MODE_NATURAL

        kinesthetic_held, kinesthetic_released = self._kinesthetic_request_state()
        if kinesthetic_released:
            self.kinesthetic_armed = True

        if self.mode == MODE_KINESTHETIC:
            return MODE_TELEOP if self.curr_sm else MODE_KINESTHETIC

        if self.kinesthetic_entry_armed_after_hold:
            if kinesthetic_released:
                self.kinesthetic_armed = False
                self.kinesthetic_entry_armed_after_hold = False
                return MODE_KINESTHETIC
            return MODE_TELEOP

        if kinesthetic_held and self.kinesthetic_armed:
            self.kinesthetic_entry_armed_after_hold = True

        if self.curr_sm:
            return MODE_TELEOP

        return MODE_TELEOP

    def mode_processor(self):
        """Main mode processing loop."""
        if self.calibration_required and not self.calibration_ready:
            if self.mode != MODE_INIT:
                self.mode = MODE_INIT
                self.get_logger().info('Mode changed to: calibration/init')
            self.led_pub.publish(String(data='a'))
            if self.enable_freedrive_controller_topic:
                self.freedrive_pub.publish(Bool(data=False))
            mode_msg = Int32()
            mode_msg.data = MODE_INIT
            self.mode_pub.publish(mode_msg)
            return

        if self.target_mode is not None:
            new_mode = self.target_mode
            self.target_mode = None
            self.kinesthetic_entry_armed_after_hold = False
        else:
            new_mode = self._automatic_mode()

        # Set mode and command LEDs
        if new_mode != self.mode:
            self.mode = new_mode
            self.get_logger().info(f'Mode changed to: {self.mode}')
        self.led_pub.publish(String(data=self.led_command_for_mode()))

        if self.enable_freedrive_controller_topic:
            self.freedrive_pub.publish(Bool(data=(self.mode == MODE_KINESTHETIC)))

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
