#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32, String


class FreedriveController(Node):
    def __init__(self):
        super().__init__('freedrive_controller')

        self.declare_parameter('mode_topic', '/mode')
        self.declare_parameter('script_topic', '/urscript_interface/script_command')
        self.declare_parameter('freedrive_mode_value', 2)
        self.declare_parameter('check_rate_hz', 50.0)
        self.declare_parameter('activation_delay_s', 0.20)
        self.declare_parameter('publish_end_on_exit', True)

        self.mode_topic = str(self.get_parameter('mode_topic').value)
        self.script_topic = str(self.get_parameter('script_topic').value)
        self.freedrive_mode_value = int(self.get_parameter('freedrive_mode_value').value)
        self.check_rate_hz = float(self.get_parameter('check_rate_hz').value)
        self.activation_delay_s = float(self.get_parameter('activation_delay_s').value)
        self.publish_end_on_exit = bool(self.get_parameter('publish_end_on_exit').value)

        self.curr_mode = -1
        self.was_active = False
        self.enter_requested_at = None

        self.script_pub = self.create_publisher(String, self.script_topic, 1)
        self.mode_sub = self.create_subscription(Int32, self.mode_topic, self.on_mode, 10)
        self.timer = self.create_timer(1.0 / max(self.check_rate_hz, 1.0), self.tick)

        self.get_logger().info(f'Freedrive controller mode topic: {self.mode_topic}')
        self.get_logger().info(f'Freedrive controller script topic: {self.script_topic}')
        self.get_logger().info(f'Freedrive active mode value: {self.freedrive_mode_value}')
        self.get_logger().info(f'Freedrive activation delay: {self.activation_delay_s:.2f}s')

    def on_mode(self, msg: Int32):
        self.curr_mode = int(msg.data)

    def publish_script(self, cmd: str):
        self.script_pub.publish(String(data=cmd))

    @staticmethod
    def enter_freedrive_script():
        # Keep freedrive alive in one persistent program instead of command spam.
        return (
            "def mmdi_freedrive_loop():\n"
            "  while True:\n"
            "    freedrive_mode()\n"
            "    sync()\n"
            "  end\n"
            "end\n"
            "mmdi_freedrive_loop()\n"
        )

    @staticmethod
    def exit_freedrive_script():
        return (
            "def mmdi_freedrive_stop():\n"
            "  end_freedrive_mode()\n"
            "  stopl(0.5)\n"
            "end\n"
            "mmdi_freedrive_stop()\n"
        )

    def tick(self):
        now_s = self.get_clock().now().nanoseconds / 1e9
        active = (self.curr_mode == self.freedrive_mode_value)
        if active:
            if self.was_active:
                return

            if self.enter_requested_at is None:
                self.enter_requested_at = now_s
                return

            if (now_s - self.enter_requested_at) < max(self.activation_delay_s, 0.0):
                return

            self.get_logger().info('Entering freedrive mode (URScript).')
            self.publish_script(self.enter_freedrive_script())
            self.was_active = True
            self.enter_requested_at = None
        elif self.was_active and self.publish_end_on_exit:
            self.get_logger().info('Exiting freedrive mode (URScript).')
            self.publish_script(self.exit_freedrive_script())
            self.was_active = False
            self.enter_requested_at = None
        else:
            self.was_active = False
            self.enter_requested_at = None

    def destroy_node(self):
        if self.was_active and self.publish_end_on_exit:
            try:
                self.publish_script(self.exit_freedrive_script())
            except Exception:
                pass
        super().destroy_node()


def main():
    rclpy.init()
    node = FreedriveController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
