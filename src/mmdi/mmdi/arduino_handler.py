#!/usr/bin/env python3

"""Bridge LED and detach/contact state between ROS 2 and the Arduino Nano."""

from glob import glob
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, String

try:
    import serial
    from serial import SerialException
except Exception:  # pragma: no cover - import failure is handled at runtime
    serial = None
    SerialException = Exception


class ArduinoHandler(Node):
    def __init__(self):
        super().__init__('arduino_handler')

        self.declare_parameter(
            'port',
            '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0',
        )
        self.declare_parameter('baudrate', 9600)
        self.declare_parameter('read_hz', 50.0)
        self.declare_parameter('reconnect_hz', 1.0)
        self.declare_parameter('boot_delay_s', 2.0)
        self.declare_parameter('led_refresh_hz', 2.0)
        self.declare_parameter('default_external_force', 0.0)
        self.declare_parameter('initial_led_state', 'g')

        self.port = str(self.get_parameter('port').value)
        self.baudrate = int(self.get_parameter('baudrate').value)
        read_hz = float(self.get_parameter('read_hz').value)
        reconnect_hz = float(self.get_parameter('reconnect_hz').value)
        self.boot_delay_s = float(self.get_parameter('boot_delay_s').value)
        led_refresh_hz = float(self.get_parameter('led_refresh_hz').value)
        self.default_external_force = float(
            self.get_parameter('default_external_force').value
        )
        self.current_led = str(self.get_parameter('initial_led_state').value)

        self.ser = None
        self.connected_port = None
        self.last_contact = 1

        self.contact_pub = self.create_publisher(Int32, '/tool_contact', 10)
        self.uniforce_raw_pub = self.create_publisher(Float32, '/uniforce/raw', 10)
        self.uniforce_pub = self.create_publisher(Float32, '/uniforce/force', 10)
        self.create_subscription(String, '/led_state', self.on_led_state, 10)

        self.read_timer = self.create_timer(1.0 / max(read_hz, 1.0), self.read_once)
        self.reconnect_timer = self.create_timer(
            1.0 / max(reconnect_hz, 0.1), self.ensure_serial
        )
        self.led_timer = self.create_timer(
            1.0 / max(led_refresh_hz, 0.2), self.refresh_led_state
        )

        self.publish_external_force(self.default_external_force)
        self.publish_contact(self.last_contact)
        self.ensure_serial()

    def _candidate_ports(self):
        candidates = []
        if self.port:
            candidates.append(self.port)
        for path in sorted(glob('/dev/serial/by-id/*')):
            if path not in candidates:
                candidates.append(path)
        for path in sorted(glob('/dev/ttyUSB*')) + sorted(glob('/dev/ttyACM*')):
            if path not in candidates:
                candidates.append(path)
        return candidates

    def ensure_serial(self):
        if serial is None:
            self.get_logger().error('pyserial is not available')
            self.reconnect_timer.cancel()
            return
        if self.ser is not None and self.ser.is_open:
            return

        for port in self._candidate_ports():
            try:
                self.ser = serial.Serial(port, self.baudrate, timeout=0.05)
                self.connected_port = port
                time.sleep(max(self.boot_delay_s, 0.0))
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                self.get_logger().info(f'Connected to Arduino on {port}')
                self._write_led(self.current_led)
                return
            except Exception:
                self.ser = None
                self.connected_port = None

    def close_serial(self):
        if self.ser is None:
            return
        try:
            self.ser.close()
        except Exception:
            pass
        self.ser = None
        self.connected_port = None

    def publish_contact(self, contact_value):
        msg = Int32()
        msg.data = int(contact_value)
        self.contact_pub.publish(msg)

    def publish_external_force(self, force_value):
        msg = Float32()
        msg.data = float(force_value)
        self.uniforce_raw_pub.publish(msg)
        self.uniforce_pub.publish(msg)

    def on_led_state(self, msg: String):
        if not msg.data:
            return
        self.current_led = msg.data[0]
        self._write_led(self.current_led)

    def _write_led(self, led_value):
        if self.ser is None or not self.ser.is_open:
            return
        try:
            self.ser.write(led_value.encode('ascii', errors='ignore')[:1])
            self.ser.flush()
        except Exception as exc:
            self.get_logger().warn(f'Failed to write LED state to Arduino: {exc}')
            self.close_serial()

    def refresh_led_state(self):
        self._write_led(self.current_led)

    def read_once(self):
        if self.ser is None or not self.ser.is_open:
            self.publish_external_force(self.default_external_force)
            return

        try:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
        except (SerialException, OSError) as exc:
            self.get_logger().warn(f'Lost Arduino serial connection: {exc}')
            self.close_serial()
            self.publish_external_force(self.default_external_force)
            return

        if not line:
            self.publish_external_force(self.default_external_force)
            return

        try:
            force_text, contact_text = [field.strip() for field in line.split(',', 1)]
            force_value = float(force_text)
            contact_value = int(contact_text)
        except Exception:
            self.get_logger().debug(f'Ignoring malformed Arduino line: {line!r}')
            self.publish_external_force(self.default_external_force)
            return

        self.publish_external_force(force_value)
        if contact_value != self.last_contact:
            self.last_contact = contact_value
        self.publish_contact(contact_value)

    def destroy_node(self):
        self.close_serial()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArduinoHandler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
