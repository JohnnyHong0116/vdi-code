import rclpy
from rclpy.node import Node

import numpy as np
import socket
import struct

from geometry_msgs.msg import WrenchStamped
from rclpy.time import Time

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from scipy.spatial.transform import Rotation as ScipyR


class WrenchEnvSensor(Node):
    def __init__(self):
        super().__init__("wrench_env_sensor")

        self.get_logger().info("Starting wrench env sensor...")

        self.declare_parameter("sensor_ip", "192.168.2.1")
        self.declare_parameter("sensor_port", 49152)
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("output_topic", "/ur7e/ft_env_sensor_raw")
        self.declare_parameter("output_frame", "base")
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("tare_on_startup", False)

        self.sensor_ip = str(self.get_parameter("sensor_ip").value)
        self.sensor_port = int(self.get_parameter("sensor_port").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tool_frame = str(self.get_parameter("tool_frame").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.output_frame = str(self.get_parameter("output_frame").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.tare_on_startup = bool(self.get_parameter("tare_on_startup").value)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        self.sock.setblocking(False)
        self.sock.bind(("", self.sensor_port))

        init_msg = bytes.fromhex("1234000200000000")
        self.sock.sendto(init_msg, (self.sensor_ip, self.sensor_port))

        self.unpacker = struct.Struct("> I I I i i i i i i")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tool_q = None
        self.last_tf_error = None
        self.force_prev = None
        self.torque_prev = None

        self.bias_enabled = False
        self.force_bias = np.zeros(3)
        self.torque_bias = np.zeros(3)

        self.pub = self.create_publisher(WrenchStamped, self.output_topic, 1)

        period = 1.0 / max(self.rate_hz, 1e-6)
        self.create_timer(period, self.update_tool_pose)
        self.create_timer(period, self.set_raw_wrench)

        self.R_ftmini_to_tool = ScipyR.from_euler("z", 90.0, degrees=True)
        self.get_logger().info(
            f"Publishing {self.output_topic} in {self.output_frame}; "
            f"TF {self.base_frame} <- {self.tool_frame}; "
            f"FTmini {self.sensor_ip}:{self.sensor_port}"
        )

    def update_tool_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tool_frame,
                Time(),
            )
            self.tool_q = np.array([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ])
            self.last_tf_error = None
        except Exception as exc:
            self.tool_q = None
            self.last_tf_error = str(exc)

    def read_ftmini_latest(self):
        latest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
                latest = data
            except BlockingIOError:
                break
            except Exception:
                break

        if latest is None:
            return None

        unpacked = self.unpacker.unpack(bytearray(latest))
        ft = unpacked[3:]
        ft = [x / 1_000_000.0 for x in ft]

        force = np.array(ft[:3], dtype=float)
        torque = np.array(ft[3:], dtype=float)
        return force, torque

    def set_raw_wrench(self):
        sample = self.read_ftmini_latest()
        if sample is None:
            self.get_logger().warn(
                "No FTmini packets received",
                throttle_duration_sec=1.0,
            )
            return

        if self.tool_q is None:
            detail = f": {self.last_tf_error}" if self.last_tf_error else ""
            self.get_logger().warn(
                f"No TF yet ({self.base_frame} <- {self.tool_frame}){detail}. "
                "Waiting before publishing FTmini wrench.",
                throttle_duration_sec=1.0,
            )
            return

        force_s, torque_s = sample

        if self.tare_on_startup:
            if self.bias_enabled:
                force_s = force_s - self.force_bias
                torque_s = torque_s - self.torque_bias
            else:
                self.force_bias = force_s.copy()
                self.torque_bias = torque_s.copy()
                self.bias_enabled = True

        force_tool = self.R_ftmini_to_tool.apply(force_s)
        torque_tool = self.R_ftmini_to_tool.apply(torque_s)

        if self.output_frame == self.base_frame:
            R_base_tool = ScipyR.from_quat(self.tool_q)
            force_out = R_base_tool.apply(force_tool)
            torque_out = R_base_tool.apply(torque_tool)
        elif self.output_frame == self.tool_frame:
            force_out = force_tool
            torque_out = torque_tool
        else:
            self.get_logger().warn(
                f"Unsupported output_frame '{self.output_frame}', using "
                f"{self.base_frame}.",
                throttle_duration_sec=2.0,
            )
            R_base_tool = ScipyR.from_quat(self.tool_q)
            force_out = R_base_tool.apply(force_tool)
            torque_out = R_base_tool.apply(torque_tool)

        a = 0.307
        if self.force_prev is not None:
            force_out = a * self.force_prev + (1 - a) * force_out
            torque_out = a * self.torque_prev + (1 - a) * torque_out

        self.force_prev = force_out
        self.torque_prev = torque_out

        out = WrenchStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.output_frame
        out.wrench.force.x = float(force_out[0])
        out.wrench.force.y = float(force_out[1])
        out.wrench.force.z = float(force_out[2])
        out.wrench.torque.x = float(torque_out[0])
        out.wrench.torque.y = float(torque_out[1])
        out.wrench.torque.z = float(torque_out[2])

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = WrenchEnvSensor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sock.close()
        node.destroy_node()
        rclpy.shutdown()
