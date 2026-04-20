import rclpy
from rclpy.node import Node

import numpy as np
import socket
import struct

from geometry_msgs.msg import WrenchStamped

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from scipy.spatial.transform import Rotation as ScipyR


class WrenchEnvSensor(Node):
    def __init__(self):
        super().__init__("wrench_env_sensor")

        self.get_logger().info("Starting wrench env sensor...")

        self.UDP_IP = "192.168.2.1"
        self.UDP_PORT = 49152

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        self.sock.setblocking(False)
        self.sock.bind(("", self.UDP_PORT))

        init_msg = bytes.fromhex("1234000200000000")
        self.sock.sendto(init_msg, (self.UDP_IP, self.UDP_PORT))

        self.unpacker = struct.Struct("> I I I i i i i i i")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tool_q = None
        self.force_prev = None
        self.torque_prev = None

        self.bias_enabled = False
        self.force_bias = np.zeros(3)
        self.torque_bias = np.zeros(3)

        self.pub = self.create_publisher(WrenchStamped, "/ur7e/ft_env_sensor", 1)

        self.create_timer(0.01, self.update_tool_pose)
        self.create_timer(0.01, self.set_raw_wrench)

        self.R_ftmini_to_tool = ScipyR.from_euler("z", 90.0, degrees=True)

    def update_tool_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                "base",    # 改了这里
                "tool0",   # 改了这里
                rclpy.time.Time()
            )
            self.tool_q = np.array([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ])
        except Exception:
            pass

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
        if self.tool_q is None:
            self.get_logger().warn("No TF yet (base -> tool0). Still reading FTmini.")
            return

        sample = self.read_ftmini_latest()
        if sample is None:
            self.get_logger().warn("No FTmini packets received in this tick")
            return

        force_s, torque_s = sample

        if self.bias_enabled:
            force_s = force_s - self.force_bias
            torque_s = torque_s - self.torque_bias
        else:
            self.force_bias = force_s.copy()
            self.torque_bias = torque_s.copy()
            self.bias_enabled = True

        force_tool = self.R_ftmini_to_tool.apply(force_s)
        torque_tool = self.R_ftmini_to_tool.apply(torque_s)

        R_base_tool = ScipyR.from_quat(self.tool_q)
        force_global = R_base_tool.apply(force_tool)
        torque_global = R_base_tool.apply(torque_tool)

        a = 0.307
        if self.force_prev is not None:
            force_global = a * self.force_prev + (1 - a) * force_global
            torque_global = a * self.torque_prev + (1 - a) * torque_global

        self.force_prev = force_global
        self.torque_prev = torque_global

        out = WrenchStamped()
        out.header.frame_id = "base"
        out.wrench.force.x = float(force_global[0])
        out.wrench.force.y = float(force_global[1])
        out.wrench.force.z = float(force_global[2])
        out.wrench.torque.x = float(torque_global[0])
        out.wrench.torque.y = float(torque_global[1])
        out.wrench.torque.z = float(torque_global[2])

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = WrenchEnvSensor()
    rclpy.spin(node)
    rclpy.shutdown()
