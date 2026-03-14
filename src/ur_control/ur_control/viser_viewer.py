#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node

import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

import viser

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except Exception:  # pragma: no cover - optional dependency path
    load_robot_description = None

try:
    from viser.extras import ViserUrdf
except Exception:  # pragma: no cover - optional dependency path
    ViserUrdf = None


class ViserViewer(Node):
    def __init__(self):
        super().__init__('viser_viewer')

        # Parameters
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('tool_frame', 'tool0')
        self.declare_parameter('desired_pose_topic', '/ur7e/desired_pose')
        self.declare_parameter('rate_hz', 450.0)
        self.declare_parameter('viser_host', '0.0.0.0')
        self.declare_parameter('viser_port', 8080)
        self.declare_parameter('show_actual_pose', True)
        self.declare_parameter('show_desired_pose', True)
        self.declare_parameter('grid_size', 2.0)
        self.declare_parameter('axes_length', 0.2)
        self.declare_parameter('axes_radius', 0.01)
        self.declare_parameter('robot_description', 'ur5e_description')
        self.declare_parameter('robot_urdf_path', '')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('load_meshes', True)
        self.declare_parameter('load_collision_meshes', False)
        self.declare_parameter('show_robot_visual', True)
        self.declare_parameter('show_robot_collision', False)
        self.declare_parameter('align_robot_to_tf', True)
        self.declare_parameter('robot_root_frame', '')

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.desired_pose_topic = self.get_parameter('desired_pose_topic').value
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.viser_host = self.get_parameter('viser_host').value
        self.viser_port = int(self.get_parameter('viser_port').value)
        self.show_actual = bool(self.get_parameter('show_actual_pose').value)
        self.show_desired = bool(self.get_parameter('show_desired_pose').value)
        self.grid_size = float(self.get_parameter('grid_size').value)
        self.axes_length = float(self.get_parameter('axes_length').value)
        self.axes_radius = float(self.get_parameter('axes_radius').value)
        self.robot_description = self.get_parameter('robot_description').value
        self.robot_urdf_path = self.get_parameter('robot_urdf_path').value
        self.joint_state_topic = self.get_parameter('joint_state_topic').value
        self.load_meshes = bool(self.get_parameter('load_meshes').value)
        self.load_collision_meshes = bool(
            self.get_parameter('load_collision_meshes').value
        )
        self.show_robot_visual = bool(
            self.get_parameter('show_robot_visual').value
        )
        self.show_robot_collision = bool(
            self.get_parameter('show_robot_collision').value
        )
        self.align_robot_to_tf = bool(
            self.get_parameter('align_robot_to_tf').value
        )
        self.robot_root_frame = self.get_parameter('robot_root_frame').value

        # Viser server + scene
        self.server = viser.ViserServer(
            host=self.viser_host,
            port=self.viser_port,
            label='ur_control',
        )
        self.scene = self.server.scene

        self.robot_root_handle = self.scene.add_frame(
            '/robot',
            show_axes=False,
        )
        self.viser_urdf = None
        self.robot_joint_names = []
        self.last_joint_positions = {}
        self.warned_joint_map = False
        self.warned_robot_tf = False
        self.warned_desired_tf = False

        self._init_robot_model()

        self.scene.add_grid(
            'grid',
            width=self.grid_size,
            height=self.grid_size,
            plane='xy',
            cell_size=0.25,
            section_size=1.0,
        )

        self.base_handle = self.scene.add_frame(
            'base',
            show_axes=True,
            axes_length=self.axes_length,
            axes_radius=self.axes_radius,
        )

        self.tool_handle = None
        if self.show_actual:
            self.tool_handle = self.scene.add_frame(
                'tool',
                show_axes=True,
                axes_length=self.axes_length,
                axes_radius=self.axes_radius,
                origin_color=(0, 255, 0),
            )

        self.des_handle = None
        if self.show_desired:
            self.des_handle = self.scene.add_frame(
                'desired',
                show_axes=True,
                axes_length=self.axes_length,
                axes_radius=self.axes_radius,
                origin_color=(255, 0, 0),
            )

        self.des_pos = None
        self.des_q = None
        self.desired_pose_sub = None
        self.joint_state_sub = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.warned_tf = False

        if self.show_desired:
            self.desired_pose_sub = self.create_subscription(
                PoseStamped,
                self.desired_pose_topic,
                self.on_desired_pose,
                1,
            )

        if self.viser_urdf is not None:
            self.joint_state_sub = self.create_subscription(
                JointState,
                self.joint_state_topic,
                self.on_joint_state,
                1,
            )

        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

        self.get_logger().info(
            f"Viser server at http://{self.viser_host}:{self.viser_port} "
            f"(base={self.base_frame}, tool={self.tool_frame}, "
            f"robot={self.robot_description})"
        )

    def _init_robot_model(self):
        if ViserUrdf is None:
            self.get_logger().warn(
                "URDF view disabled: 'viser.extras.ViserUrdf' not available."
            )
            return

        urdf_or_path = None
        loaded_from = ''

        if self.robot_urdf_path:
            urdf_or_path = self.robot_urdf_path
            loaded_from = self.robot_urdf_path
        elif load_robot_description is None:
            self.get_logger().warn(
                "URDF view disabled: install 'robot_descriptions' and "
                "'yourdfpy' to enable robot model rendering."
            )
            return
        else:
            description_candidates = [self.robot_description]
            if self.robot_description == 'ur5e_description':
                description_candidates.append('ur5_description')

            load_error = None
            for description in description_candidates:
                try:
                    urdf_or_path = load_robot_description(
                        description,
                        load_meshes=self.load_meshes,
                        build_scene_graph=self.load_meshes,
                        load_collision_meshes=self.load_collision_meshes,
                        build_collision_scene_graph=self.load_collision_meshes,
                    )
                    loaded_from = description
                    if description != self.robot_description:
                        self.get_logger().warn(
                            f"'{self.robot_description}' not available; "
                            f"using '{description}' instead."
                        )
                    break
                except Exception as ex:
                    load_error = ex

            if urdf_or_path is None:
                self.get_logger().warn(
                    f"Failed to load '{self.robot_description}': {load_error}"
                )
                return

        try:
            self.viser_urdf = ViserUrdf(
                self.server,
                urdf_or_path=urdf_or_path,
                root_node_name='/robot',
                load_meshes=self.load_meshes,
                load_collision_meshes=self.load_collision_meshes,
                collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
            )
            self.viser_urdf.show_visual = (
                self.load_meshes and self.show_robot_visual
            )
            self.viser_urdf.show_collision = (
                self.load_collision_meshes and self.show_robot_collision
            )

            self.robot_joint_names = list(
                self.viser_urdf.get_actuated_joint_limits().keys()
            )
            if self.robot_joint_names:
                self.viser_urdf.update_cfg(
                    np.zeros(len(self.robot_joint_names), dtype=float)
                )

            if not self.robot_root_frame:
                scene = (
                    self.viser_urdf._urdf.scene
                    or self.viser_urdf._urdf.collision_scene
                )
                if scene is not None:
                    self.robot_root_frame = str(scene.graph.base_frame)

            self.get_logger().info(
                f"Loaded URDF model '{loaded_from}' "
                f"({len(self.robot_joint_names)} actuated joints)."
            )
            if self.robot_root_frame:
                self.get_logger().info(
                    f"Robot model root frame: {self.robot_root_frame}"
                )
        except Exception as ex:
            self.viser_urdf = None
            self.robot_joint_names = []
            self.get_logger().warn(f"Failed to build URDF viewer: {ex}")

    @staticmethod
    def quat_xyzw_to_wxyz(q):
        return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))

    @staticmethod
    def quat_xyzw_normalize(q):
        n = float(np.linalg.norm(q))
        if n < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q / n

    @staticmethod
    def quat_xyzw_multiply(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=float,
        )

    @staticmethod
    def rotate_vec_by_quat_xyzw(v, q):
        q_xyz = q[:3]
        uv = np.cross(q_xyz, v)
        uuv = np.cross(q_xyz, uv)
        return v + 2.0 * (q[3] * uv + uuv)

    def update_handle(self, handle, pos, quat_xyzw):
        quat_xyzw = self.quat_xyzw_normalize(quat_xyzw)
        handle.position = (float(pos[0]), float(pos[1]), float(pos[2]))
        handle.wxyz = self.quat_xyzw_to_wxyz(quat_xyzw)

    def transform_pose_to_base(self, pos, quat_xyzw, source_frame):
        if not source_frame or source_frame == self.base_frame:
            return pos, quat_xyzw
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                rclpy.time.Time(),
            )
        except Exception as ex:
            if not self.warned_desired_tf:
                self.get_logger().warn(
                    f"TF lookup failed ({self.base_frame} <- {source_frame}): "
                    f"{ex}"
                )
                self.warned_desired_tf = True
            return None

        t = np.array(
            [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ],
            dtype=float,
        )
        q_tf = np.array(
            [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ],
            dtype=float,
        )
        q_tf = self.quat_xyzw_normalize(q_tf)
        pos_out = t + self.rotate_vec_by_quat_xyzw(pos, q_tf)
        quat_out = self.quat_xyzw_multiply(q_tf, quat_xyzw)
        return pos_out, self.quat_xyzw_normalize(quat_out)

    def update_robot_alignment(self):
        if (
            self.viser_urdf is None
            or not self.align_robot_to_tf
            or not self.robot_root_frame
        ):
            return
        if self.robot_root_frame == self.base_frame:
            self.robot_root_handle.position = (0.0, 0.0, 0.0)
            self.robot_root_handle.wxyz = (1.0, 0.0, 0.0, 0.0)
            return

        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.robot_root_frame,
                rclpy.time.Time(),
            )
        except Exception as ex:
            if not self.warned_robot_tf:
                self.get_logger().warn(
                    f"TF lookup failed ({self.base_frame} <- "
                    f"{self.robot_root_frame}): {ex}"
                )
                self.warned_robot_tf = True
            return

        pos = np.array(
            [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ],
            dtype=float,
        )
        quat = np.array(
            [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ],
            dtype=float,
        )
        self.update_handle(self.robot_root_handle, pos, quat)

    def on_joint_state(self, msg: JointState):
        if self.viser_urdf is None or not self.robot_joint_names:
            return
        if not msg.name or not msg.position:
            return

        joint_map = dict(zip(msg.name, msg.position))
        q = []
        found_any = False
        for joint_name in self.robot_joint_names:
            value = joint_map.get(joint_name)
            if value is not None:
                self.last_joint_positions[joint_name] = float(value)
                found_any = True
            else:
                value = self.last_joint_positions.get(joint_name, 0.0)
            q.append(float(value))

        if not found_any:
            if not self.warned_joint_map:
                self.get_logger().warn(
                    f"No joint name overlap on topic {self.joint_state_topic}."
                )
                self.warned_joint_map = True
            return

        self.viser_urdf.update_cfg(np.array(q, dtype=float))

    def on_desired_pose(self, msg: PoseStamped):
        pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
        )
        quat = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=float,
        )
        source_frame = msg.header.frame_id.strip() if msg.header.frame_id else ''
        transformed = self.transform_pose_to_base(pos, quat, source_frame)
        if transformed is None:
            return
        self.des_pos, self.des_q = transformed
        if self.des_handle is not None:
            self.update_handle(self.des_handle, self.des_pos, self.des_q)

    def tick(self):
        self.update_robot_alignment()

        if self.tool_handle is None:
            return

        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tool_frame,
                rclpy.time.Time(),
            )
        except Exception as ex:
            if not self.warned_tf:
                self.get_logger().warn(
                    f"TF lookup failed ({self.base_frame} -> "
                    f"{self.tool_frame}): {ex}"
                )
                self.warned_tf = True
            return

        pos = np.array(
            [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ],
            dtype=float,
        )
        quat = np.array(
            [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ],
            dtype=float,
        )
        self.update_handle(self.tool_handle, pos, quat)


def main():
    rclpy.init()
    node = ViserViewer()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.server.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
