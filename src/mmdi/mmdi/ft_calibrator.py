#!/usr/bin/env python3

"""Startup force/torque calibration gate.

The calibrator waits until the tool is attached, gathers samples from the
internal and external FT streams, estimates the startup gravity/bias wrench,
then republishes calibrated wrenches.  Downstream controllers should consume
the calibrated topics and wait for /ft_calibration/ready before allowing robot
inputs.
"""

from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation as ScipyR
from std_msgs.msg import Bool, Int32
from tf2_ros import Buffer, TransformListener


def _vec3_to_list(vec: np.ndarray):
    return [float(x) for x in np.round(vec, 6)]


class FTCalibrator(Node):
    def __init__(self):
        super().__init__("ft_calibrator")

        self.declare_parameter("base_frame", "base")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter(
            "internal_raw_topic",
            "/force_torque_sensor_broadcaster/wrench",
        )
        self.declare_parameter("internal_raw_frame", "tool0")
        self.declare_parameter("internal_use_msg_frame_id", True)
        self.declare_parameter("internal_source_frame_bias", False)
        self.declare_parameter("internal_force_bias_axes", [True, True, True])
        self.declare_parameter("internal_force_deadband_n", 1.5)
        self.declare_parameter("internal_torque_deadband_nm", 0.05)
        self.declare_parameter("internal_output_topic", "/ur7e/ft_internal_calibrated")
        self.declare_parameter("external_raw_topic", "/ur7e/ft_env_sensor_raw")
        self.declare_parameter("external_raw_frame", "base")
        self.declare_parameter("external_source_frame_bias", False)
        self.declare_parameter("external_output_topic", "/ur7e/ft_env_sensor")
        self.declare_parameter("tool_contact_topic", "/tool_contact")
        self.declare_parameter("ready_topic", "/ft_calibration/ready")
        self.declare_parameter("sample_count", 200)
        self.declare_parameter("external_required", True)
        self.declare_parameter("require_tool_attached", True)
        self.declare_parameter("stability_check_enabled", True)
        self.declare_parameter("force_stddev_max_n", 1.0)
        self.declare_parameter("torque_stddev_max_nm", 0.08)
        self.declare_parameter("status_publish_hz", 5.0)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tool_frame = str(self.get_parameter("tool_frame").value)
        self.internal_raw_topic = str(self.get_parameter("internal_raw_topic").value)
        self.internal_raw_frame = str(self.get_parameter("internal_raw_frame").value)
        self.internal_use_msg_frame_id = bool(
            self.get_parameter("internal_use_msg_frame_id").value
        )
        self.internal_source_frame_bias = bool(
            self.get_parameter("internal_source_frame_bias").value
        )
        self.internal_force_bias_axes = self._axis_mask_parameter(
            "internal_force_bias_axes"
        )
        self.internal_force_deadband_n = max(
            0.0,
            float(self.get_parameter("internal_force_deadband_n").value),
        )
        self.internal_torque_deadband_nm = max(
            0.0,
            float(self.get_parameter("internal_torque_deadband_nm").value),
        )
        self.internal_output_topic = str(
            self.get_parameter("internal_output_topic").value
        )
        self.external_raw_topic = str(self.get_parameter("external_raw_topic").value)
        self.external_raw_frame = str(self.get_parameter("external_raw_frame").value)
        self.external_source_frame_bias = bool(
            self.get_parameter("external_source_frame_bias").value
        )
        self.external_output_topic = str(
            self.get_parameter("external_output_topic").value
        )
        self.tool_contact_topic = str(self.get_parameter("tool_contact_topic").value)
        self.ready_topic = str(self.get_parameter("ready_topic").value)
        self.sample_count = max(1, int(self.get_parameter("sample_count").value))
        self.external_required = bool(self.get_parameter("external_required").value)
        self.require_tool_attached = bool(
            self.get_parameter("require_tool_attached").value
        )
        self.stability_check_enabled = bool(
            self.get_parameter("stability_check_enabled").value
        )
        self.force_stddev_max_n = max(
            0.0,
            float(self.get_parameter("force_stddev_max_n").value),
        )
        self.torque_stddev_max_nm = max(
            0.0,
            float(self.get_parameter("torque_stddev_max_nm").value),
        )
        status_publish_hz = float(self.get_parameter("status_publish_hz").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tool_attached = False
        self.have_contact_msg = False
        self.ready = False
        self.logged_waiting = False
        self.logged_ready = False
        self.rejected_internal_windows = 0
        self.rejected_external_windows = 0

        self.internal_samples = []
        self.external_samples = []
        self.internal_force_bias = np.zeros(3, dtype=float)
        self.internal_torque_bias = np.zeros(3, dtype=float)
        self.external_force_bias = np.zeros(3, dtype=float)
        self.external_torque_bias = np.zeros(3, dtype=float)

        self.ready_pub = self.create_publisher(Bool, self.ready_topic, 10)
        self.internal_pub = self.create_publisher(
            WrenchStamped, self.internal_output_topic, 10
        )
        self.external_pub = self.create_publisher(
            WrenchStamped, self.external_output_topic, 10
        )

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            Int32, self.tool_contact_topic, self.on_tool_contact, 10
        )
        self.create_subscription(
            WrenchStamped,
            self.internal_raw_topic,
            self.on_internal_wrench,
            qos_sensor,
        )
        self.create_subscription(
            WrenchStamped,
            self.external_raw_topic,
            self.on_external_wrench,
            qos_sensor,
        )

        self.create_timer(
            1.0 / max(status_publish_hz, 0.2),
            self.publish_ready_state,
        )

        self.get_logger().info(
            "FT calibration waiting for attached tool and "
            f"{self.sample_count} samples; internal={self.internal_raw_topic} -> "
            f"{self.internal_output_topic} ({self.internal_raw_frame}, "
            f"use_msg_frame_id={self.internal_use_msg_frame_id}, "
            f"source_frame_bias={self.internal_source_frame_bias}, "
            f"force_bias_axes={self.internal_force_bias_axes.tolist()}, "
            f"force_deadband={self.internal_force_deadband_n:.3f} N), "
            f"external={self.external_raw_topic} -> {self.external_output_topic} "
            f"({self.external_raw_frame}, "
            f"source_frame_bias={self.external_source_frame_bias}), "
            f"external_required={self.external_required}"
        )
        if self.stability_check_enabled:
            self.get_logger().info(
                "FT calibration stability gate: "
                f"force_stddev_max={self.force_stddev_max_n:.3f} N, "
                f"torque_stddev_max={self.torque_stddev_max_nm:.3f} Nm"
            )

    def on_tool_contact(self, msg: Int32):
        self.have_contact_msg = True
        self.tool_attached = int(msg.data) != 0

    def _can_calibrate(self) -> bool:
        if not self.require_tool_attached:
            return True
        return self.have_contact_msg and self.tool_attached

    def _axis_mask_parameter(self, name: str) -> np.ndarray:
        values = list(self.get_parameter(name).value)
        if len(values) != 3:
            self.get_logger().warn(
                f"{name} must contain 3 booleans; using [True, True, True]."
            )
            values = [True, True, True]
        return np.asarray([bool(v) for v in values], dtype=bool)

    @staticmethod
    def _component_deadband(vec: np.ndarray, limit: float) -> np.ndarray:
        if limit <= 0.0:
            return vec
        out = vec.copy()
        out[np.abs(out) < limit] = 0.0
        return out

    def _wrench_arrays(self, msg: WrenchStamped) -> Tuple[np.ndarray, np.ndarray]:
        force = np.array(
            [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
            dtype=float,
        )
        torque = np.array(
            [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z],
            dtype=float,
        )
        return force, torque

    def _source_frame(
        self,
        msg: WrenchStamped,
        fallback_frame: str,
        use_msg_frame_id: bool = True,
    ) -> str:
        if use_msg_frame_id and msg.header.frame_id:
            return msg.header.frame_id
        return fallback_frame

    def _rotation_to_base(self, source_frame: str) -> Optional[ScipyR]:
        if source_frame == self.base_frame:
            return ScipyR.identity()

        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                Time(),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"FT calibration waiting for TF {self.base_frame} <- "
                f"{source_frame}: {exc}",
                throttle_duration_sec=1.0,
            )
            return None

        q = trans.transform.rotation
        return ScipyR.from_quat([q.x, q.y, q.z, q.w])

    def _wrench_to_base(
        self,
        msg: WrenchStamped,
        fallback_frame: str,
        use_msg_frame_id: bool = True,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        force, torque = self._wrench_arrays(msg)
        source_frame = self._source_frame(msg, fallback_frame, use_msg_frame_id)
        rot = self._rotation_to_base(source_frame)
        if rot is None:
            return None
        return rot.apply(force), rot.apply(torque)

    def _publish_calibrated(
        self,
        msg: WrenchStamped,
        force_bias: np.ndarray,
        torque_bias: np.ndarray,
        pub,
        fallback_frame: str,
    ):
        converted = self._wrench_to_base(msg, fallback_frame)
        if converted is None:
            return
        force_base, torque_base = converted
        force_out = force_base - force_bias
        torque_out = torque_base - torque_bias

        out = WrenchStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.base_frame
        out.wrench.force.x = float(force_out[0])
        out.wrench.force.y = float(force_out[1])
        out.wrench.force.z = float(force_out[2])
        out.wrench.torque.x = float(torque_out[0])
        out.wrench.torque.y = float(torque_out[1])
        out.wrench.torque.z = float(torque_out[2])
        pub.publish(out)

    def _publish_external_calibrated(self, msg: WrenchStamped):
        if not self.external_source_frame_bias:
            self._publish_calibrated(
                msg,
                self.external_force_bias,
                self.external_torque_bias,
                self.external_pub,
                self.external_raw_frame,
            )
            return

        force, torque = self._wrench_arrays(msg)
        source_frame = self._source_frame(msg, self.external_raw_frame)
        rot_base_source = self._rotation_to_base(source_frame)
        if rot_base_source is None:
            return

        force_out = rot_base_source.apply(force - self.external_force_bias)
        torque_out = rot_base_source.apply(torque - self.external_torque_bias)

        out = WrenchStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.base_frame
        out.wrench.force.x = float(force_out[0])
        out.wrench.force.y = float(force_out[1])
        out.wrench.force.z = float(force_out[2])
        out.wrench.torque.x = float(torque_out[0])
        out.wrench.torque.y = float(torque_out[1])
        out.wrench.torque.z = float(torque_out[2])
        self.external_pub.publish(out)

    def _publish_internal_calibrated(self, msg: WrenchStamped):
        force, torque = self._wrench_arrays(msg)
        source_frame = self._source_frame(
            msg,
            self.internal_raw_frame,
            self.internal_use_msg_frame_id,
        )
        rot_base_source = self._rotation_to_base(source_frame)
        if rot_base_source is None:
            return

        if self.internal_source_frame_bias:
            force_bias_source = np.where(
                self.internal_force_bias_axes,
                self.internal_force_bias,
                0.0,
            )
            torque_bias_source = self.internal_torque_bias
            force_out = rot_base_source.apply(force - force_bias_source)
            torque_out = rot_base_source.apply(torque - torque_bias_source)
        else:
            # The internal sensor reports in the tool/sensor frame on some UR setups,
            # sometimes with an empty header.  Project the startup base-frame gravity
            # estimate into the current sensor frame before subtracting it, so wrist
            # rotations do not reintroduce XYZ force offsets.
            force_bias_base = np.where(
                self.internal_force_bias_axes,
                self.internal_force_bias,
                0.0,
            )
            force_bias_source = rot_base_source.inv().apply(force_bias_base)
            torque_bias_source = rot_base_source.inv().apply(self.internal_torque_bias)
            force_out = rot_base_source.apply(force - force_bias_source)
            torque_out = rot_base_source.apply(torque - torque_bias_source)

        force_out = self._component_deadband(
            force_out,
            self.internal_force_deadband_n,
        )
        torque_out = self._component_deadband(
            torque_out,
            self.internal_torque_deadband_nm,
        )

        out = WrenchStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.base_frame
        out.wrench.force.x = float(force_out[0])
        out.wrench.force.y = float(force_out[1])
        out.wrench.force.z = float(force_out[2])
        out.wrench.torque.x = float(torque_out[0])
        out.wrench.torque.y = float(torque_out[1])
        out.wrench.torque.z = float(torque_out[2])
        self.internal_pub.publish(out)

    def on_internal_wrench(self, msg: WrenchStamped):
        if self.ready:
            self._publish_internal_calibrated(msg)
            return
        if self.internal_source_frame_bias:
            self._collect_internal_source_sample(msg)
            return
        self._collect_sample(
            msg,
            self.internal_samples,
            self.internal_raw_frame,
            self.internal_use_msg_frame_id,
        )

    def _collect_internal_source_sample(self, msg: WrenchStamped):
        if not self._can_calibrate():
            if not self.logged_waiting:
                self.get_logger().warn(
                    "FT calibration blocked: attach the tool before startup "
                    "calibration can run."
                )
                self.logged_waiting = True
            return

        if len(self.internal_samples) < self.sample_count:
            self.internal_samples.append(self._wrench_arrays(msg))
        self._try_finish_calibration()

    def on_external_wrench(self, msg: WrenchStamped):
        if self.ready:
            self._publish_external_calibrated(msg)
            return
        if self.external_source_frame_bias:
            self._collect_external_source_sample(msg)
        else:
            self._collect_sample(msg, self.external_samples, self.external_raw_frame)

    def _collect_external_source_sample(self, msg: WrenchStamped):
        if not self._can_calibrate():
            if not self.logged_waiting:
                self.get_logger().warn(
                    "FT calibration blocked: attach the tool before startup "
                    "calibration can run."
                )
                self.logged_waiting = True
            return

        if len(self.external_samples) < self.sample_count:
            self.external_samples.append(self._wrench_arrays(msg))
        self._try_finish_calibration()

    def _collect_sample(
        self,
        msg: WrenchStamped,
        samples: list,
        fallback_frame: str,
        use_msg_frame_id: bool = True,
    ):
        if not self._can_calibrate():
            if not self.logged_waiting:
                self.get_logger().warn(
                    "FT calibration blocked: attach the tool before startup "
                    "calibration can run."
                )
                self.logged_waiting = True
            return

        converted = self._wrench_to_base(msg, fallback_frame, use_msg_frame_id)
        if converted is None:
            return
        if len(samples) < self.sample_count:
            samples.append(converted)
        self._try_finish_calibration()

    def _stream_ready(self, samples: list) -> bool:
        return len(samples) >= self.sample_count

    def _average_samples(self, samples: list) -> Tuple[np.ndarray, np.ndarray]:
        forces = np.asarray([sample[0] for sample in samples], dtype=float)
        torques = np.asarray([sample[1] for sample in samples], dtype=float)
        return np.mean(forces, axis=0), np.mean(torques, axis=0)

    def _sample_stddev(self, samples: list) -> Tuple[np.ndarray, np.ndarray]:
        forces = np.asarray([sample[0] for sample in samples], dtype=float)
        torques = np.asarray([sample[1] for sample in samples], dtype=float)
        return np.std(forces, axis=0), np.std(torques, axis=0)

    def _samples_stable(self, samples: list, label: str) -> bool:
        if not self.stability_check_enabled:
            return True

        force_std, torque_std = self._sample_stddev(samples)
        max_force_std = float(np.max(np.abs(force_std)))
        max_torque_std = float(np.max(np.abs(torque_std)))

        force_ok = (
            self.force_stddev_max_n <= 0.0
            or max_force_std <= self.force_stddev_max_n
        )
        torque_ok = (
            self.torque_stddev_max_nm <= 0.0
            or max_torque_std <= self.torque_stddev_max_nm
        )
        if force_ok and torque_ok:
            return True

        self.get_logger().warn(
            f"Rejecting unstable {label} FT calibration window: "
            f"force_std={_vec3_to_list(force_std)} N, "
            f"torque_std={_vec3_to_list(torque_std)} Nm. "
            "Keep the tool still and unloaded during startup calibration.",
            throttle_duration_sec=1.0,
        )
        return False

    def _try_finish_calibration(self):
        if self.ready or not self._stream_ready(self.internal_samples):
            return
        if self.external_required and not self._stream_ready(self.external_samples):
            return

        if not self._samples_stable(self.internal_samples, "internal"):
            self.internal_samples.clear()
            self.rejected_internal_windows += 1
            return
        if self._stream_ready(self.external_samples) and not self._samples_stable(
            self.external_samples,
            "external",
        ):
            self.external_samples.clear()
            self.rejected_external_windows += 1
            if self.external_required:
                return

        self.internal_force_bias, self.internal_torque_bias = self._average_samples(
            self.internal_samples
        )
        if self._stream_ready(self.external_samples):
            self.external_force_bias, self.external_torque_bias = self._average_samples(
                self.external_samples
            )
        self.ready = True

        self.get_logger().info("FT startup calibration complete.")
        internal_bias_frame = (
            "source-frame" if self.internal_source_frame_bias else "base-frame"
        )
        self.get_logger().info(
            f"Internal FT bias {internal_bias_frame} params: "
            f"force={_vec3_to_list(self.internal_force_bias)}, "
            f"torque={_vec3_to_list(self.internal_torque_bias)}"
        )
        if self._stream_ready(self.external_samples):
            self.get_logger().info(
                "External FT bias base-frame params: "
                f"force={_vec3_to_list(self.external_force_bias)}, "
                f"torque={_vec3_to_list(self.external_torque_bias)}"
            )

    def publish_ready_state(self):
        msg = Bool()
        msg.data = bool(self.ready)
        self.ready_pub.publish(msg)

        if self.ready and not self.logged_ready:
            self.logged_ready = True
            self.get_logger().info("FT calibration ready; control may start.")
        elif not self.ready:
            self.get_logger().info(
                "FT calibration progress: "
                f"internal={len(self.internal_samples)}/{self.sample_count}, "
                f"external={len(self.external_samples)}/{self.sample_count}, "
                f"attached={self.tool_attached}, "
                f"rejected_windows="
                f"{self.rejected_internal_windows}/"
                f"{self.rejected_external_windows}",
                throttle_duration_sec=2.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = FTCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
