#!/usr/bin/env python3

"""Probe tracker: single-node ArUco-based probe tip tracking via RealSense.

Replaces the old 5-node pipeline (usb_cam -> aruco_opencv -> april_state_aggregator
-> static TF publishers -> EKF) with one self-contained ROS 2 node.

Probe geometry and detection logic ported from the standalone tracker script.

Pipeline:
  pyrealsense2 @ 640x480 60fps
  -> ArUco detection (cv2, DICT_4X4_50)
  -> Probe tip math (ring geometry + CAD offsets)
  -> Particle filter (3000 particles) + rotation averaging
  -> Publishes /tf (head_camera -> odom), /vo (Odometry), debug image,
     /probe_tracker/camera_info,
     and Foxglove markers in head_camera frame
"""

import threading

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as SciRot

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

from mmdi.probe_particle_filter import (
    ProbeParticleFilter,
    average_rotations,
    get_angular_distance,
    gate_and_filter_glitches,
    stabilize_rotation_to_reference,
    enforce_display_z_up,
    _rotx_deg,
    _rotz_deg,
)


# ===========================================================================
# Probe geometry (from standalone tracker script)
# ===========================================================================

TAG_IDS = [0, 1, 2, 3, 4, 5, 9, 7]
DEFAULT_RING_ORDER = [2, 3, 4, 9, 1, 5]
DEFAULT_IGNORE_IDS = {6}
DEFAULT_RING_SIGN = -1
TAG_TILT_DEG = 90.0
TAG_MOUNT_ROTZ_DEG = 0.0

# Ring tags: tag center -> tool tip offset in tag frame (mm)
CAD_DY_MM = 110.89
CAD_DZ_MM = -34.15
# Off-ring tag 0
CAD0_DY_MM = 48.10
CAD0_DZ_MM = -25.94
# Off-ring tag 7
CAD7_DY_MM = 23.11
CAD7_DZ_MM = -30.0

MARKER_MM = 19.83


def build_static(tag_ids, ring_order, ring_sign, ring_yaw0_deg=0.0,
                 ring_step_deg=None):
    """Build per-tag rotation and translation offsets (tag frame -> probe tip).

    Returns dict with 'id_to_R_tag_probe', 'id_to_t_tag_probe', 'tag_ids'.
    """
    dY_default = CAD_DY_MM / 1000.0
    dZ_default = CAD_DZ_MM / 1000.0
    t_default = np.array([0.0, dY_default, dZ_default], dtype=np.float64)

    t_by_id = {int(tid): t_default.copy() for tid in tag_ids}
    t_by_id[0] = np.array([0.0, CAD0_DY_MM / 1000.0, CAD0_DZ_MM / 1000.0],
                           dtype=np.float64)
    t_by_id[7] = np.array([0.0, CAD7_DY_MM / 1000.0, CAD7_DZ_MM / 1000.0],
                           dtype=np.float64)

    R_tilt = _rotx_deg(TAG_TILT_DEG)

    # Compute yaw per ring tag
    step = (360.0 / len(ring_order)) if ring_step_deg is None else ring_step_deg
    step *= ring_sign
    yaw_per_tag = {
        int(tid): (ring_yaw0_deg + step * i) % 360.0
        for i, tid in enumerate(ring_order)
    }
    # Off-ring tags 0, 7 share yaw with reference tag 2
    yaw_ref = yaw_per_tag[2]
    yaw_per_tag.setdefault(0, yaw_ref)
    yaw_per_tag.setdefault(7, yaw_ref)

    id_to_R = {}
    id_to_t = {}
    for tag_id in tag_ids:
        if tag_id not in yaw_per_tag:
            continue
        yaw_deg = yaw_per_tag[tag_id] + TAG_MOUNT_ROTZ_DEG
        R_tag_probe = R_tilt @ _rotz_deg(yaw_deg)
        id_to_R[tag_id] = R_tag_probe
        id_to_t[tag_id] = t_by_id[int(tag_id)]

    return {
        "id_to_R_tag_probe": id_to_R,
        "id_to_t_tag_probe": id_to_t,
        "tag_ids": list(tag_ids),
    }


def compute_probe_from_tag(tag_id, R_c_tag, t_c_tag, static):
    """Compute probe tip pose in camera frame from a single detected tag.

    Returns (R_c_probe(3,3), t_c_probe(3,)).
    """
    R_tag_probe = static["id_to_R_tag_probe"][tag_id]
    t_tag_probe = static["id_to_t_tag_probe"][tag_id]
    R_c_probe = R_c_tag @ R_tag_probe
    t_c_probe = t_c_tag + R_c_tag @ t_tag_probe
    return R_c_probe, t_c_probe


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def draw_axes(img, K, dist, R_mat, t_vec, axis_len=0.04, thickness=2):
    pts = np.float32([[0, 0, 0],
                      [axis_len, 0, 0],
                      [0, axis_len, 0],
                      [0, 0, axis_len]])
    rvec, _ = cv2.Rodrigues(R_mat)
    imgpts, _ = cv2.projectPoints(pts, rvec, t_vec, K, dist)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    origin = tuple(imgpts[0])
    cv2.line(img, origin, tuple(imgpts[1]), (0, 0, 255), thickness)   # X red
    cv2.line(img, origin, tuple(imgpts[2]), (0, 255, 0), thickness)   # Y green
    cv2.line(img, origin, tuple(imgpts[3]), (255, 0, 0), thickness)   # Z blue


# ===========================================================================
# ROS 2 node
# ===========================================================================

class ProbeTracker(Node):
    def __init__(self):
        super().__init__('probe_tracker')

        # ---- ROS parameters ------------------------------------------------
        self.declare_parameter('marker_mm', float(MARKER_MM))
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('camera_fps', 60)
        self.declare_parameter('exposure', 100.0)
        self.declare_parameter('gain', 70.0)
        self.declare_parameter('show_gui', True)
        self.declare_parameter('n_particles', 3000)
        self.declare_parameter('process_noise', 0.002)
        self.declare_parameter('meas_noise', 0.015)
        self.declare_parameter('rot_jump_deg', 80.0)
        self.declare_parameter('viz_frame', 'head_camera')
        self.declare_parameter('markers_topic', '/probe_tracker/markers')
        self.declare_parameter('camera_info_topic', '/probe_tracker/camera_info')
        self.declare_parameter('marker_lifetime_s', 0.25)

        self.marker_m = self.get_parameter('marker_mm').value / 1000.0
        self.cam_w = self.get_parameter('camera_width').value
        self.cam_h = self.get_parameter('camera_height').value
        self.cam_fps = self.get_parameter('camera_fps').value
        self.exposure = self.get_parameter('exposure').value
        self.gain = self.get_parameter('gain').value
        self.show_gui = self.get_parameter('show_gui').value
        n_particles = self.get_parameter('n_particles').value
        proc_noise = self.get_parameter('process_noise').value
        meas_noise = self.get_parameter('meas_noise').value
        self.rot_jump_deg = self.get_parameter('rot_jump_deg').value
        self.viz_frame = self.get_parameter('viz_frame').value
        self.markers_topic = self.get_parameter('markers_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.marker_lifetime = Duration(
            seconds=float(self.get_parameter('marker_lifetime_s').value)
        ).to_msg()

        # ---- Probe geometry -------------------------------------------------
        active_ids = [tid for tid in TAG_IDS if tid not in DEFAULT_IGNORE_IDS]
        self.static = build_static(
            tag_ids=active_ids,
            ring_order=DEFAULT_RING_ORDER,
            ring_sign=DEFAULT_RING_SIGN,
        )
        self.known_ids = set(self.static["tag_ids"])
        self.ignore_ids = DEFAULT_IGNORE_IDS
        self.R_probe_display_fix = _rotx_deg(180.0)

        # ---- Publishers -----------------------------------------------------
        self.tf_broadcaster = TransformBroadcaster(self)
        self.vo_pub = self.create_publisher(Odometry, '/vo', 10)
        self.debug_pub = self.create_publisher(Image, '/probe_tracker/debug', 10)
        self.camera_info_pub = self.create_publisher(
            CameraInfo, self.camera_info_topic, 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, self.markers_topic, 10)
        self.bridge = CvBridge()

        # ---- Particle filter ------------------------------------------------
        self.pf = ProbeParticleFilter(
            n_particles=n_particles,
            process_noise_std=proc_noise,
            meas_noise_std=meas_noise,
        )
        self.prev_estimate = None
        self.prev_rot = None

        # ---- ArUco detector (OpenCV 4.6 legacy API) -------------------------
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # ---- Camera intrinsics (filled by RealSense) ------------------------
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_distortion_model = 'plumb_bob'
        self.camera_info_coeffs = None

        # ---- RealSense pipeline (background thread) -------------------------
        self._rs_pipe = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._running = True

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        # ---- Main processing timer ------------------------------------------
        self.create_timer(1.0 / self.cam_fps, self._process_frame)

        self.get_logger().info(
            f'ProbeTracker started: {self.cam_w}x{self.cam_h}@{self.cam_fps}fps, '
            f'{n_particles} particles, marker={self.marker_m*1000:.0f}mm'
        )

    # -----------------------------------------------------------------------
    # RealSense capture (blocking, runs in dedicated thread)
    # -----------------------------------------------------------------------
    def _capture_loop(self):
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(
                rs.stream.color, self.cam_w, self.cam_h,
                rs.format.bgr8, self.cam_fps)
            profile = pipe.start(cfg)

            # Manual exposure
            sensor = profile.get_device().first_color_sensor()
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, self.exposure)
            sensor.set_option(rs.option.gain, self.gain)

            # Read intrinsics
            intrinsics = (
                profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
                .get_intrinsics()
            )
            self.camera_matrix = np.array([
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            coeffs = np.array(intrinsics.coeffs, dtype=np.float64)
            self.dist_coeffs = np.zeros(8, dtype=np.float64)
            self.dist_coeffs[:min(len(coeffs), 8)] = coeffs[:min(len(coeffs), 8)]
            self.camera_info_coeffs = [float(c) for c in coeffs.tolist()]
            if intrinsics.model == rs.distortion.kannala_brandt4:
                self.camera_info_distortion_model = 'equidistant'
            else:
                self.camera_info_distortion_model = 'plumb_bob'

            self.get_logger().info(
                f'RealSense started – fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f}'
            )

            self._rs_pipe = pipe
            while self._running:
                frames = pipe.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                img = np.asanyarray(color.get_data())
                with self._frame_lock:
                    self._latest_frame = img

        except Exception as e:
            self.get_logger().error(f'RealSense capture failed: {e}')
        finally:
            if self._rs_pipe is not None:
                try:
                    self._rs_pipe.stop()
                except Exception:
                    pass

    # -----------------------------------------------------------------------
    # Per-frame processing
    # -----------------------------------------------------------------------
    def _process_frame(self):
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
        if frame is None or self.camera_matrix is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        tag_probe_positions = []
        tag_probe_rotations = []
        detected_info = []
        detected_tag_poses = []

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_m, self.camera_matrix, self.dist_coeffs)

            for rvec, tvec, mid in zip(rvecs, tvecs, ids.flatten()):
                tag_id = int(mid)
                R_c_tag, _ = cv2.Rodrigues(rvec.reshape(3, 1))
                t_c_tag = tvec.reshape(3)
                detected_tag_poses.append((tag_id, t_c_tag, R_c_tag))
                if tag_id in self.ignore_ids or tag_id not in self.known_ids:
                    continue

                R_c_probe, t_c_probe = compute_probe_from_tag(
                    tag_id, R_c_tag, t_c_tag, self.static)

                tag_probe_positions.append(t_c_probe)
                tag_probe_rotations.append(R_c_probe)
                detected_info.append((tag_id, t_c_probe, t_c_tag,
                                      R_c_probe @ self.R_probe_display_fix))

        # Gate and filter
        fused_pos = None
        fused_rot = None

        if tag_probe_positions:
            meas_positions, clean_rotations = gate_and_filter_glitches(
                tag_probe_positions, tag_probe_rotations,
                self.prev_estimate, self.prev_rot,
                max_jump=0.40, pos_thresh=0.08,
                rot_jump_deg=self.rot_jump_deg, rot_thresh_deg=15.0)

            if meas_positions.shape[0] > 0:
                if not self.pf.initialized:
                    self.pf.init_from_measurements(meas_positions)
                else:
                    self.pf.predict()
                    self.pf.update(meas_positions)

                fused_pos = self.pf.estimate_position()
                self.prev_estimate = fused_pos

                fused_rot_candidate = average_rotations(clean_rotations)
                if fused_rot_candidate is not None:
                    fused_rot_candidate = stabilize_rotation_to_reference(
                        fused_rot_candidate, self.prev_rot)
                    fused_rot_candidate = enforce_display_z_up(
                        fused_rot_candidate, self.R_probe_display_fix,
                        self.prev_rot)
                    fused_rot = fused_rot_candidate
                    self.prev_rot = fused_rot_candidate
                else:
                    fused_rot = self.prev_rot

        # Publish
        now = self.get_clock().now().to_msg()
        if fused_pos is not None and fused_rot is not None:
            quat = SciRot.from_matrix(fused_rot).as_quat()  # xyzw
            self._publish_tf(fused_pos, quat, now)
            self._publish_vo(fused_pos, quat, now)
        self._publish_markers(detected_tag_poses, fused_pos, fused_rot, now)

        # Debug image + GUI
        if fused_pos is not None and fused_rot is not None:
            fused_rot_disp = fused_rot @ self.R_probe_display_fix
            draw_axes(frame, self.camera_matrix, self.dist_coeffs,
                      fused_rot_disp, fused_pos, axis_len=0.06, thickness=3)
            euler = SciRot.from_matrix(fused_rot_disp).as_euler(
                'xyz', degrees=True)
            cv2.putText(
                frame,
                f'FUSED: [{fused_pos[0]:.4f}, {fused_pos[1]:.4f}, {fused_pos[2]:.4f}]',
                (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                cv2.LINE_AA)
            cv2.putText(
                frame,
                f'ROT:   [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]',
                (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                cv2.LINE_AA)

        n_tags = len(ids) if ids is not None else 0
        cv2.putText(frame, f'Tags: {n_tags}', (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    cv2.LINE_AA)

        self._publish_debug(frame, now)
        self._publish_camera_info(now)

        if self.show_gui:
            cv2.imshow('Probe Tracker', frame)
            cv2.waitKey(1)

    # -----------------------------------------------------------------------
    # Publishers
    # -----------------------------------------------------------------------
    def _publish_tf(self, pos, quat, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'head_camera'
        t.child_frame_id = 'odom'
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        self.tf_broadcaster.sendTransform(t)

    def _publish_vo(self, pos, quat, stamp):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'head_camera'
        msg.child_frame_id = 'odom'
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.x = float(quat[0])
        msg.pose.pose.orientation.y = float(quat[1])
        msg.pose.pose.orientation.z = float(quat[2])
        msg.pose.pose.orientation.w = float(quat[3])
        cov = (0.01 * np.eye(6)).astype(float).ravel().tolist()
        msg.pose.covariance = cov
        self.vo_pub.publish(msg)

    def _publish_debug(self, frame, stamp):
        if self.debug_pub.get_subscription_count() == 0:
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = 'head_camera'
        self.debug_pub.publish(msg)

    def _publish_camera_info(self, stamp):
        if self.camera_info_pub.get_subscription_count() == 0:
            return
        if self.camera_matrix is None:
            return

        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])

        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = 'head_camera'
        msg.width = int(self.cam_w)
        msg.height = int(self.cam_h)
        msg.distortion_model = self.camera_info_distortion_model
        msg.d = self.camera_info_coeffs if self.camera_info_coeffs else []
        msg.k = [fx, 0.0, cx,
                 0.0, fy, cy,
                 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0,
                 0.0, fy, cy, 0.0,
                 0.0, 0.0, 1.0, 0.0]
        self.camera_info_pub.publish(msg)

    def _publish_markers(self, detected_tag_poses, fused_pos, fused_rot, stamp):
        if self.marker_pub.get_subscription_count() == 0:
            return

        marker_array = MarkerArray()

        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = self.viz_frame
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for tag_id, t_c_tag, R_c_tag in detected_tag_poses:
            tag_quat = SciRot.from_matrix(R_c_tag).as_quat()
            used_for_fusion = tag_id in self.known_ids and tag_id not in self.ignore_ids

            tag_marker = Marker()
            tag_marker.header.stamp = stamp
            tag_marker.header.frame_id = self.viz_frame
            tag_marker.ns = 'april_tags'
            tag_marker.id = int(tag_id)
            tag_marker.type = Marker.CUBE
            tag_marker.action = Marker.ADD
            tag_marker.pose.position.x = float(t_c_tag[0])
            tag_marker.pose.position.y = float(t_c_tag[1])
            tag_marker.pose.position.z = float(t_c_tag[2])
            tag_marker.pose.orientation.x = float(tag_quat[0])
            tag_marker.pose.orientation.y = float(tag_quat[1])
            tag_marker.pose.orientation.z = float(tag_quat[2])
            tag_marker.pose.orientation.w = float(tag_quat[3])
            tag_marker.scale.x = float(self.marker_m)
            tag_marker.scale.y = float(self.marker_m)
            tag_marker.scale.z = 0.003
            if used_for_fusion:
                tag_marker.color.r = 0.20
                tag_marker.color.g = 0.85
                tag_marker.color.b = 0.20
            else:
                tag_marker.color.r = 1.00
                tag_marker.color.g = 0.50
                tag_marker.color.b = 0.10
            tag_marker.color.a = 0.90
            tag_marker.lifetime = self.marker_lifetime
            marker_array.markers.append(tag_marker)

            label_marker = Marker()
            label_marker.header.stamp = stamp
            label_marker.header.frame_id = self.viz_frame
            label_marker.ns = 'april_tag_labels'
            label_marker.id = 100 + int(tag_id)
            label_marker.type = Marker.TEXT_VIEW_FACING
            label_marker.action = Marker.ADD
            label_marker.pose.position.x = float(t_c_tag[0])
            label_marker.pose.position.y = float(t_c_tag[1])
            label_marker.pose.position.z = float(t_c_tag[2] + 0.03)
            label_marker.pose.orientation.w = 1.0
            label_marker.scale.z = 0.02
            label_marker.color.r = 1.0
            label_marker.color.g = 1.0
            label_marker.color.b = 1.0
            label_marker.color.a = 1.0
            label_marker.text = f'id:{tag_id}'
            label_marker.lifetime = self.marker_lifetime
            marker_array.markers.append(label_marker)

        if fused_pos is not None and fused_rot is not None:
            fused_quat = SciRot.from_matrix(fused_rot).as_quat()

            fused_sphere = Marker()
            fused_sphere.header.stamp = stamp
            fused_sphere.header.frame_id = self.viz_frame
            fused_sphere.ns = 'fused_pose'
            fused_sphere.id = 1000
            fused_sphere.type = Marker.SPHERE
            fused_sphere.action = Marker.ADD
            fused_sphere.pose.position.x = float(fused_pos[0])
            fused_sphere.pose.position.y = float(fused_pos[1])
            fused_sphere.pose.position.z = float(fused_pos[2])
            fused_sphere.pose.orientation.w = 1.0
            fused_sphere.scale.x = 0.018
            fused_sphere.scale.y = 0.018
            fused_sphere.scale.z = 0.018
            fused_sphere.color.r = 0.0
            fused_sphere.color.g = 0.95
            fused_sphere.color.b = 0.95
            fused_sphere.color.a = 1.0
            fused_sphere.lifetime = self.marker_lifetime
            marker_array.markers.append(fused_sphere)

            fused_arrow = Marker()
            fused_arrow.header.stamp = stamp
            fused_arrow.header.frame_id = self.viz_frame
            fused_arrow.ns = 'fused_pose'
            fused_arrow.id = 1001
            fused_arrow.type = Marker.ARROW
            fused_arrow.action = Marker.ADD
            fused_arrow.pose.position.x = float(fused_pos[0])
            fused_arrow.pose.position.y = float(fused_pos[1])
            fused_arrow.pose.position.z = float(fused_pos[2])
            fused_arrow.pose.orientation.x = float(fused_quat[0])
            fused_arrow.pose.orientation.y = float(fused_quat[1])
            fused_arrow.pose.orientation.z = float(fused_quat[2])
            fused_arrow.pose.orientation.w = float(fused_quat[3])
            fused_arrow.scale.x = 0.10
            fused_arrow.scale.y = 0.008
            fused_arrow.scale.z = 0.008
            fused_arrow.color.r = 1.0
            fused_arrow.color.g = 0.90
            fused_arrow.color.b = 0.10
            fused_arrow.color.a = 1.0
            fused_arrow.lifetime = self.marker_lifetime
            marker_array.markers.append(fused_arrow)

        self.marker_pub.publish(marker_array)

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------
    def destroy_node(self):
        self._running = False
        if self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self.show_gui:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ProbeTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
