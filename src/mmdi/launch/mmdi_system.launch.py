#!/usr/bin/env python3

"""Main MMDI System Launch File
Converted to ROS 2 from ROS 1
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mmdi',
            executable='arduino_handler',
            name='arduino_handler',
            output='screen'
        ),

        # Mode handler node
        Node(
            package='mmdi',
            executable='mode_handler',
            name='mode_handler',
            output='screen'
        ),

        # Probe tracker (replaces usb_cam + aruco_opencv + april_state_aggregator + EKF)
        Node(
            package='mmdi',
            executable='probe_tracker',
            name='probe_tracker',
            output='screen',
            parameters=[{
                'use_infrared': False,
                'camera_width': 848,
                'camera_height': 480,
                'camera_fps': 60,
                'show_gui': True,
                'n_particles': 2500,
                'process_noise': 0.003,
                'meas_noise': 0.010,
                'velocity_noise': 0.025,
                'velocity_decay': 0.90,
                'pf_injection_rate': 0.12,
                'pf_injection_trigger_m': 0.008,
                'pf_injection_ramp_m': 0.015,
                'min_tag_confirm_frames': 1,
                'single_tag_alpha_scale': 0.75,
                'min_pos_alpha': 0.55,
                'position_response_m': 0.008,
                'measurement_velocity_alpha': 0.45,
                'prediction_lead_s': 0.008,
                'min_rot_alpha': 0.45,
                'rotation_response_deg': 4.0,
                'use_aruco3_detection': True,
                'debug_publish_hz': 10.0,
                'marker_publish_hz': 15.0,
                'camera_info_publish_hz': 2.0,
                'enable_latency_diagnostics': False,
                'latency_log_interval_s': 1.0,
            }]
        ),

        # Static transform: hand_e_link -> hande_link_tool
        # Original: 0 0 0 0.7071068 0.7071068 0 0
        # Args order in ROS 2: x y z qx qy qz qw frame_id child_frame_id
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='hande_frame_converter',
            arguments=[
                '--x', '0',
                '--y', '0',
                '--z', '0',
                '--qx', '0.7071068',
                '--qy', '0.7071068',
                '--qz', '0',
                '--qw', '0',
                '--frame-id', 'tool0',
                '--child-frame-id', 'hande_link_tool'
            ]
        ),

        # Static transform: hande_link_tool -> head_camera
        # Measured camera mount:
        #   camera center in tool0 frame:
        #     x=+99.47 mm, y=+31.80 mm, z=+78.76 mm
        #   hande_link_tool axes map into tool0 as:
        #     +X_hande -> +Y_tool0, +Y_hande -> +X_tool0, +Z_hande -> -Z_tool0
        #   so the translation in hande_link_tool is:
        #     x=+0.03180 m, y=+0.09947 m, z=-0.07876 m
        #   camera axes in tool0 should be:
        #     x_cam = -y_tool0
        #   then rotate +30 deg about the current camera x-axis:
        #     y_cam =  cos(30) * x_tool0 + sin(30) * z_tool0
        #     z_cam = -sin(30) * x_tool0 + cos(30) * z_tool0
        #   with the existing hande_link_tool alignment, this corresponds to:
        #   q = [0.0, 0.9659258263, -0.2588190451, 0.0]
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_frame_converter',
            arguments=[
                '--x', '0.03180',
                '--y', '0.09947',
                '--z', '-0.07876',
                '--qx', '0.0',
                '--qy', '0.9659258263',
                '--qz', '-0.2588190451',
                '--qw', '0.0',
                '--frame-id', 'hande_link_tool',
                '--child-frame-id', 'head_camera'
            ]
        ),

        # NOTE: SpaceMouse teleoperation is handled by ur_control/sm_teleop
        # (launched separately with compliance_controller and position_controller).
        # Do NOT launch mmdi/sm_teleop here — it conflicts on /ur7e/target_pose.

        # Natural demonstration handler
        Node(
            package='mmdi',
            executable='natural_handler',
            name='natural_handler',
            output='screen',
            parameters=[{
                'max_relative_euler_deg': [25.0, 25.0, 45.0],
                'max_relative_rotation_deg': 45.0,
                'delta_pos_max_m': 0.015,
                'global_tool_pos_min': [0.18, -0.28, 0.25],
                'global_tool_pos_max': [0.38, 0.28, 0.45],
            }]
        ),
    ])
