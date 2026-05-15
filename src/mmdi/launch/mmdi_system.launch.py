#!/usr/bin/env python3

"""Main MMDI System Launch File
Converted to ROS 2 from ROS 1
"""

from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    external_ft_required = LaunchConfiguration('external_ft_required')

    return LaunchDescription([
        DeclareLaunchArgument(
            'external_ft_required',
            default_value='false',
            description='Wait for external FT sensor startup calibration.',
        ),

        Node(
            package='mmdi',
            executable='arduino_handler',
            name='arduino_handler',
            output='screen',
            parameters=[{
                'initial_led_state': 'a',
            }]
        ),

        # Startup FT calibration gate. Blocks mode/control until the internal
        # FT bias is estimated with the tool attached; external FT is optional.
        Node(
            package='mmdi',
            executable='ft_calibrator',
            name='ft_calibrator',
            output='screen',
            parameters=[{
                'internal_raw_topic': '/force_torque_sensor_broadcaster/wrench',
                'internal_raw_frame': 'tool0',
                'internal_use_msg_frame_id': True,
                'internal_source_frame_bias': False,
                'internal_force_bias_axes': [True, True, True],
                'internal_force_deadband_n': 1.5,
                'internal_torque_deadband_nm': 0.05,
                'internal_output_topic': '/ur7e/ft_internal_calibrated',
                'external_raw_topic': '/ur7e/ft_env_sensor_raw',
                'external_raw_frame': 'tool0',
                'external_source_frame_bias': True,
                'external_output_topic': '/ur7e/ft_env_sensor',
                'sample_count': 200,
                'external_required': ParameterValue(
                    external_ft_required,
                    value_type=bool,
                ),
                'require_tool_attached': True,
                'stability_check_enabled': True,
                'force_stddev_max_n': 1.0,
                'torque_stddev_max_nm': 0.08,
            }]
        ),

        # Mode handler node
        Node(
            package='mmdi',
            executable='mode_handler',
            name='mode_handler',
            output='screen',
            parameters=[{
                'calibration_required': True,
                'wrench_topic': '/ur7e/ft_internal_calibrated',
                'wrench_tare_samples': 0,
                'enable_freedrive_controller_topic': False,
            }]
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
        # CAD camera mount:
        #   RGB camera center in tool0 frame:
        #     x=-94.763 mm, y=-32.500 mm, z=+46.56311 mm
        #   hande_link_tool axes map into tool0 as:
        #     +X_hande -> +Y_tool0, +Y_hande -> +X_tool0, +Z_hande -> -Z_tool0
        #   so the equivalent translation in hande_link_tool is:
        #     x=-0.032500 m, y=-0.094763 m, z=-0.04656311 m
        #   camera axes in tool0 are the old mount orientation yawed 180 deg
        #   about tool0 +Z while preserving the same 25 deg camera tilt:
        #     x_cam = +y_tool0
        #     y_cam = -cos(25) * x_tool0 + sin(25) * z_tool0
        #     z_cam =  sin(25) * x_tool0 + cos(25) * z_tool0
        #   with the existing hande_link_tool alignment, this corresponds to:
        #   q = [0.9762960071, 0.0, 0.0, -0.2164396139]
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_frame_converter',
            arguments=[
                '--x', '-0.032500',
                '--y', '-0.094763',
                '--z', '-0.04656311',
                '--qx', '0.9762960071',
                '--qy', '0.0',
                '--qz', '0.0',
                '--qw', '-0.2164396139',
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
                'internal_bound_radius_m': 0.03,
                'global_tool_pos_min': [0.09, -0.43, 0.24],
                'global_tool_pos_max': [0.64, 0.44, 0.49],
                'look_alignment_weight': 100.0,
                'target_aim_offset_m': [0.0, 0.0, 0.0],
                'camera_distance_min_m': 0.24,
                'camera_distance_max_m': 0.40,
                'camera_distance_weight': 40.0,
                'side_offset_m': 0.20,
                'side_preference_weight': 10.0,
                'side_axis_sign': -1.0,
                'side_deadband': 0.15,
                'desired_tool_z_m': 0.36,
                'tool_height_weight': 2.0,
                'fused_y_alignment_weight': 0.1,
                'motion_weight': 1.0,
                'rotation_motion_weight': 0.1,
                'command_translation_deadband_m': 0.004,
                'command_rotation_deadband_rad': 0.035,
                'search_yaw_amplitude_rad': 0.35,
                'search_yaw_frequency_hz': 0.25,
                'search_yaw_axis_sign': 1.0,
            }]
        ),
    ])
