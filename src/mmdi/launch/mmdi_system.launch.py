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
            output='screen'
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
        #   translation: x=-0.037 m, y=+0.104 m, z=0.0 m
        #   orientation assumption: 30 deg downward tilt toward (-y, -z),
        #   modeled as +30 deg rotation about +x:
        #   q = [0.2588190451, 0.0, 0.0, 0.9659258263]
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_frame_converter',
            arguments=[
                '--x', '-0.037',
                '--y', '0.104',
                '--z', '0.0',
                '--qx', '0.2588190451',
                '--qy', '0.0',
                '--qz', '0.0',
                '--qw', '0.9659258263',
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
            output='screen'
        ),
    ])
