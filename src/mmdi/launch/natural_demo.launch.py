#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    start_mmdi = LaunchConfiguration("start_mmdi")
    use_urscript_freedrive = LaunchConfiguration("use_urscript_freedrive")

    mmdi_launch = PythonLaunchDescriptionSource(
        [
            get_package_share_directory("mmdi"),
            "/launch/mmdi_system.launch.py",
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "start_mmdi",
            default_value="true",
            description="Launch the MMDI stack in addition to the UR control nodes.",
        ),
        DeclareLaunchArgument(
            "use_urscript_freedrive",
            default_value="true",
            description="Use URScript-based freedrive controller (experimental).",
        ),
        Node(
            package="ur_control",
            executable="position_controller",
            name="position_controller",
            output="screen",
        ),
        Node(
            package="ur_control",
            executable="sm_teleop",
            name="sm_teleop",
            output="screen",
        ),
        Node(
            package="ur_control",
            executable="compliance_controller",
            name="compliance_controller",
            output="screen",
        ),
        Node(
            package="ur_control",
            executable="freedrive_controller",
            name="freedrive_controller",
            output="screen",
            condition=IfCondition(use_urscript_freedrive),
        ),
        IncludeLaunchDescription(
            mmdi_launch,
            condition=IfCondition(start_mmdi),
        ),
    ])
