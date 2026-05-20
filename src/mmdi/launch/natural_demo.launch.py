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
    external_ft_required = LaunchConfiguration("external_ft_required")

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
            description=(
                "Use custom URScript-based freedrive controller for mode 2."
            ),
        ),
        DeclareLaunchArgument(
            "external_ft_required",
            default_value="true",
            description="Wait for external FT sensor startup calibration.",
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
        Node(
            package="mmdi",
            executable="wrench_env_sensor",
            name="wrench_env_sensor",
            output="screen",
            parameters=[{
                "output_topic": "/ur7e/ft_env_sensor_raw",
                "output_frame": "tool0",
                "tare_on_startup": False,
            }],
        ),
        IncludeLaunchDescription(
            mmdi_launch,
            condition=IfCondition(start_mmdi),
            launch_arguments={
                "external_ft_required": external_ft_required,
            }.items(),
        ),
    ])
