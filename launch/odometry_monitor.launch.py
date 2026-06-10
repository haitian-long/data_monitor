from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    odom_topic = LaunchConfiguration("odom_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "odom_topic",
                default_value="/robot/dlio/odom_node/odom",
                description="Odometry topic to monitor.",
            ),
            Node(
                package="data_monitor",
                executable="odometry_monitor",
                name="odometry_monitor",
                output="screen",
                parameters=[{"odom_topic": odom_topic}],
            ),
        ]
    )
