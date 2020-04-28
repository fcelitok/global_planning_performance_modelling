from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'configuration',
            description='Configuration yaml file path'),
        Node(
            package='localization_performance_modelling',
            node_executable='localization_benchmark_supervisor',
            node_name='localization_benchmark_supervisor',
            output='screen',
            parameters=[
                LaunchConfiguration('configuration')
            ],
        )
    ])
