from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Set env var to print messages to stdout immediately
        SetEnvironmentVariable('RCUTILS_CONSOLE_STDOUT_LINE_BUFFERED', '1'),

        DeclareLaunchArgument(
            'configuration',
            description='Configuration yaml file path'),

        Node(
            package='localization_performance_modelling',
            node_executable='localization_benchmark_supervisor',
            node_name='localization_benchmark_supervisor',
            output='screen',
            emulate_tty=True,
            parameters=[
                LaunchConfiguration('configuration')
            ],
        )
    ])
