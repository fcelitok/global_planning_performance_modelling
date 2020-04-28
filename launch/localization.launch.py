import launch.actions
import launch.substitutions
from launch import LaunchDescription

import launch_ros.actions

from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    map_yaml_file = launch.substitutions.LaunchConfiguration('map')
    use_sim_time = launch.substitutions.LaunchConfiguration('use_sim_time')
    autostart = launch.substitutions.LaunchConfiguration('autostart')
    params_file = launch.substitutions.LaunchConfiguration('params')

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'yaml_filename': map_yaml_file
    }

    configured_params = RewrittenYaml(source_file=params_file, rewrites=param_substitutions, convert_types=True)

    return LaunchDescription([
        # Set env var to print messages to stdout immediately
        launch.actions.SetEnvironmentVariable(
            'RCUTILS_CONSOLE_STDOUT_LINE_BUFFERED', '1'),

        launch.actions.DeclareLaunchArgument(
            'map',
            description='Full path to map file to load'),

        launch.actions.DeclareLaunchArgument(
            'use_sim_time',
            description='Use simulation (Gazebo) clock if true'),

        launch.actions.DeclareLaunchArgument(
            'autostart',
            description='Automatically startup the nav2 stack'),

        launch.actions.DeclareLaunchArgument(
            'params',
            description='Full path to the ROS2 parameters file to use'),

        launch_ros.actions.Node(
            package='nav2_map_server',
            node_executable='map_server',
            node_name='map_server',
            output='screen',
            parameters=[configured_params]),

        launch_ros.actions.Node(
            package='nav2_amcl',
            node_executable='amcl',
            node_name='amcl',
            output='screen',
            parameters=[configured_params]),

        launch_ros.actions.Node(
            package='nav2_lifecycle_manager',
            node_executable='lifecycle_manager',
            node_name='lifecycle_manager_localizer',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time},
                        {'autostart': autostart},
                        {'node_names': ['map_server', 'amcl']}]),

    ])
