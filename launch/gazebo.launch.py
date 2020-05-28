from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

from nav2_common.launch import Node


def generate_launch_description():

    ld = LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_STDOUT_LINE_BUFFERED', '1'),
        DeclareLaunchArgument(
            'headless',
            default_value='False',
            description='Whether to execute gzclient)'),
        DeclareLaunchArgument(
            'world_model_file',
            description='Full path to world model file to load'),
        DeclareLaunchArgument(
            'robot_gt_urdf_file',
            description='Full path to robot urdf model file to load'),
        DeclareLaunchArgument(
            'robot_realistic_urdf_file',
            description='Full path to robot urdf model file to load'),
        ExecuteProcess(
            cmd=['gzserver', '--verbose', '-s', 'libgazebo_ros_init.so', LaunchConfiguration('world_model_file')],
            output='screen'),
        ExecuteProcess(
            condition=IfCondition(PythonExpression(['not ', LaunchConfiguration('headless')])),
            cmd=['gzclient'],
            output='log'),
        Node(
            package='robot_state_publisher',
            node_executable='robot_state_publisher',
            node_name='robot_state_publisher_gt',
            output='log',
            parameters=[{'use_sim_time': True}],
            arguments=[LaunchConfiguration('robot_gt_urdf_file')]),
        Node(
            package='tf2_ros',
            node_executable='static_transform_publisher',
            node_name='gt_odom_static_transform_publisher',
            output='log',
            parameters=[{'use_sim_time': True}],
            arguments=["0", "0", "0", "0", "0", "0", "map", "odom"]),  # odom is actually the ground truth, because currently (Eloquent) odom and base_link are hardcoded in the navigation stack (recoveries and controller) >:C
        Node(
            package='robot_state_publisher',
            node_executable='robot_state_publisher',
            node_name='robot_state_publisher_realistic',
            output='log',
            parameters=[{'use_sim_time': True}],
            arguments=[LaunchConfiguration('robot_realistic_urdf_file')]),
    ])

    return ld
