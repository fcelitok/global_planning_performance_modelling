import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent
from launch.conditions import IfCondition
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration

from nav2_common.launch import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():

    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    bt_xml_file = LaunchConfiguration('bt_xml_file')
    use_remappings = LaunchConfiguration('use_remappings')

    # TODO(orduno) Remove once `PushNodeRemapping` is resolved
    #              https://github.com/ros2/launch_ros/issues/56
    remappings = [((namespace, '/tf'), '/tf'),
                  ((namespace, '/tf_static'), '/tf_static'),
                  ('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'bt_xml_filename': bt_xml_file,
    }

    configured_params = RewrittenYaml(
            source_file=params_file,
            root_key=namespace,
            param_rewrites=param_substitutions,
            convert_types=True)

    return LaunchDescription([

        DeclareLaunchArgument(
            'namespace', default_value='',
            description='Top-level namespace'),

        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation/bag clock if true'),

        DeclareLaunchArgument(
            'params_file',
            description='Full path to the ROS2 parameters file to use'),

        DeclareLaunchArgument(
            'bt_xml_file',
            default_value=os.path.join(get_package_share_directory('nav2_bt_navigator'), 'behavior_trees', 'navigate_w_replanning_and_recovery.xml'),
            description='Full path to the behavior tree xml file to use'),

        DeclareLaunchArgument(
            'use_remappings',
            default_value='false',
            description='Arguments to pass to all nodes launched by the file'),

        Node(
            package='nav2_controller',
            node_executable='controller_server',
            output='log',
            parameters=[configured_params],
            use_remappings=IfCondition(use_remappings),
            remappings=remappings,
            on_exit=EmitEvent(event=Shutdown(reason='nav2_controller_error'))
        ),

        Node(
            package='nav2_planner',
            node_executable='planner_server',
            node_name='planner_server',
            output='log',
            parameters=[configured_params],
            use_remappings=IfCondition(use_remappings),
            remappings=remappings,
            on_exit=EmitEvent(event=Shutdown(reason='nav2_planner_error'))
        ),

        Node(
            package='nav2_recoveries',
            node_executable='recoveries_server',
            node_name='recoveries_server',
            output='log',
            parameters=[{'use_sim_time': use_sim_time}],
            use_remappings=IfCondition(use_remappings),
            remappings=remappings,
            on_exit=EmitEvent(event=Shutdown(reason='nav2_recoveries_error'))
        ),

        Node(
            package='nav2_bt_navigator',
            node_executable='bt_navigator',
            node_name='bt_navigator',
            output='log',
            parameters=[configured_params],
            use_remappings=IfCondition(use_remappings),
            remappings=remappings,
            on_exit=EmitEvent(event=Shutdown(reason='nav2_bt_navigator_error'))
        ),

        Node(
            package='nav2_waypoint_follower',
            node_executable='waypoint_follower',
            node_name='waypoint_follower',
            output='log',
            parameters=[configured_params],
            use_remappings=IfCondition(use_remappings),
            remappings=remappings,
            on_exit=EmitEvent(event=Shutdown(reason='nav2_waypoint_follower_error'))
        ),

    ])
