# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is all-in-one launch script intended for use by nav2 developers."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

from nav2_common.launch import Node


def generate_launch_description():

    # Create the launch configuration variables
    namespace = LaunchConfiguration('namespace')
    use_remappings = LaunchConfiguration('use_remappings')

    # Launch configuration variables specific to simulation
    use_robot_state_pub = LaunchConfiguration('use_robot_state_pub')
    headless = LaunchConfiguration('headless')
    world = LaunchConfiguration('world_model_file')

    # TODO(orduno) Remove once `PushNodeRemapping` is resolved
    #              https://github.com/ros2/launch_ros/issues/56
    remappings = [((namespace, '/tf'), '/tf'),
                  ((namespace, '/tf_static'), '/tf_static'),
                  ('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]

    # Declare the launch arguments
    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')

    declare_use_remappings_cmd = DeclareLaunchArgument(
        'use_remappings', default_value='false',
        description='Arguments to pass to all nodes launched by the file')

    declare_use_robot_state_pub_cmd = DeclareLaunchArgument(
        'use_robot_state_pub',
        default_value='True',
        description='Whether to start the robot state publisher')

    declare_simulator_cmd = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Whether to execute gzclient)')

    declare_world_cmd = DeclareLaunchArgument(
        'world_model_file',
        description='Full path to world model file to load')

    # Specify the actions
    start_gazebo_server_cmd = ExecuteProcess(
        cmd=['gzserver', '--verbose', '-s', 'libgazebo_ros_init.so', world],
        output='screen')

    start_gazebo_client_cmd = ExecuteProcess(
        condition=IfCondition(PythonExpression(['not ', headless])),
        cmd=['gzclient'],
        output='screen')

    urdf = os.path.join(get_package_share_directory('turtlebot3_description'), 'urdf', 'turtlebot3_waffle.urdf')

    start_robot_state_publisher_cmd = Node(
        condition=IfCondition(use_robot_state_pub),
        package='robot_state_publisher',
        node_executable='robot_state_publisher',
        node_name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': 'true'}],
        use_remappings=IfCondition(use_remappings),
        remappings=remappings,
        arguments=[urdf])

    # Create the launch description and populate
    ld = LaunchDescription()

    # Set env var to print messages to stdout immediately
    ld.add_action(SetEnvironmentVariable('RCUTILS_CONSOLE_STDOUT_LINE_BUFFERED', '1'))

    # Declare the launch options
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_remappings_cmd)
    ld.add_action(declare_use_robot_state_pub_cmd)

    ld.add_action(declare_simulator_cmd)
    ld.add_action(declare_world_cmd)

    # Add any conditioned actions
    ld.add_action(start_gazebo_server_cmd)
    ld.add_action(start_gazebo_client_cmd)

    # Add the actions to launch all of the navigation nodes
    ld.add_action(start_robot_state_publisher_cmd)

    return ld
