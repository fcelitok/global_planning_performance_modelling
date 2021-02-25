#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import time
import traceback
from collections import defaultdict, deque, OrderedDict

import geometry_msgs
import lifecycle_msgs
import nav_msgs
import networkx as nx
import numpy as np
import pandas as pd
import pyquaternion
import operator
import rclpy
from action_msgs.msg import GoalStatus
# from gazebo_msgs.msg import EntityState
# from gazebo_msgs.srv import SetEntityState
from lifecycle_msgs.msg import TransitionEvent
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ManageLifecycleNodes
from nav_msgs.msg import Odometry, Path
from performance_modelling_py.environment import ground_truth_map
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, Pose, Quaternion, PoseStamped, Point32
from rclpy.qos import qos_profile_sensor_data, QoSDurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud

import tf2_ros

import copy
import pickle
import psutil

import os
from os import path

from performance_modelling_py.utils import backup_file_if_exists, print_info, print_error, nanoseconds_to_seconds
from std_srvs.srv import Empty


class RunFailException(Exception):
    pass


def main(args=None):
    rclpy.init(args=args)

    node = None

    # noinspection PyBroadException
    try:
        node = GlobalPlanningBenchmarkSupervisor()
        # node.start_run()
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.ros_shutdown_callback()
    except RunFailException as e:
        print_error(e)
    except Exception:
        print_error(traceback.format_exc())

    finally:
        if node is not None:
            node.end_run()


class GlobalPlanningBenchmarkSupervisor(Node):
    def __init__(self):
        super().__init__('global_planning_benchmark_supervisor', automatically_declare_parameters_from_overrides=True)

        # topics, services, actions, entities and frames names
        initial_pose_topic = self.get_parameter('initial_pose_topic').value                # /initialpose
        navFn_topic = self.get_parameter('navFnROS_topic').value                           # /plan
        goal_pose_topic = self.get_parameter('goal_pose_topic').value                      # /goal_pose
        navigate_to_pose_action = self.get_parameter('navigate_to_pose_action').value      # /NavigateToPose
        self.fixed_frame = self.get_parameter('fixed_frame').value
        self.child_frame = self.get_parameter('child_frame').value
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.robot_entity_name = self.get_parameter('robot_entity_name').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.random_points = self.get_parameter('random_points').value  # define for how many random path will draw
        lifecycle_manager_service = self.get_parameter('lifecycle_manager_service').value
        set_entity_state_service = self.get_parameter('set_entity_state_service').value
        navigate_to_pose_action = self.get_parameter('navigate_to_pose_action').value
        # ground_truth_pose_topic = self.get_parameter('ground_truth_pose_topic').value
        # estimated_pose_correction_topic = self.get_parameter('estimated_pose_correction_topic').value
        # localization_node_transition_event_topic = self.get_parameter('localization_node_transition_event_topic').value
        # global_costmap_get_parameters_service = self.get_parameter('global_costmap_get_parameters_service').value
        # pause_physics_service = self.get_parameter('pause_physics_service').value
        # unpause_physics_service = self.get_parameter('unpause_physics_service').value

        # file system paths
        self.run_output_folder = self.get_parameter('run_output_folder').value
        self.benchmark_data_folder = path.join(self.run_output_folder, "benchmark_data")
        self.plan_output_folder = path.join(self.benchmark_data_folder, "plan_output")
        self.ground_truth_map_info_path = self.get_parameter("ground_truth_map_info_path").value
        # self.ps_output_folder = path.join(self.benchmark_data_folder, "ps_snapshots")

        # run parameters
        run_timeout = self.get_parameter('run_timeout').value
        self.ground_truth_map = ground_truth_map.GroundTruthMap(self.ground_truth_map_info_path)
        # ps_snapshot_period = self.get_parameter('ps_snapshot_period').value
        # write_estimated_poses_period = self.get_parameter('write_estimated_poses_period').value
        # self.ps_pid_father = self.get_parameter('pid_father').value
        # self.ps_processes = psutil.Process(self.ps_pid_father).children(recursive=True)  # list of processes children of the benchmark script, i.e., all ros nodes of the benchmark including this one
        # self.initial_pose_covariance_matrix = np.zeros((6, 6), dtype=float)
        # self.initial_pose_covariance_matrix[0, 0] = self.get_parameter('initial_pose_std_xy').value**2
        # self.initial_pose_covariance_matrix[1, 1] = self.get_parameter('initial_pose_std_xy').value**2
        # self.initial_pose_covariance_matrix[5, 5] = self.get_parameter('initial_pose_std_theta').value**2
        # self.goal_tolerance = self.get_parameter('goal_tolerance').value

        # run variables
        self.initial_pose = None
        self.goal_send_count = 0
        self.aborted_path_counter = 0
        self.path_receive = False
        self.path_aborted = False
        self.path_distance_token = False
        self.path_and_goal_write_token = False  # If this is false it will not write csv file to your path goal and initial points
        self.pathCounter = 0
        self.all_path_counter = 0
        self.initial_pose_counter = 1
        self.execution_timer = 0
        self.execution_timer2 = 0

        # self.total_paths = 0
        self.feasible_paths = 0
        self.unfeasible_paths = 0

        self.initial_goal_dict = OrderedDict()  # initial_goal_dict => key: initial node, value: goal nodes list
        self.voronoi_distance_dict = OrderedDict()  # voronoi_distance_dict => key : initial node, value: list(goal node, voronoi distance)
        self.shortest_path_dict = OrderedDict()  # shortest_path_dict => key: initial node, value: key: goal  value: shortest path nodes list

        # prepare folder structure
        if not path.exists(self.benchmark_data_folder):
            os.makedirs(self.benchmark_data_folder)

        if not path.exists(self.plan_output_folder):
            os.makedirs(self.plan_output_folder)

        # file paths for benchmark data
        self.run_events_file_path = path.join(self.benchmark_data_folder, "run_events.csv")
        self.initial_pose_file_path = path.join(self.plan_output_folder, "initial_pose.csv")
        self.goal_pose_file_path = path.join(self.plan_output_folder, "goal_pose.csv")
        self.received_plan_file_path = path.join(self.plan_output_folder, "received_plan.csv")
        self.execution_time_file_path = path.join(self.plan_output_folder, "execution_time.csv")
        self.voronoi_distance_file_path = path.join(self.plan_output_folder, "voronoi_distance.csv")
        self.euclidean_distance_file_path = path.join(self.plan_output_folder, 'euclidean_distance.csv')
        self.feasibility_rate_file_path = path.join(self.plan_output_folder, 'feasibility_rate.csv')
        self.mean_passage_width_file_path = path.join(self.plan_output_folder, 'mean_passage_width.csv')
        self.minimum_passage_width_file_path = path.join(self.plan_output_folder, 'minimum_passage_width.csv')
        self.mean_normalized_passage_width_file_path = path.join(self.plan_output_folder, 'mean_normalized_passage_width.csv')

        # self.estimated_poses_file_path = path.join(self.benchmark_data_folder, "estimated_poses.csv")
        # self.estimated_correction_poses_file_path = path.join(self.benchmark_data_folder, "estimated_correction_poses.csv")
        # self.ground_truth_poses_file_path = path.join(self.benchmark_data_folder, "ground_truth_poses.csv")
        # self.scans_file_path = path.join(self.benchmark_data_folder, "scans.csv")
        # self.run_events_file_path = path.join(self.benchmark_data_folder, "run_events.csv")
        # self.init_run_events_file()

        # setup publishers
        self.initial_pose_publisher = self.create_publisher(PoseWithCovarianceStamped, initial_pose_topic, 10)
        if self.path_and_goal_write_token:
            self.voronoi_publisher = self.create_publisher(PointCloud, 'VoronoiPointCloud', 10)

        # pandas dataframes for benchmark data
        # pandas dataframes for benchmark data
        self.run_events_df = pd.DataFrame(columns=['time', 'event'])
        self.execution_time_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'time'])
        self.voronoi_distance_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'voronoi_distance'])
        self.euclidean_distance_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'euclidean_distance'])
        self.feasibility_rate_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'path_feasibility'])
        self.mean_passage_width_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'mean_passage_width'])
        self.mean_normalized_passage_width_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'mean_normalized_passage_width'])
        self.minimum_passage_width_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'minimum_passage_width'])
        # self.estimated_poses_df = pd.DataFrame(columns=['t', 'x', 'y', 'theta'])
        # self.estimated_correction_poses_df = pd.DataFrame(columns=['t', 'x', 'y', 'theta', 'cov_x_x', 'cov_x_y', 'cov_y_y', 'cov_theta_theta'])
        # self.ground_truth_poses_df = pd.DataFrame(columns=['t', 'x', 'y', 'theta', 'v_x', 'v_y', 'v_theta'])

        self.voronoi_graph_node_finder()  # finding all initial nodes and goal nodes

        qos_profile = rclpy.qos.qos_profile_system_default

        # setup subscribers
        # self.create_subscription(LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Path, navFn_topic, self.pathCallback, qos_profile)
        self.create_subscription(PoseStamped, goal_pose_topic, self.goal_callback, qos_profile)

        # setup timers
        self.create_timer(run_timeout, self.run_timeout_callback)
        self.tfTimer = self.create_timer(1.0/50, self.tfTimerCallback)
        # self.create_timer(ps_snapshot_period, self.ps_snapshot_timer_callback)
        # self.create_timer(write_estimated_poses_period, self.write_estimated_pose_timer_callback)

        # setup service clients
        self.lifecycle_manager_service_client = self.create_client(ManageLifecycleNodes, lifecycle_manager_service)
        # self.global_costmap_get_parameters_service_client = self.create_client(GetParameters, global_costmap_get_parameters_service)
        # self.pause_physics_service_client = self.create_client(Empty, pause_physics_service)
        # self.unpause_physics_service_client = self.create_client(Empty, unpause_physics_service)
        # self.set_entity_state_service_client = self.create_client(SetEntityState, set_entity_state_service)

        # setup buffers
        self.broadcaster = tf2_ros.TransformBroadcaster(self)            # TODO check for self it must be node ??
        self.transformStamped = geometry_msgs.msg.TransformStamped()
        # self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # setup publishers
        self.initial_pose_publisher = self.create_publisher(PoseWithCovarianceStamped, initial_pose_topic, 1)
        # traversal_path_publisher_qos_profile = copy.copy(qos_profile_sensor_data)
        # traversal_path_publisher_qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        # self.traversal_path_publisher = self.create_publisher(Path, "~/traversal_path", traversal_path_publisher_qos_profile)

        # setup subscribers
        # self.create_subscription(LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data)
        # self.create_subscription(PoseWithCovarianceStamped, estimated_pose_correction_topic, self.estimated_pose_correction_callback, qos_profile_sensor_data)
        # self.create_subscription(Odometry, ground_truth_pose_topic, self.ground_truth_pose_callback, qos_profile_sensor_data)
        # self.localization_node_transition_event_subscriber = self.create_subscription(TransitionEvent, localization_node_transition_event_topic, self.localization_node_transition_event_callback, qos_profile_sensor_data)

        # setup action clients
        self.navigate_to_pose_action_client = ActionClient(self, NavigateToPose, navigate_to_pose_action)
        self.navigate_to_pose_action_goal_future = None
        self.navigate_to_pose_action_result_future = None

        self.write_event('run_start', self.get_clock().now())
        for initial_node_key, goal_node_value in self.initial_goal_dict.items():
            for goal_node in goal_node_value:
                self.start_run(initial_node=initial_node_key, goal_node=goal_node)
                print(" ")

                time.sleep(0.5)

    def voronoi_graph_node_finder(self):
        print_info("Entered -> deleaved reduced Voronoi graph from ground truth map")

        # get deleaved reduced Voronoi graph from ground truth map
        self.voronoi_graph = self.ground_truth_map.deleaved_reduced_voronoi_graph(minimum_radius=1 * self.robot_radius).copy()

        # get all voronoi graph nodes from ground truth map
        self.real_voronoi_graph = self.ground_truth_map.voronoi_graph.subgraph(filter(
            lambda n: self.ground_truth_map.voronoi_graph.nodes[n]['radius'] >= 1 * self.robot_radius,
            self.ground_truth_map.voronoi_graph.nodes
        )).copy()

        self.all_path_counter = len(self.voronoi_graph.nodes) + (len(self.voronoi_graph.nodes) * self.random_points)

        minimum_length_paths = nx.all_pairs_dijkstra_path(self.voronoi_graph, weight='voronoi_path_distance')
        minimum_length_costs = dict(nx.all_pairs_dijkstra_path_length(self.voronoi_graph, weight='voronoi_path_distance'))
        costs = defaultdict(dict)

        for i, paths_dict in minimum_length_paths:
            for j in paths_dict.keys():
                if i != j:
                    costs[i][j] = minimum_length_costs[i][j]
                    # print("Costs[{}][{}]: {}".format(i, j, costs[i][j]))

        # in case the graph has multiple unconnected components, remove the components with less than two nodes
        too_small_voronoi_graph_components = list(
            filter(lambda component: len(component) < 2, nx.connected_components(self.voronoi_graph)))

        for graph_component in too_small_voronoi_graph_components:
            self.voronoi_graph.remove_nodes_from(graph_component)

        if len(self.voronoi_graph.nodes) < 2:
            self.write_event('insufficient_number_of_nodes_in_deleaved_reduced_voronoi_graph', self.get_clock().now())
            raise RunFailException(
                "insufficient number of nodes in deleaved_reduced_voronoi_graph, can not generate traversal path")

        # self.total_paths = len(voronoi_graph.nodes)

        # find initial and goal nodes
        for node in list(self.voronoi_graph.nodes):
            # node our initial points and we will find farthest node
            initial_node = node
            farthest_node = max(costs[node].items(), key=operator.itemgetter(1))[0]
            # print("Max Costs[{}][{}] = {}".format(node, farthest_node, max_node_cost))
            max_node_cost = costs[node][farthest_node]

            goal_node_list = list()
            goal_node_list.append(farthest_node)

            # shortest path calculated in real voronoi node for each start and goal point
            shortest_path = nx.shortest_path(self.real_voronoi_graph, initial_node, farthest_node)
            # shortest path dictionary hold shortest path with key initial node
            shortest_path_ord_dict = OrderedDict()
            shortest_path_ord_dict[farthest_node] = shortest_path

            goal_and_distance_dict = OrderedDict()
            goal_and_distance_dict[farthest_node] = max_node_cost

            remove_list = [initial_node, farthest_node]

            if 0 <= self.random_points < len(self.voronoi_graph.nodes) - 1:
                random_final_point_list = random.sample(list(set(list(self.voronoi_graph.nodes)) - set(remove_list)), self.random_points)  # TODO voronoi_graph.nodes or node ??
            else:
                print_error("Cannot select random points more than nodes")

            for goal_node in random_final_point_list:
                goal_node_list.append(goal_node)
                shortest_path_for_goal = nx.shortest_path(self.real_voronoi_graph, initial_node, goal_node)
                shortest_path_ord_dict[goal_node] = shortest_path_for_goal  # TODO is there any problem in shortest path
                node_cost_for_goal = costs[initial_node][goal_node]
                goal_and_distance_dict[goal_node] = node_cost_for_goal

            self.initial_goal_dict[initial_node] = goal_node_list
            self.shortest_path_dict[initial_node] = shortest_path_ord_dict
            self.voronoi_distance_dict[initial_node] = goal_and_distance_dict

            node_diameter_list = list()
            for goal_key, path_nodes in shortest_path_ord_dict.items():
                for path_node in path_nodes:
                    node_diameter = self.real_voronoi_graph.nodes[path_node]['radius'] * 2
                    node_diameter_list.append(node_diameter)
                node_diameter_mean = np.mean(node_diameter_list)
                minimum_node_diameter = min(node_diameter_list)
                normalized_node_diameter_mean = node_diameter_mean / self.robot_radius

                initial_node_pose, goal_node_pose = self.pose_finder(initial_node, goal_key)

                self.mean_passage_width_callback(initial_node, goal_key, initial_node_pose, goal_node_pose, node_diameter_mean)
                self.mean_normalized_passage_width_callback(initial_node, goal_key, initial_node_pose, goal_node_pose, normalized_node_diameter_mean)
                self.minimum_passage_width_callback(initial_node, goal_key, initial_node_pose, goal_node_pose, minimum_node_diameter)
        # TODO: check for Point32()
        # if self.path_and_goal_write_token:
        #     point = Point32()
        #     self.point_cloud = PointCloud()
        #     self.point_cloud.header.stamp = self.get_clock().now()
        #     self.point_cloud.header.frame_id = 'map'
        #     for pose in list(self.real_voronoi_graph.nodes):
        #         point.x, point.y = self.real_voronoi_graph.nodes[pose]['vertex']
        #         point.z = 0.0
        #         self.point_cloud.points.append(Point32(point.x, point.y, point.z))

    def pose_finder(self, start_node, final_node):

        initial_node_pose = Pose()
        initial_node_pose.position.x, initial_node_pose.position.y = self.voronoi_graph.nodes[start_node]['vertex']
        q = pyquaternion.Quaternion(axis=[0, 0, 1], radians=np.random.uniform(-np.pi, np.pi))
        initial_node_pose.orientation = Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)
        # print("node number:", node, "INITIAL:" , initial_node_pose_stamped)

        goal_node_pose = Pose()
        goal_node_pose.position.x, goal_node_pose.position.y = self.voronoi_graph.nodes[final_node]['vertex']
        q = pyquaternion.Quaternion(axis=[0, 0, 1], radians=np.random.uniform(-np.pi, np.pi))
        goal_node_pose.orientation = Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)
        # print("farthest node number:",farthest_node, "GOAL:", goal_node_pose)

        return initial_node_pose, goal_node_pose

    def minimum_passage_width_callback(self, i_point, g_point, i_node_pose, g_node_pose, minimum_passage_width):
        try:
            self.minimum_passage_width_df = self.minimum_passage_width_df.append({
                'node_i': i_point,
                'i_x': i_node_pose.position.x,
                'i_y': i_node_pose.position.y,
                'node_g': g_point,
                'g_x': g_node_pose.position.x,
                'g_y': g_node_pose.position.y,
                'minimum_passage_width': minimum_passage_width
            }, ignore_index=True)
        except:
            print_error(traceback.format_exc())

    def mean_passage_width_callback(self, i_point, g_point, i_node_pose, g_node_pose, mean_passage_width):
        try:
            self.mean_passage_width_df = self.mean_passage_width_df.append({
                'node_i': i_point,
                'i_x': i_node_pose.position.x,
                'i_y': i_node_pose.position.y,
                'node_g': g_point,
                'g_x': g_node_pose.position.x,
                'g_y': g_node_pose.position.y,
                'mean_passage_width': mean_passage_width
            }, ignore_index=True)
        except:
            print_error(traceback.format_exc())

    def mean_normalized_passage_width_callback(self, i_point, g_point, i_node_pose, g_node_pose, mean_normalized_passage_width):
        try:
            self.mean_normalized_passage_width_df = self.mean_normalized_passage_width_df.append({
                'node_i': i_point,
                'i_x': i_node_pose.position.x,
                'i_y': i_node_pose.position.y,
                'node_g': g_point,
                'g_x': g_node_pose.position.x,
                'g_y': g_node_pose.position.y,
                'mean_normalized_passage_width': mean_normalized_passage_width
            }, ignore_index=True)
        except:
            print_error(traceback.format_exc())

    # TODO : ****************** active_cb and done_cb dont forget

    def get_result_callback(self, future):  ## ??????????????????
        status = future.result()   # it can be status = future.result().status

        if status == GoalStatus.STATUS_ACCEPTED:
            self.get_logger().info("Goal pose " + str(self.goal_send_count) + " received a cancel request after it started executing, and has since completed its execution!")
            time.sleep(1.0)
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().info("Goal pose " + str(self.goal_send_count) + " was aborted by the Action Server")
            self.path_aborted = True
            time.sleep(1.0)
        elif status == GoalStatus.STATUS_UNKNOWN:
            self.get_logger().warning("Goal pose " + str(self.goal_send_count) + " has yet to be processed by the action server")
            self.path_aborted = True
        elif status == GoalStatus.STATUS_EXECUTING:
            self.get_logger().warning("Goal pose " + str(self.goal_send_count) + " is currently being processed by the action server")
        elif status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warning("Goal pose " + str(self.goal_send_count) + " was achieved successfully by the action server")
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warning("Goal pose " + str(self.goal_send_count) + " received a cancel request before it started executing and was successfully cancelled")
            self.path_aborted = True
        elif status == GoalStatus.STATUS_CANCELING:
            self.get_logger().warning("Goal pose " + str(self.goal_send_count) + " received a cancel request. Cancelling!")
            self.path_aborted = True
        else:
            self.get_logger().error("There is no GoalStatus")
            self.path_aborted = True

    def goal_response_callback(self, future):
        goal_handle = future.result()

        self.execution_timer = self.get_clock().now()
        self.get_logger().info("Goal pose " + str(self.goal_send_count) + " is now being processed by the Action Server...")

        # if the goal is rejected try with the next goal
        if not goal_handle.accepted:
            print_error('navigation action goal rejected')
            self.write_event('target_pose_rejected', self.get_clock().now())
            self.path_aborted = True
            return

        self.write_event('target_pose_accepted', self.get_clock().now())

        self.navigate_to_pose_action_result_future = goal_handle.get_result_async()
        self.navigate_to_pose_action_result_future.add_done_callback(self.get_result_callback)

    def start_run(self, initial_node, goal_node):
        print_info("prepare start run for each path ")
        self.write_event('prepare_start_run_for_each_path', self.get_clock().now())
        self.send_initial_node = initial_node
        self.send_goal_node = goal_node

        initial_node_pose, goal_node_pose = self.pose_finder(start_node=initial_node, final_node=goal_node)

        self.send_initial_node_pose = initial_node_pose
        self.send_goal_node_pose = goal_node_pose

        self.voronoi_distance_callback(i_point=initial_node, g_point=goal_node, i_node_pose=initial_node_pose, g_node_pose=goal_node_pose)
        # self.longest_path_publisher_callback(initial_node)
        # if self.path_and_goal_write_token:
        #     self.voronoi_publisher.publish(self.point_cloud)

        initial_node_pose_stamped = PoseWithCovarianceStamped()
        initial_node_pose_stamped.header.frame_id = self.fixed_frame
        initial_node_pose_stamped.pose.pose = initial_node_pose

        # tf configuration
        self.transformStamped.header.frame_id = self.fixed_frame  # map -> fixed_frame
        self.transformStamped.child_frame_id = self.child_frame  # base_footprint -> child_frame
        # self.transformStamped.header.stamp = initial_node_pose_stamped.header.stamp  # stamp given in tfTimerCallback
        self.transformStamped.transform.translation.x = initial_node_pose_stamped.pose.pose.position.x
        self.transformStamped.transform.translation.y = initial_node_pose_stamped.pose.pose.position.y
        self.transformStamped.transform.translation.z = initial_node_pose_stamped.pose.pose.position.z
        self.transformStamped.transform.rotation.x = initial_node_pose_stamped.pose.pose.orientation.x
        self.transformStamped.transform.rotation.y = initial_node_pose_stamped.pose.pose.orientation.y
        self.transformStamped.transform.rotation.z = initial_node_pose_stamped.pose.pose.orientation.z
        self.transformStamped.transform.rotation.w = initial_node_pose_stamped.pose.pose.orientation.w
        # print("INITIAL POSE", initial_node_pose_stamped.pose)
        # print("TF transform OK for node:", initial_node)

        initial_node_pose_stamped.header.stamp = self.get_clock().now().to_msg()  # before publishing adding time stamp
        self.initial_pose_publisher.publish(initial_node_pose_stamped)
        self.initial_pose_callback(initial_node_pose_stamped)

        time.sleep(1.0)

        # self.traversal_path_publisher.publish(self.traversal_path_msg)  #traversal path publisher for visualization

        self.write_event('start_run_for_each_path', self.get_clock().now())

        # goal node send
        if not self.navigate_to_pose_action_client.wait_for_server(timeout_sec=5.0):  # just for control duration time is not important in here
            self.write_event('failed_to_communicate_with_navigation_node', self.get_clock().now())
            raise RunFailException("navigate_to_pose action server not available")

        maxGoalPose = NavigateToPose.Goal()
        maxGoalPose.pose.header.frame_id = self.fixed_frame
        maxGoalPose.pose.header.stamp = self.get_clock().now().to_msg()
        maxGoalPose.pose.pose = goal_node_pose

        self.goal_send_count += 1
        print("counter:{}/{}. For initial node {}, sending goal node {} ".format(self.goal_send_count, self.all_path_counter, initial_node, goal_node))
        # self.navigate_to_pose_action_client.send_goal_and_wait(maxGoalPose,execute_timeout=rospy.Duration.from_sec(1.0)),
        self.navigate_to_pose_action_goal_future = self.navigate_to_pose_action_client.send_goal_async(maxGoalPose)
        self.navigate_to_pose_action_goal_future.add_done_callback(self.goal_response_callback)
        self.write_event('target_pose_set', self.get_clock().now())
        # self.navigate_to_pose_action_client.wait_for_result(rospy.Duration.from_sec(1.0))

        time.sleep(0.5)
        while rclpy.ok():
            if self.path_receive:
                self.path_receive = False
                self.path_distance_token = True
                self.feasible_paths += 1
                feasibility_token = True
                self.feasibility_rate(initial_node, goal_node, initial_node_pose, goal_node_pose, feasibility_token)
                self.navigate_to_pose_action_client.remove_future(self.navigate_to_pose_action_goal_future)
                self.write_event('target_pose_reached', self.get_clock().now())
                self.write_event('finish_run_for_each_path', self.get_clock().now())
                print_info("PATH RECEIVED")
                break
            if self.path_aborted:
                self.path_aborted = False
                self.aborted_path_counter += 1
                self.unfeasible_paths += 1
                feasibility_token = False
                self.feasibility_rate(initial_node, goal_node, initial_node_pose, goal_node_pose, feasibility_token)
                self.navigate_to_pose_action_client.remove_future(self.navigate_to_pose_action_goal_future)
                self.write_event('target_pose_aborted', self.get_clock().now())
                self.write_event('finish_run_for_each_path', self.get_clock().now())
                print_info("PATH ABORTED. Counter: ", self.aborted_path_counter)
                break

        if self.path_distance_token:
            self.path_distance_token = False
            latest_initial_path_pose = self.latest_path.poses[0]
            distance_from_initial = np.sqrt(
                (initial_node_pose.position.x - latest_initial_path_pose.pose.position.x) ** 2 + (
                        initial_node_pose.position.y - latest_initial_path_pose.pose.position.y) ** 2)
            print_info("Distance: ", distance_from_initial)
            if distance_from_initial < self.goal_tolerance:
                self.write_event('initial_pose_true', self.get_clock().now())
            else:
                self.write_event('away_from_initial_pose', self.get_clock().now())
                print_error("current position farther from initial position than tolerance")

            latest_goal_path_pose = self.latest_path.poses[-1]
            distance_from_goal = np.sqrt((goal_node_pose.position.x - latest_goal_path_pose.pose.position.x) ** 2 + (
                    goal_node_pose.position.y - latest_goal_path_pose.pose.position.y) ** 2)
            print_info("Distance: ", distance_from_goal)
            if distance_from_goal < self.goal_tolerance:
                self.write_event('goal_pose_true', self.get_clock().now())
            else:
                self.write_event('away_from_goal_pose', self.get_clock().now())
                print_error("current position farther from goal position than tolerance")

        if self.goal_send_count == len(self.initial_goal_dict) * (self.random_points + 1):
            self.write_event('run_completed', self.get_clock().now())
            rclpy.shutdown()

    def tfTimerCallback(self, event):
        self.transformStamped.header.stamp = self.get_clock().now().to_msg()
        self.broadcaster.sendTransform(self.transformStamped)
        # print(event)

    def pathCallback(self, path_message):
        if not path_message.poses:
            self.path_aborted = True
        else:
            self.execution_timer2 = self.get_clock().now()
            if not self.path_receive:
                self.pathCounter += 1
                # print("sending path message ", pathMessage)
                print("Path message received for initial node {}, sending goal node {} ".format(self.send_initial_node, self.send_goal_node))
                self.execution_time_callback(self.send_initial_node, self.send_goal_node, self.send_initial_node_pose, self.send_goal_node_pose, self.execution_timer,
                                             self.execution_timer2)
                self.euclidean_distance_callback(path_message, self.send_initial_node, self.send_goal_node, self.send_initial_node_pose, self.send_goal_node_pose)
                self.path_receive = True
                self.latest_path = path_message

                if self.path_and_goal_write_token:
                    for pose in path_message.poses:
                        msg_time = pose.header.stamp.to_sec()
                        with open(self.received_plan_file_path, 'a') as path_file:
                            path_file.write(
                                "{counter}, {t}, {positionX}, {positionY}, {positionZ}, {orientationX}, {orientationY}, {orientationZ}, {orientationW}\n".format(
                                    counter=self.pathCounter,
                                    t=msg_time,
                                    positionX=pose.pose.position.x,
                                    positionY=pose.pose.position.y,
                                    positionZ=pose.pose.position.z,
                                    orientationX=pose.pose.orientation.x,
                                    orientationY=pose.pose.orientation.y,
                                    orientationZ=pose.pose.orientation.z,
                                    orientationW=pose.pose.orientation.w
                                ))

    def euclidean_distance_callback(self, pathMessage, i_point, g_point, i_node_pose, g_node_pose):
        try:
            x_y_position_list = list()
            # print("sending path message ", pathMessage)
            for pose in pathMessage.poses:
                x_y_position_list.append([pose.pose.position.x, pose.pose.position.y])

            x_y_position_first_del = list(x_y_position_list)
            x_y_position_last_del = list(x_y_position_list)
            x_y_position_first_del.pop(0)
            x_y_position_last_del.pop()
            # start from 1 to last-1 element so in euclidean first element (x2-x1) + (y2-y1)
            x_y_position_np_array_first = np.array(x_y_position_first_del)
            # start from 2 to last element so in euclidean second element (x2-x1) + (y2-y1)
            x_y_position_np_array_second = np.array(x_y_position_last_del)
            # print (x_y_position_np_array_first)
            # print (" ")
            # print (x_y_position_np_array_second)
            if len(x_y_position_np_array_first) == len(x_y_position_np_array_second):
                euclidean_distance = np.sum(
                    np.sqrt(np.sum(np.square(x_y_position_np_array_first - x_y_position_np_array_second), axis=1)))
                # print ("Euclidean Distance of path: ", euclidean_distance)

            self.euclidean_distance_df = self.euclidean_distance_df.append({
                'node_i': i_point,
                'i_x': i_node_pose.position.x,
                'i_y': i_node_pose.position.y,
                'node_g': g_point,
                'g_x': g_node_pose.position.x,
                'g_y': g_node_pose.position.y,
                'euclidean_distance': euclidean_distance
            }, ignore_index=True)
        except:
            print_error(traceback.format_exc())

    def initial_pose_callback(self, initial_pose_msg):
        self.initial_pose_counter += 1

        if self.path_and_goal_write_token:
            msg_time = initial_pose_msg.header.stamp.to_sec()
            with open(self.initial_pose_file_path, 'a') as initial_pose_file:
                initial_pose_file.write(
                    "{t}, {positionX}, {positionY}, {positionZ}, {orientationX}, {orientationY}, {orientationZ}, {orientationW}\n".format(
                        t=msg_time,
                        positionX=initial_pose_msg.pose.pose.position.x,
                        positionY=initial_pose_msg.pose.pose.position.y,
                        positionZ=initial_pose_msg.pose.pose.position.z,
                        orientationX=initial_pose_msg.pose.pose.orientation.x,
                        orientationY=initial_pose_msg.pose.pose.orientation.y,
                        orientationZ=initial_pose_msg.pose.pose.orientation.z,
                        orientationW=initial_pose_msg.pose.pose.orientation.w
                    ))

    def goal_callback(self, goal_msg):
        if self.path_and_goal_write_token:
            msg_time = goal_msg.header.stamp.to_sec()
            with open(self.goal_pose_file_path, 'a') as goal_file:
                goal_file.write(
                    "{t}, {positionX}, {positionY}, {positionZ}, {orientationX}, {orientationY}, {orientationZ}, {orientationW}\n".format(
                        t=msg_time,
                        positionX=goal_msg.goal.target_pose.pose.position.x,
                        positionY=goal_msg.goal.target_pose.pose.position.y,
                        positionZ=goal_msg.goal.target_pose.pose.position.z,
                        orientationX=goal_msg.goal.target_pose.pose.orientation.x,
                        orientationY=goal_msg.goal.target_pose.pose.orientation.y,
                        orientationZ=goal_msg.goal.target_pose.pose.orientation.z,
                        orientationW=goal_msg.goal.target_pose.pose.orientation.w
                    ))

    def voronoi_distance_callback(self, i_point, g_point, i_node_pose, g_node_pose):
        try:
            voronoi_dist = self.voronoi_distance_dict[i_point][g_point]  # goal -> key      voronoi -> value

            self.voronoi_distance_df = self.voronoi_distance_df.append({
                'node_i': i_point,
                'i_x': i_node_pose.position.x,
                'i_y': i_node_pose.position.y,
                'node_g': g_point,
                'g_x': g_node_pose.position.x,
                'g_y': g_node_pose.position.y,
                'voronoi_distance': voronoi_dist
            }, ignore_index=True)
        except:
            print_error(traceback.format_exc())

    def execution_time_callback(self, i_point, g_point, i_node_pose, g_node_pose, time_message, time_message2):
        try:
            msg_time = abs(time_message2 - time_message)
            self.get_logger().info("Global Planning execution time: " + str(msg_time))

            self.execution_time_df = self.execution_time_df.append({
                'node_i': i_point,
                'i_x': i_node_pose.position.x,
                'i_y': i_node_pose.position.y,
                'node_g': g_point,
                'g_x': g_node_pose.position.x,
                'g_y': g_node_pose.position.y,
                'time': msg_time
            }, ignore_index=True)
        except:
            print_error(traceback.format_exc())

    def feasibility_rate(self, i_point, g_point, i_node_pose, g_node_pose, feasible_token):
        try:
            if feasible_token:
                path_feasibility = 1
            elif not feasible_token:
                path_feasibility = 0
            else:
                print_error("There is problem in feasibility token")

            self.feasibility_rate_df = self.feasibility_rate_df.append({
                'node_i': i_point,
                'i_x': i_node_pose.position.x,
                'i_y': i_node_pose.position.y,
                'node_g': g_point,
                'g_x': g_node_pose.position.x,
                'g_y': g_node_pose.position.y,
                'path_feasibility': path_feasibility
            }, ignore_index=True)
        except:
            print_error(traceback.format_exc())

    def run_timeout_callback(self):
        print_error("terminating supervisor due to timeout, terminating run")
        self.write_event(self.get_clock().now(), 'run_timeout')
        self.write_event(self.get_clock().now(), 'supervisor_finished')
        raise RunFailException("timeout")

    def write_event(self, event, time):
        backup_file_if_exists(self.run_events_file_path)
        print_info("t: {t}, event: {event}".format(t=time, event=str(event)))
        try:
            self.run_events_df = self.run_events_df.append({
                'time': time,
                'event': event
            }, ignore_index=True)
        except IOError as e:
            self.get_logger().error(
                "slam_benchmark_supervisor.write_event: could not write event to run_events_file: {t} {event}".format(
                    t=time, event=str(event)))
            self.get_logger().error(e)

    def ros_shutdown_callback(self):
        """
        This function is called when the node receives an interrupt signal (KeyboardInterrupt).
        """
        print_info("asked to shutdown, terminating run")
        self.write_event(self.get_clock().now(), 'ros_shutdown')
        self.write_event(self.get_clock().now(), 'supervisor_finished')

    def end_run(self):
        """
        This function is called after the run has completed, whether the run finished correctly, or there was an exception.
        The only case in which this function is not called is if an exception was raised from self.__init__
        """
        self.run_events_df.to_csv(self.run_events_file_path, index=False)
        self.execution_time_df.to_csv(self.execution_time_file_path, index=False)
        self.voronoi_distance_df.to_csv(self.voronoi_distance_file_path, index=False)
        self.euclidean_distance_df.to_csv(self.euclidean_distance_file_path, index=False)
        self.feasibility_rate_df.to_csv(self.feasibility_rate_file_path, index=False)
        self.mean_passage_width_df.to_csv(self.mean_passage_width_file_path, index=False)
        self.minimum_passage_width_df.to_csv(self.minimum_passage_width_file_path, index=False)
        self.mean_normalized_passage_width_df.to_csv(self.mean_normalized_passage_width_file_path, index=False)
