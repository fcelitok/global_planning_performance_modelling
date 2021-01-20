#!/usr/bin/env python
# -*- coding: utf-8 -*-

import traceback
from collections import defaultdict
import os
from os import path
import numpy as np
import networkx as nx
import pyquaternion
import geometry_msgs
import operator
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import rospy
import tf2_ros
from collections import OrderedDict
from actionlib import SimpleActionClient
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction, MoveBaseActionGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Quaternion, PoseStamped
from nav_msgs.msg import Path

from performance_modelling_py.environment import ground_truth_map
from performance_modelling_py.utils import backup_file_if_exists, print_info, print_error

from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud


class RunFailException(Exception):
    pass


def main():
    rospy.init_node('slam_benchmark_supervisor', anonymous=False)

    node = None

    # noinspection PyBroadException
    try:
        node = GlobalPlanningBenchmarkSupervisor()
        # node.start_run()
        rospy.spin()

    except KeyboardInterrupt:
        node.ros_shutdown_callback()
    except RunFailException as e:
        print_error(e)
    except Exception:
        print_error(traceback.format_exc())

    finally:
        if node is not None:
            node.end_run()
        if not rospy.is_shutdown():
            print_info("calling rospy signal_shutdown")
            rospy.signal_shutdown("run_terminated")


class GlobalPlanningBenchmarkSupervisor:
    def __init__(self):

        # topics, services, actions, entities and frames names
        initial_pose_topic = rospy.get_param('~initial_pose_topic')  # /initialpose
        navFn_topic = rospy.get_param('~navFnROS_topic')  # /move_base/NavfnROS/plan
        global_planner_topic = rospy.get_param('~global_planner_topic')  # /move_base/GlobalPlanner/plan
        sbpl_lattice_planner_topic = rospy.get_param('~sbpl_lattice_planner_topic')  # /move_base/SBPLLatticePlanner/plan
        ompl_planner_topic = rospy.get_param('~ompl_planner_topic')  # /move_base/OmplGlobalPlanner/plan
        goal_pose_topic = rospy.get_param('~goal_pose_topic')  # /move_base/goal
        navigate_to_pose_action = rospy.get_param('~navigate_to_pose_action')  # /move_base
        self.fixed_frame = rospy.get_param('~fixed_frame')
        self.child_frame = rospy.get_param('~child_frame')
        self.robot_base_frame = rospy.get_param('~robot_base_frame')
        self.robot_entity_name = rospy.get_param('~robot_entity_name')
        self.goal_tolerance = rospy.get_param('~goal_tolerance')
        self.random_points = rospy.get_param('~random_points')          # define for how many random path will draw

        # file system paths
        self.run_output_folder = rospy.get_param('~run_output_folder')
        self.benchmark_data_folder = path.join(self.run_output_folder, "benchmark_data")
        self.plan_output_folder = path.join(self.benchmark_data_folder, "plan_output")
        self.ground_truth_map_info_path = rospy.get_param('~ground_truth_map_info_path')

        # run parameters
        self.robot_kinematic = rospy.get_param('~robot_kinematic')
        self.robot_radius = rospy.get_param('~robot_radius')
        self.robot_major_radius = rospy.get_param('~robot_major_radius')
        run_timeout = rospy.get_param('~run_timeout')
        self.ground_truth_map = ground_truth_map.GroundTruthMap(self.ground_truth_map_info_path)

        # run variables
        self.initial_pose = None
        self.goal_send_count = 0
        self.aborted_path_counter = 0
        self.path_receive = False
        self.path_aborted = False
        self.path_distance_token = False    
        self.voronoi_visualize = False               # If you want to see voronio graph in rviz open here
        self.path_and_goal_write_token = False      # If this is false it will not write csv file to your path goal and initial points
        self.pathCounter = 0
        self.all_path_counter = 0
        self.initial_pose_counter = 1
        self.execution_timer = 0
        self.execution_timer2 = 0
  

        # self.total_paths = 0
        self.feasible_paths = 0
        self.unfeasible_paths = 0

        self.initial_goal_dict = OrderedDict()      # initial_goal_dict => key: initial node, value: goal nodes list
        self.voronoi_distance_dict = OrderedDict()  # voronoi_distance_dict => key : initial node, value: list(goal node, voronoi distance)
        self.shortest_path_dict = OrderedDict()     # shortest_path_dict => key: initial node, value: key: goal  value: shortest path nodes list

        # prepare folder structure
        if not path.exists(self.benchmark_data_folder):
            os.makedirs(self.benchmark_data_folder)

        if not path.exists(self.plan_output_folder):
            os.makedirs(self.plan_output_folder)

        # file paths for benchmark data    #we are using it for collecting benchmark data
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
        # self.init_run_events_file()  # closed because added new pandas data frame file write function

        # setup publishers
        self.initial_pose_publisher = rospy.Publisher(initial_pose_topic, PoseWithCovarianceStamped, queue_size=10)
        
        if self.voronoi_visualize:
            self.voronoi_publisher = rospy.Publisher('VoronoiPointCloud', PointCloud, queue_size=10)
            # TODO check path or point cloud
            self.radius_voronoi_publisher = rospy.Publisher('Longest_voronoi_path', PointCloud, queue_size=10)
            # self.vor_path_pub = rospy.Publisher('xxx', Path, queue_size=10)

        # pandas dataframes for benchmark data
        self.run_events_df = pd.DataFrame(columns=['time', 'event'])
        self.execution_time_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'time'])
        self.voronoi_distance_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'voronoi_distance'])
        self.euclidean_distance_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'euclidean_distance'])
        self.feasibility_rate_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'path_feasibility'])
        self.mean_passage_width_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'mean_passage_width'])
        self.mean_normalized_passage_width_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'mean_normalized_passage_width'])
        self.minimum_passage_width_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'minimum_passage_width'])

        self.voronoi_graph_node_finder()  # finding all initial nodes and goal nodes

        # setup subscribers
        rospy.Subscriber(navFn_topic, Path, self.pathCallback)
        rospy.Subscriber(global_planner_topic, Path, self.pathCallback)
        rospy.Subscriber(sbpl_lattice_planner_topic, Path, self.pathCallback)
        rospy.Subscriber(ompl_planner_topic, Path, self.pathCallback)
        rospy.Subscriber(goal_pose_topic, MoveBaseActionGoal, self.goal_callback)
        # you can add your subscribers here
        # you can add subscriber path here

        # setup action clients
        # navigate_to_pose_action => /move_base
        self.navigate_to_pose_action_client = SimpleActionClient(navigate_to_pose_action, MoveBaseAction)
        self.navigate_to_pose_action_client.wait_for_server(rospy.Duration.from_sec(0.5))

        # setup timers
        rospy.Timer(rospy.Duration.from_sec(run_timeout), self.run_timeout_callback)
        self.tfTimer = rospy.Timer(rospy.Duration().from_sec(1.0 / 50), self.tfTimerCallback)

        # tf send
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.transformStamped = geometry_msgs.msg.TransformStamped()

        # # only one node send part just for debugging  (close for loop)
        # initial_node_key = list(self.initial_goal_dict)[0]
        # goal_node_key = self.initial_goal_dict[initial_node_key][0]
        # self.start_run(initial_node=initial_node_key, goal_node=goal_node_key)

        # send initial node and goal node
        self.write_event('run_start', rospy.Time.now().to_sec())
        for initial_node_key, goal_node_value in self.initial_goal_dict.items():
            for goal_node in goal_node_value:
                self.start_run(initial_node=initial_node_key, goal_node=goal_node)
                print(" ")

                rospy.sleep(0.5)

    # collect all initial nodes and goal nodes in dictionary
    def voronoi_graph_node_finder(self):
        print_info("Entered -> deleaved reduced Voronoi graph from ground truth map")

        if self.robot_kinematic == 'unicycle':
            safety_factor = 2
            major_safety_factor = 2
        else:
            safety_factor = 1
            major_safety_factor = 1.35

        # get deleaved reduced Voronoi graph from ground truth map
        self.voronoi_graph = self.ground_truth_map.deleaved_reduced_voronoi_graph(minimum_radius=safety_factor * self.robot_radius).copy()
        
        # get all voronoi graph nodes from ground truth map
        self.real_voronoi_graph = self.ground_truth_map.voronoi_graph.subgraph(filter(
            lambda n: self.ground_truth_map.voronoi_graph.nodes[n]['radius'] >= safety_factor * self.robot_radius,
            self.ground_truth_map.voronoi_graph.nodes)).copy()

        minimum_length_paths = nx.all_pairs_dijkstra_path(self.voronoi_graph, weight='voronoi_path_distance')
        minimum_length_costs = dict(nx.all_pairs_dijkstra_path_length(self.voronoi_graph, weight='voronoi_path_distance'))
        costs = defaultdict(dict)

        for i, paths_dict in minimum_length_paths:
            for j in paths_dict.keys():
                if i != j:
                    costs[i][j] = minimum_length_costs[i][j]
                    # print("Costs[{}][{}]: {}".format(i, j, costs[i][j]))

        # in case the graph has multiple unconnected components, remove the components with less than %10 of total graph
        too_small_real_voronoi_graph_components = list(filter(lambda component: len(component) < 0.1*len(self.real_voronoi_graph), nx.connected_components(self.real_voronoi_graph)))

        for graph_component in too_small_real_voronoi_graph_components:
            print_info("ignoring {} nodes from unconnected components in the Real Voronoi graph".format(len(graph_component)))
            self.real_voronoi_graph.remove_nodes_from(graph_component)

        if len(self.real_voronoi_graph.nodes) < 2:
            self.write_event('insufficient_number_of_nodes_in_deleaved_reduced_voronoi_graph')
            raise RunFailException("insufficient number of nodes in deleaved_reduced_voronoi_graph, can not generate traversal path")
        
         # in case the graph has multiple unconnected components, remove the components with less than %10 of total graph
        too_small_voronoi_graph_components = list(filter(lambda component: len(component) < 0.1*len(self.voronoi_graph), nx.connected_components(self.voronoi_graph)))

        for graph_component in too_small_voronoi_graph_components:
            print_info("ignoring {} nodes from unconnected components in the Voronoi graph".format(len(graph_component)))
            self.voronoi_graph.remove_nodes_from(graph_component)

        if len(self.voronoi_graph.nodes) < 2:
            self.write_event('insufficient_number_of_nodes_in_deleaved_reduced_voronoi_graph', rospy.Time.now().to_sec())
            raise RunFailException("insufficient number of nodes in deleaved_reduced_voronoi_graph, can not generate traversal path")

        self.radius_voronoi_graph = self.voronoi_graph.subgraph(filter(lambda n: self.voronoi_graph.nodes[n]['radius'] > major_safety_factor * self.robot_major_radius, self.voronoi_graph.nodes)).copy()

        self.all_path_counter = len(self.radius_voronoi_graph.nodes) + (len(self.radius_voronoi_graph.nodes)*self.random_points)
        # radius_voronoi_nodes = list(filter(lambda n: self.voronoi_graph.nodes[n]['radius'] > 0.65, self.voronoi_graph))

        # Open this area If you want to see voronoi graph in map
        # draw_map = 1

        # if draw_map == 1:
        #     ground_truth_map_png_path = path.join("/home/furkan/ds/performance_modelling/test_datasets/dataset/airlab", "data", "map.pgm")
        #     image = mpimg.imread(ground_truth_map_png_path)
        #     plt.imshow(image)
            
        #     num_nodes = 0
        #     for node_index, node_data in self.radius_voronoi_graph.nodes.data():
        #         num_nodes += 1

        #         x_11, y_11 = node_data['vertex']
        #         x_1, y_1 = self.ground_truth_map.map_frame_to_image_coordinates([x_11, y_11])
        #         radius_1 = node_data['radius']

        #         plt.scatter(x_1, y_1, color='red', s=12.0, marker='o')

        #         # plot segments
        #         for neighbor_index in self.radius_voronoi_graph.neighbors(node_index):
        #             if neighbor_index < node_index:
        #                 radius_2 = self.radius_voronoi_graph.nodes[neighbor_index]['radius']
        #                 if radius_2 > 0.4:
        #                     x_22, y_22 = self.radius_voronoi_graph.nodes[neighbor_index]['vertex']
        #                     x_2, y_2 = self.ground_truth_map.map_frame_to_image_coordinates([x_22, y_22])
        #                     plt.plot((x_1, x_2), (y_1, y_2), color='black', linewidth=2.0)

        #     num_nodes = 0
        #     for node_index, node_data in self.real_voronoi_graph.nodes.data():
        #         num_nodes += 1

        #         x_11, y_11 = node_data['vertex']
        #         x_1, y_1 = self.ground_truth_map.map_frame_to_image_coordinates([x_11, y_11])
        #         radius_1 = node_data['radius']

        #         plt.scatter(x_1, y_1, color='blue', s=0.5, marker='*')

        #         # plot segments
        #         for neighbor_index in self.real_voronoi_graph.neighbors(node_index):
        #             if neighbor_index < node_index:
        #                 radius_2 = self.real_voronoi_graph.nodes[neighbor_index]['radius']
        #                 if radius_2 > 0.4:
        #                     x_22, y_22 = self.real_voronoi_graph.nodes[neighbor_index]['vertex']
        #                     x_2, y_2 = self.ground_truth_map.map_frame_to_image_coordinates([x_22, y_22])
        #                     plt.plot((x_1, x_2), (y_1, y_2), color='green', linewidth=0.5)
            
        #     plt.show()
        #     rospy.sleep(100)

        # find initial and goal nodes
        for node in list(self.radius_voronoi_graph.nodes):
            # node our initial points and we will find farthest node
            cost_copy = costs.copy()
            initial_node = node
            # looking for farthest goal is related to major radius for given inital node
            for i in range(len(list(self.radius_voronoi_graph.nodes))):
                farthest_node = max(cost_copy[node].items(), key=operator.itemgetter(1))[0]
                if self.voronoi_graph.nodes[farthest_node]['radius'] > major_safety_factor * self.robot_major_radius :
                    max_node_cost = cost_copy[node][farthest_node]
                    break
                else:
                    del cost_copy[node][farthest_node]
                    i+=1
                # radius_voronoi_nodes = list(filter(lambda n: self.voronoi_graph.nodes[n]['radius'] > 0.65, self.voronoi_graph))
                # print("Max Costs[{}][{}] = {}".format(node, farthest_node, max_node_cost))

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

            # random points selection part
            if 0 <= self.random_points < len(self.radius_voronoi_graph.nodes)-1:
                random_final_point_list = random.sample(list(set(list(self.radius_voronoi_graph.nodes)) - set(remove_list)), self.random_points)
            else:
                self.random_points = len(self.radius_voronoi_graph.nodes)-2
                random_final_point_list = random.sample(list(set(list(self.radius_voronoi_graph.nodes)) - set(remove_list)), self.random_points)
                print_error("Cannot select random points more than nodes. Random point changed: ",self.random_points)
                # rospy.signal_shutdown("Signal shutdown because of too much random points")
                # break

            for goal_node in random_final_point_list:
                goal_node_list.append(goal_node)
                shortest_path_for_goal = nx.shortest_path(self.real_voronoi_graph, initial_node, goal_node)
                shortest_path_ord_dict[goal_node] = shortest_path_for_goal   
                node_cost_for_goal = costs[initial_node][goal_node]
                goal_and_distance_dict[goal_node] = node_cost_for_goal

            self.initial_goal_dict[initial_node] = goal_node_list
            self.shortest_path_dict[initial_node] = shortest_path_ord_dict
            self.voronoi_distance_dict[initial_node] = goal_and_distance_dict

            node_diameter_list = list()
            for goal_key, path_nodes in shortest_path_ord_dict.items():
                for path_node in path_nodes:
                    node_diameter = self.real_voronoi_graph.nodes[path_node]['radius']*2
                    node_diameter_list.append(node_diameter)
                node_diameter_mean = np.mean(node_diameter_list)
                minimum_node_diameter = min(node_diameter_list)
                normalized_node_diameter_mean = node_diameter_mean/self.robot_major_radius

                initial_node_pose, goal_node_pose = self.pose_finder(initial_node, goal_key)

                self.mean_passage_width_callback(initial_node, goal_key, initial_node_pose, goal_node_pose, node_diameter_mean)
                self.mean_normalized_passage_width_callback(initial_node, goal_key, initial_node_pose, goal_node_pose, normalized_node_diameter_mean)
                self.minimum_passage_width_callback(initial_node, goal_key, initial_node_pose, goal_node_pose, minimum_node_diameter)
        if self.voronoi_visualize:
            point = Point32()
            self.point_cloud = PointCloud()
            self.point_cloud.header.stamp = rospy.Time.now()
            self.point_cloud.header.frame_id = 'map'
            for pose in list(self.real_voronoi_graph.nodes):
                point.x, point.y = self.real_voronoi_graph.nodes[pose]['vertex']
                point.z = 0.0
                self.point_cloud.points.append(Point32(point.x, point.y, point.z))

            point2 = Point32()
            self.radius_voronoi_point_cloud = PointCloud()
            self.radius_voronoi_point_cloud.header.stamp = rospy.Time.now()
            self.radius_voronoi_point_cloud.header.frame_id = 'map'
            for pose in list(self.radius_voronoi_graph.nodes):
                point2.x, point2.y = self.radius_voronoi_graph.nodes[pose]['vertex']
                point2.z = 0.0
                self.radius_voronoi_point_cloud.points.append(Point32(point2.x, point2.y, point2.z))

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


    def active_cb(self):
        self.execution_timer = rospy.Time.now().to_time()
        rospy.loginfo("Goal pose " + str(self.goal_send_count) + " is now being processed by the Action Server...")

    def done_cb(self, status, result):

        if status == GoalStatus.PREEMPTED:
            rospy.loginfo("Goal pose " + str(
                self.goal_send_count) + " received a cancel request after it started executing, and has since completed its execution!")
            rospy.sleep(1.0)
        elif status == GoalStatus.ABORTED:
            rospy.loginfo("Goal pose " + str(self.goal_send_count) + " was aborted by the Action Server")
            self.path_aborted = True
            rospy.sleep(1.0)
        elif status == GoalStatus.PENDING:
            rospy.logwarn("Goal pose " + str(self.goal_send_count) + " has yet to be processed by the action server")
            self.path_aborted = True
        elif status == GoalStatus.ACTIVE:
            rospy.logwarn(
                "Goal pose " + str(self.goal_send_count) + " is currently being processed by the action server")
        elif status == GoalStatus.SUCCEEDED:
            rospy.logwarn("Goal pose " + str(self.goal_send_count) + " was achieved successfully by the action server")
        elif status == GoalStatus.REJECTED:
            rospy.logwarn("Goal pose " + str(
                self.goal_send_count) + " was rejected by the action server without being processed, because the goal was unattainable or invalid")
            self.path_aborted = True
        elif status == GoalStatus.PREEMPTING:
            rospy.logwarn("Goal pose " + str(
                self.goal_send_count) + " received a cancel request after it started executing and has not yet completed execution")
            self.path_aborted = True
        elif status == GoalStatus.RECALLING:
            rospy.logwarn("Goal pose " + str(
                self.goal_send_count) + " received a cancel request before it started executing but the action server has not yet confirmed that the goal is canceled")
            self.path_aborted = True
        elif status == GoalStatus.RECALLED:
            rospy.logwarn("Goal pose " + str(
                self.goal_send_count) + " received a cancel request before it started executing and was successfully cancelled")
            self.path_aborted = True
        elif status == GoalStatus.LOST:
            rospy.logwarn("Goal pose " + str(self.goal_send_count) + " received a cancel request. Goal lost!")
            self.path_aborted = True
        else:
            rospy.logerr("There is no GoalStatus")
            self.path_aborted = True

    def start_run(self, initial_node, goal_node):
        print_info("prepare start run for each path ")
        self.write_event('prepare_start_run_for_each_path', rospy.Time.now().to_sec())
        self.send_initial_node = initial_node
        self.send_goal_node = goal_node

        initial_node_pose, goal_node_pose = self.pose_finder(start_node=initial_node, final_node=goal_node)

        self.send_initial_node_pose = initial_node_pose
        self.send_goal_node_pose = goal_node_pose

        self.voronoi_distance_callback(i_point=initial_node, g_point=goal_node, i_node_pose=initial_node_pose, g_node_pose=goal_node_pose)
        #self.longest_path_publisher_callback(initial_node)
        if self.voronoi_visualize:
            self.voronoi_publisher.publish(self.point_cloud)
        # TODO check plan or pointcloud
        # self.vor_path_pub.publish(self.my_path)
            self.radius_voronoi_publisher.publish(self.radius_voronoi_point_cloud)

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

        initial_node_pose_stamped.header.stamp = rospy.Time.now()  # before publishing adding time stamp
        self.initial_pose_publisher.publish(initial_node_pose_stamped)
        self.initial_pose_callback(initial_node_pose_stamped)

        rospy.sleep(1.0)

        # self.traversal_path_publisher.publish(self.traversal_path_msg)  #traversal path publisher for visualization

        self.write_event('start_run_for_each_path', rospy.Time.now().to_sec())

        # goal node send
        if not self.navigate_to_pose_action_client.wait_for_server(
                timeout=rospy.Duration.from_sec(100.0)):  # just for control duration time is not important in here
            self.write_event('failed_to_communicate_with_navigation_node', rospy.Time.now().to_sec())
            raise RunFailException("navigate_to_pose action server not available")

        maxGoalPose = MoveBaseGoal()
        maxGoalPose.target_pose.header.frame_id = self.fixed_frame
        maxGoalPose.target_pose.header.stamp = rospy.Time.now()
        maxGoalPose.target_pose.pose = goal_node_pose

        self.goal_send_count += 1
        print("counter:{}/{}. For initial node {}, sending goal node {} ".format(self.goal_send_count, self.all_path_counter, initial_node, goal_node))
        # self.navigate_to_pose_action_client.send_goal_and_wait(maxGoalPose,execute_timeout=rospy.Duration.from_sec(1.0)),
        self.navigate_to_pose_action_client.send_goal(maxGoalPose, done_cb=self.done_cb, active_cb=self.active_cb)
        self.write_event('target_pose_set', rospy.Time.now().to_sec())
        # self.navigate_to_pose_action_client.wait_for_result(rospy.Duration.from_sec(1.0))

        rospy.sleep(0.5)
        while not rospy.is_shutdown():
            if self.path_receive:
                self.path_receive = False
                self.path_distance_token = True
                self.feasible_paths += 1
                feasibility_token = True
                self.feasibility_rate(initial_node, goal_node, initial_node_pose, goal_node_pose, feasibility_token)
                self.navigate_to_pose_action_client.cancel_all_goals()
                self.navigate_to_pose_action_client.wait_for_result(rospy.Duration.from_sec(5.0))
                self.write_event('target_pose_reached', rospy.Time.now().to_sec())
                self.write_event('finish_run_for_each_path', rospy.Time.now().to_sec())
                print_info("PATH RECEIVED")
                break
            if self.path_aborted:
                self.path_aborted = False
                self.aborted_path_counter += 1
                self.unfeasible_paths += 1
                feasibility_token = False
                self.feasibility_rate(initial_node, goal_node, initial_node_pose, goal_node_pose, feasibility_token)
                self.navigate_to_pose_action_client.cancel_all_goals()
                self.navigate_to_pose_action_client.wait_for_result(rospy.Duration.from_sec(5.0))
                self.write_event('target_pose_aborted', rospy.Time.now().to_sec())
                self.write_event('finish_run_for_each_path', rospy.Time.now().to_sec())
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
                self.write_event('initial_pose_true', rospy.Time.now().to_sec())
            else:
                self.write_event('away_from_initial_pose', rospy.Time.now().to_sec())
                print_error("current position farther from initial position than tolerance")

            latest_goal_path_pose = self.latest_path.poses[-1]
            distance_from_goal = np.sqrt((goal_node_pose.position.x - latest_goal_path_pose.pose.position.x) ** 2 + (
                    goal_node_pose.position.y - latest_goal_path_pose.pose.position.y) ** 2)
            print_info("Distance: ", distance_from_goal)
            if distance_from_goal < self.goal_tolerance:
                self.write_event('goal_pose_true', rospy.Time.now().to_sec())
            else:
                self.write_event('away_from_goal_pose', rospy.Time.now().to_sec())
                print_error("current position farther from goal position than tolerance")

        if self.goal_send_count == len(self.initial_goal_dict)*(self.random_points + 1):
            self.write_event('run_completed', rospy.Time.now().to_sec())
            rospy.signal_shutdown("run_completed")

    # def longest_path_publisher_callback(self, initial_node):
    #
    #     # self.my_path = Path()
    #     # self.my_path.header.stamp = rospy.Time.now()
    #     # self.my_path.header.frame_id = 'map'
    #     #
    #     # for goal_key, path_nodes in self.shortest_path_dict[initial_node].items():
    #     #     for pose2 in path_nodes:
    #     #
    #     #         stamped_pose = PoseStamped()
    #     #         stamped_pose.header.frame_id = 'map'
    #     #         stamped_pose.header.stamp = rospy.Time.now()
    #     #         point2 = Point32()
    #     #         point2.x, point2.y = self.real_voronoi_graph.nodes[pose2]['vertex']
    #     #         point2.z = 0.0
    #     #
    #     #         stamped_pose.pose.position = point2
    #     #         self.my_path.poses.append(PoseStamped(stamped_pose.header, stamped_pose.pose))
    #
    #     point = Point32()
    #     self.shortest_path_point_cloud = PointCloud()
    #     self.shortest_path_point_cloud.header.stamp = rospy.Time.now()
    #     self.shortest_path_point_cloud.header.frame_id = 'map'
    #
    #     for goal_key, path_nodes in self.shortest_path_dict[initial_node].items():
    #         for pose2 in path_nodes:
    #             point.x, point.y = self.real_voronoi_graph.nodes[pose2]['vertex']
    #             point.z = 0.0
    #             self.shortest_path_point_cloud.points.append(Point32(point.x, point.y, point.z))


    def tfTimerCallback(self, event):
        self.transformStamped.header.stamp = rospy.Time.now()
        self.broadcaster.sendTransform(self.transformStamped)
        # print(event)

    def pathCallback(self, path_message):
        if not path_message.poses:
            self.path_aborted = True
        else: 
            self.execution_timer2 = rospy.Time.now().to_time()
            if not self.path_receive:
                self.pathCounter += 1
                # print("sending path message ", pathMessage)
                print("Path message received for initial node {}, sending goal node {} ".format(self.send_initial_node, self.send_goal_node))
                self.execution_time_callback(self.send_initial_node, self.send_goal_node, self.send_initial_node_pose, self.send_goal_node_pose, self.execution_timer, self.execution_timer2)
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
            voronoi_dist = self.voronoi_distance_dict[i_point][g_point]    # goal -> key      voronoi -> value

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
            rospy.loginfo("Global Planning execution time: " + str(msg_time))

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

        # if self.total_paths == (self.feasible_paths + self.unfeasible_paths):
        #
        #
        #     feasibility_rate = float(self.feasible_paths) / self.total_paths
        #     print_info("Feasibility rate of this map is: ", feasibility_rate)
        #     with open(self.feasibility_rate_file_path, 'a') as feasibility_rate_file:
        #         feasibility_rate_file.write("{f}\n".format(f=feasibility_rate))
        # else:
        #     print_error("Total paths is not equal to sum of feasible and unfeasible paths.")

    def ros_shutdown_callback(self):
        """
        This function is called when the node receives an interrupt signal (KeyboardInterrupt).
        """
        print_info("asked to shutdown, terminating run")
        self.write_event('ros_shutdown', rospy.Time.now().to_sec())
        self.write_event('supervisor_finished', rospy.Time.now().to_sec())

    # def init_run_events_file(self):    # closed because we return to pandas data frame.
    #     backup_file_if_exists(self.run_events_file_path)
    #     try:
    #         with open(self.run_events_file_path, 'w') as run_events_file:
    #             run_events_file.write("{t}, {event}\n".format(t='timestamp', event='event'))
    #     except IOError:
    #         rospy.logerr("slam_benchmark_supervisor.init_event_file: could not write header to run_events_file")
    #         rospy.logerr(traceback.format_exc())
    #
    # def write_event(self, event):
    #     t = rospy.Time.now().to_sec()
    #     print_info("t: {t}, event: {event}".format(t=t, event=str(event)))
    #     try:
    #         with open(self.run_events_file_path, 'a') as run_events_file:
    #             run_events_file.write("{t}, {event}\n".format(t=t, event=str(event)))
    #     except IOError:
    #         rospy.logerr(
    #             "slam_benchmark_supervisor.write_event: could not write event to run_events_file: {t} {event}".format(
    #                 t=t, event=str(event)))
    #         rospy.logerr(traceback.format_exc())

    def write_event(self, event, time):
        backup_file_if_exists(self.run_events_file_path)
        print_info("t: {t}, event: {event}".format(t=time, event=str(event)))
        try:
            self.run_events_df = self.run_events_df.append({
                'time': time,
                'event': event
            }, ignore_index=True)
        except IOError:
            rospy.logerr(
                "slam_benchmark_supervisor.write_event: could not write event to run_events_file: {t} {event}".format(
                    t=time, event=str(event)))
            rospy.logerr(traceback.format_exc())

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

    def run_timeout_callback(self, _):
        print_error("terminating supervisor due to timeout, terminating run")
        self.write_event('run_timeout', rospy.Time.now().to_sec())
        self.write_event('supervisor_finished', rospy.Time.now().to_sec())
        rospy.signal_shutdown("Signal Shutdown because of run time out.")
        raise RunFailException("run_timeout")
