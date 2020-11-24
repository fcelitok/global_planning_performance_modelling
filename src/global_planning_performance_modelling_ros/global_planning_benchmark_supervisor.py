#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import traceback
from collections import defaultdict, deque
import copy
import pickle
import psutil
import os
from os import path
import numpy as np
import networkx as nx
import pyquaternion
import geometry_msgs
import operator
import pandas as pd

import rospy
import tf2_ros
from collections import OrderedDict
from actionlib import SimpleActionClient
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction, MoveBaseActionGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Quaternion, PoseStamped, TransformStamped
from nav_msgs.msg import Path

from performance_modelling_py.environment import ground_truth_map
from performance_modelling_py.utils import backup_file_if_exists, print_info, print_error


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
        sbpl_lattice_planner_topic = rospy.get_param('~sbpl_lattice_planner_topic') # /move_base/SBPLLatticePlanner/plan
        ompl_planner_topic = rospy.get_param('~ompl_planner_topic')  # /move_base/OmplGlobalPlanner/plan
        goal_pose_topic = rospy.get_param('~goal_pose_topic')  # /move_base/goal
        navigate_to_pose_action = rospy.get_param('~navigate_to_pose_action')  # /move_base
        self.fixed_frame = rospy.get_param('~fixed_frame')
        self.child_frame = rospy.get_param('~child_frame')
        self.robot_base_frame = rospy.get_param('~robot_base_frame')
        self.robot_entity_name = rospy.get_param('~robot_entity_name')
        self.robot_radius = rospy.get_param('~robot_radius')
        self.goal_tolerance = rospy.get_param(('~goal_tolerance'))

        # file system paths
        self.run_output_folder = rospy.get_param('~run_output_folder')
        self.benchmark_data_folder = path.join(self.run_output_folder, "benchmark_data")
        self.plan_output_folder = path.join(self.benchmark_data_folder, "plan_output")
        self.ground_truth_map_info_path = rospy.get_param('~ground_truth_map_info_path')

        # run parameters
        run_timeout = rospy.get_param('~run_timeout')  # when the robot stacked we can use time out so according to time out we can finish the run
        self.ground_truth_map = ground_truth_map.GroundTruthMap(self.ground_truth_map_info_path)

        # run variables
        self.initial_pose = None
        self.goal_send_count = 0
        self.aborted_path_counter = 0
        self.path_receive = False
        self.path_aborted = False
        self.path_distance_token = False
        self.pathCounter = 0
        self.initial_pose_counter = 1
        self.execution_timer = 0
        self.execution_timer2 = 0

        self.total_paths = 0
        self.feasible_paths = 0
        self.unfeasible_paths = 0

        self.initial_goal_dict = OrderedDict()  # initial_goal_dict => key: initial node, value: goal node
        self.initial_pose_dict = OrderedDict()  # initial_pose_dict => key: initial node, value: initial pose
        self.goal_pose_dict = OrderedDict()  # goal_pose_dict => key: goal node, value: goal pose
        self.voronoi_distance_dict = OrderedDict()  # voronoi_distance_dict => key : initial node, value: list(goal node, voronoi distance)

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
        self.init_run_events_file()

        self.voronoi_graph_node_finder()  # finding all initial nodes and goal nodes

        # pandas dataframes for benchmark data   #not useful right now #just hold for example
        self.execution_time_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'time'])
        self.voronoi_distance_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'voronoi_distance'])
        self.euclidean_distance_df = pd.DataFrame(columns=['node_i', 'i_x', 'i_y', 'node_g', 'g_x', 'g_y', 'euclidean_distance'])

        # setup publishers
        self.initial_pose_publisher = rospy.Publisher(initial_pose_topic, PoseWithCovarianceStamped, queue_size=10)
        # self.traversal_path_publisher = rospy.Publisher("~/traversal_path", Path, latch=True, queue_size=1) #closed right now #for visualization

        # setup subscribers
        rospy.Subscriber(navFn_topic, Path, self.pathCallback)
        rospy.Subscriber(global_planner_topic, Path, self.pathCallback)
        rospy.Subscriber(sbpl_lattice_planner_topic, Path, self.pathCallback)
        rospy.Subscriber(ompl_planner_topic, Path, self.pathCallback)
        rospy.Subscriber(goal_pose_topic, MoveBaseActionGoal, self.goal_callback)
        # you can add your subscribers here
        # you can add subscriber path here

        # setup action clients
        self.navigate_to_pose_action_client = SimpleActionClient(navigate_to_pose_action, MoveBaseAction)  # navigate_to_pose_action => /move_base

        # setup timers
        rospy.Timer(rospy.Duration.from_sec(run_timeout), self.run_timeout_callback)
        self.tfTimer = rospy.Timer(rospy.Duration().from_sec(1.0 / 50), self.tfTimerCallback)

        # tf send
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.transformStamped = geometry_msgs.msg.TransformStamped()

        # #only one node send part just for debugging  (close upper for loop)
        # initial_node_key = list(self.initial_goal_dict)[0]
        # goal_node_key = self.initial_goal_dict[initial_node_key]
        # self.start_run(initial_node = initial_node_key , goal_node = goal_node_key)

        # send initial node and goal node
        for initial_node_key, goal_node_value in self.initial_goal_dict.items():
            self.start_run(initial_node=initial_node_key, goal_node=goal_node_value)
            rospy.sleep(1.0)

    # collect all initial nodes and goal nodes in dictionary
    def voronoi_graph_node_finder(self):
        print_info("Entered -> deleaved reduced Voronoi graph from ground truth map")

        # get deleaved reduced Voronoi graph from ground truth map)
        # (in here we are taking a list from voronoi graph we will use later it
        # (but right now we can add only one goal here then we can add voronoi graph after one goal achieved)
        voronoi_graph = self.ground_truth_map.deleaved_reduced_voronoi_graph(minimum_radius=2 * self.robot_radius).copy()
        self.total_paths = len(voronoi_graph.nodes)
        minimum_length_paths = nx.all_pairs_dijkstra_path(voronoi_graph, weight='voronoi_path_distance')
        minimum_length_costs = dict(nx.all_pairs_dijkstra_path_length(voronoi_graph, weight='voronoi_path_distance'))
        costs = defaultdict(dict)

        for i, paths_dict in minimum_length_paths:
            for j in paths_dict.keys():
                if i != j:
                    costs[i][j] = minimum_length_costs[i][j]
                    # print("Costs[{}][{}]: {}".format(i, j, costs[i][j]))

        # in case the graph has multiple unconnected components, remove the components with less than two nodes
        too_small_voronoi_graph_components = list(
            filter(lambda component: len(component) < 2, nx.connected_components(voronoi_graph)))

        for graph_component in too_small_voronoi_graph_components:
            voronoi_graph.remove_nodes_from(graph_component)

        if len(voronoi_graph.nodes) < 2:
            self.write_event('insufficient_number_of_nodes_in_deleaved_reduced_voronoi_graph')
            raise RunFailException(
                "insufficient number of nodes in deleaved_reduced_voronoi_graph, can not generate traversal path")

        # find initial and goal nodes and its pose
        for node in list(voronoi_graph.nodes):
            # node our initial points and we will find farthest node
            farthest_node = max(costs[node].items(), key=operator.itemgetter(1))[0]
            max_node_cost = costs[node][farthest_node]
            # print("Max Costs[{}][{}] = {}".format(node, farthest_node, max_node_cost))

            initial_node_pose = Pose()
            initial_node_pose.position.x, initial_node_pose.position.y = voronoi_graph.nodes[node]['vertex']
            q = pyquaternion.Quaternion(axis=[0, 0, 1], radians=np.random.uniform(-np.pi, np.pi))
            initial_node_pose.orientation = Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)
            # print("node number:", node, "INITIAL:" , initial_node_pose_stamped)

            goal_node_pose = Pose()
            goal_node_pose.position.x, goal_node_pose.position.y = voronoi_graph.nodes[farthest_node]['vertex']
            q = pyquaternion.Quaternion(axis=[0, 0, 1], radians=np.random.uniform(-np.pi, np.pi))
            goal_node_pose.orientation = Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)
            # print("farthest node number:",farthest_node, "GOAL:", goal_node_pose)

            goal_and_distance_list = []
            goal_and_distance_list.append(farthest_node)
            goal_and_distance_list.append(max_node_cost)

            self.initial_goal_dict[node] = farthest_node
            self.initial_pose_dict[node] = initial_node_pose
            self.goal_pose_dict[farthest_node] = goal_node_pose
            self.voronoi_distance_dict[node] = goal_and_distance_list

    def active_cb(self):
        self.execution_timer = rospy.Time.now().to_time()
        rospy.loginfo("Goal pose " + str(self.goal_send_count) + " is now being processed by the Action Server...")

    def done_cb(self, status, result):

        if status == GoalStatus.PREEMPTED:
            rospy.loginfo("Goal pose " + str(self.goal_send_count) + " received a cancel request after it started executing, and has since completed its execution!")
            rospy.sleep(1.0)
        elif status == GoalStatus.ABORTED:
            rospy.loginfo("Goal pose " + str(self.goal_send_count) + " was aborted by the Action Server")
            self.path_aborted = True
            rospy.sleep(1.0)
        elif status == GoalStatus.PENDING:
            rospy.loginfo("Goal pose " + str(self.goal_send_count) + " has yet to be processed by the action server")
        elif status == GoalStatus.ACTIVE:
            rospy.loginfo("Goal pose " + str(self.goal_send_count) + " is currently being processed by the action server")
        elif status == GoalStatus.SUCCEEDED:
            rospy.loginfo("Goal pose " + str(self.goal_send_count) + " was achieved successfully by the action server")
        elif status == GoalStatus.REJECTED:
            rospy.loginfo("Goal pose " + str(self.goal_send_count) + " was rejected by the action server without being processed, because the goal was unattainable or invalid")
        elif status == GoalStatus.PREEMPTING:
            rospy.loginfo("Goal pose " + str(self.goal_send_count) + " received a cancel request after it started executing and has not yet completed execution")
        elif status == GoalStatus.RECALLING:
            rospy.loginfo("Goal pose " + str(self.goal_send_count) + " received a cancel request before it started executing but the action server has not yet confirmed that the goal is canceled")
        elif status == GoalStatus.RECALLED:
            rospy.loginfo("Goal pose " + str(self.goal_send_count) + " received a cancel request before it started executing and was successfully cancelled")
        elif status == GoalStatus.LOST:
            rospy.loginfo("Goal pose " + str(self.goal_send_count) + " received a cancel request. Goal lost!")
        else:
            rospy.logerr("There is no GoalStatus")

    def start_run(self, initial_node, goal_node):
        print_info("preparing to start run")
        self.send_initial_node = initial_node

        self.voronoi_distance_callback(self.send_initial_node)

        initial_node_pose1 = self.initial_pose_dict[initial_node]
        initial_node_pose_stamped = PoseWithCovarianceStamped()
        initial_node_pose_stamped.header.frame_id = self.fixed_frame
        initial_node_pose_stamped.pose.pose = initial_node_pose1

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

        self.write_event('run_start')

        # goal node send
        if not self.navigate_to_pose_action_client.wait_for_server(timeout=rospy.Duration.from_sec(5.0)):  # just for control duration time is not important in here
            self.write_event('failed_to_communicate_with_navigation_node')
            raise RunFailException("navigate_to_pose action server not available")

        goal_node_pose1 = self.goal_pose_dict[goal_node]
        maxGoalPose = MoveBaseGoal()
        maxGoalPose.target_pose.header.frame_id = self.fixed_frame
        maxGoalPose.target_pose.header.stamp = rospy.Time.now()
        maxGoalPose.target_pose.pose = goal_node_pose1

        self.goal_send_count += 1
        print("counter:{} sending goal node {} ".format(self.goal_send_count, goal_node))
        # self.navigate_to_pose_action_client.send_goal_and_wait(maxGoalPose,execute_timeout=rospy.Duration.from_sec(1.0)),
        self.navigate_to_pose_action_client.send_goal(maxGoalPose, done_cb=self.done_cb, active_cb=self.active_cb)
        self.write_event('target_pose_set')
        # self.navigate_to_pose_action_client.wait_for_result(rospy.Duration.from_sec(1.0))

        rospy.sleep(0.5)
        while not rospy.is_shutdown():
            if self.path_receive:
                self.feasible_paths += 1
                self.path_receive = False
                self.path_distance_token = True
                self.navigate_to_pose_action_client.cancel_all_goals()
                self.navigate_to_pose_action_client.wait_for_result(rospy.Duration.from_sec(5.0))
                print_info("PATH RECEIVED")
                break
            if self.path_aborted:
                self.path_aborted = False
                self.aborted_path_counter += 1
                self.unfeasible_paths += 1
                self.navigate_to_pose_action_client.cancel_all_goals()
                self.navigate_to_pose_action_client.wait_for_result(rospy.Duration.from_sec(5.0))
                print_info("PATH ABORTED. Counter: ", self.aborted_path_counter)
                break

        if self.path_distance_token:
            self.path_distance_token = False
            latest_initial_path_pose = self.latest_path.poses[0]
            distance_from_initial = np.sqrt(
                (initial_node_pose1.position.x - latest_initial_path_pose.pose.position.x) ** 2 + (
                        initial_node_pose1.position.y - latest_initial_path_pose.pose.position.y) ** 2)
            print_info("Distance: ", distance_from_initial)
            if distance_from_initial < self.goal_tolerance:
                self.write_event('starting from initial pose')
            else:
                print_error("current position farther from initial position than tolerance")

            latest_goal_path_pose = self.latest_path.poses[-1]
            distance_from_goal = np.sqrt((goal_node_pose1.position.x - latest_goal_path_pose.pose.position.x) ** 2 + (
                    goal_node_pose1.position.y - latest_goal_path_pose.pose.position.y) ** 2)
            print_info("Distance: ", distance_from_goal)
            if distance_from_goal < self.goal_tolerance:
                self.write_event('goal pose reached')
            else:
                print_error("current position farther from goal position than tolerance")

        if self.goal_send_count == len(self.initial_goal_dict):
            self.feasibility_rate()
            rospy.signal_shutdown("run_completed")

    def tfTimerCallback(self, event):
        self.transformStamped.header.stamp = rospy.Time.now()
        self.broadcaster.sendTransform(self.transformStamped)
        # print(event)

    def pathCallback(self, path_message):
        self.execution_timer2 = rospy.Time.now().to_time()
        if not self.path_receive:
            self.pathCounter += 1
            # print("sending path message ", pathMessage)
            print("Path message received")
            self.execution_time_callback(self.send_initial_node, self.execution_timer, self.execution_timer2)
            self.euclidean_distance_callback(path_message, self.send_initial_node)
            self.path_receive = True
            self.latest_path = path_message
            # msg_time = pathMessage.header.stamp.to_sec()
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

    def euclidean_distance_callback(self, pathMessage, i_point):
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
                euclidean_distance = np.sum(np.sqrt(np.sum(np.square(x_y_position_np_array_first - x_y_position_np_array_second), axis=1)))
                #print ("Euclidean Distance of path: ", euclidean_distance)

            xposition = self.initial_pose_dict[i_point].position.x
            yposition = self.initial_pose_dict[i_point].position.y
            goal_one = self.initial_goal_dict[i_point]
            xpos = self.goal_pose_dict[goal_one].position.x
            ypos = self.goal_pose_dict[goal_one].position.y

            self.euclidean_distance_df = self.euclidean_distance_df.append({
                'node_i': i_point,
                'i_x': xposition,
                'i_y': yposition,
                'node_g': goal_one,
                'g_x': xpos,
                'g_y': ypos,
                'euclidean_distance': euclidean_distance
            }, ignore_index=True)
        except:
            print_error(traceback.format_exc())

    def initial_pose_callback(self, initial_pose_msg):
        # print("WRITING", self.initial_pose_counter)
        self.initial_pose_counter += 1
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

    def voronoi_distance_callback(self, i_point):
        try:
            voronoi_list = self.voronoi_distance_dict[i_point]
            goal_point = voronoi_list[0]
            voronoi_dist = voronoi_list[1]

            xposition = self.initial_pose_dict[i_point].position.x
            yposition = self.initial_pose_dict[i_point].position.y
            goal_one = self.initial_goal_dict[i_point]
            xpos = self.goal_pose_dict[goal_one].position.x
            ypos = self.goal_pose_dict[goal_one].position.y

            if goal_point == goal_one:
                self.voronoi_distance_df = self.voronoi_distance_df.append({
                    'node_i': i_point,
                    'i_x': xposition,
                    'i_y': yposition,
                    'node_g': goal_one,
                    'g_x': xpos,
                    'g_y': ypos,
                    'voronoi_distance': voronoi_dist
                }, ignore_index=True)
            else:
                rospy.logerr("Goal did not matched.")
                self.write_event('When writing Voronoi distance, goal did not matched')
                raise RunFailException("Goal did not matched")
        except:
            print_error(traceback.format_exc())

    def execution_time_callback(self, i_point, time_message, time_message2):
        try:
            msg_time = time_message2 - time_message
            rospy.loginfo("Global Planning execution time: " + str(msg_time))
            xposition = self.initial_pose_dict[i_point].position.x
            yposition = self.initial_pose_dict[i_point].position.y
            goal_one = self.initial_goal_dict[i_point]
            xpos = self.goal_pose_dict[goal_one].position.x
            ypos = self.goal_pose_dict[goal_one].position.y

            self.execution_time_df = self.execution_time_df.append({
                'node_i': i_point,
                'i_x': xposition,
                'i_y': yposition,
                'node_g': goal_one,
                'g_x': xpos,
                'g_y': ypos,
                'time': msg_time
            }, ignore_index=True)
        except:
            print_error(traceback.format_exc())

    def feasibility_rate(self):
        if self.total_paths == (self.feasible_paths + self.unfeasible_paths):
            feasibility_rate = float(self.feasible_paths)/self.total_paths
            print_info("Feasibility rate of this map is: ", feasibility_rate)
            with open(self.feasibility_rate_file_path, 'a') as feasibility_rate_file:
                feasibility_rate_file.write("{f}\n".format(f=feasibility_rate))
        else:
            print_error("Total paths is not equal to sum of feasible and unfeasible paths.")

    def ros_shutdown_callback(self):
        """
        This function is called when the node receives an interrupt signal (KeyboardInterrupt).
        """
        print_info("asked to shutdown, terminating run")
        self.write_event('ros_shutdown')
        self.write_event('supervisor_finished')

    def init_run_events_file(self):
        backup_file_if_exists(self.run_events_file_path)
        try:
            with open(self.run_events_file_path, 'w') as run_events_file:
                run_events_file.write("{t}, {event}\n".format(t='timestamp', event='event'))
        except IOError:
            rospy.logerr("slam_benchmark_supervisor.init_event_file: could not write header to run_events_file")
            rospy.logerr(traceback.format_exc())

    def write_event(self, event):
        t = rospy.Time.now().to_sec()
        print_info("t: {t}, event: {event}".format(t=t, event=str(event)))
        try:
            with open(self.run_events_file_path, 'a') as run_events_file:
                run_events_file.write("{t}, {event}\n".format(t=t, event=str(event)))
        except IOError:
            rospy.logerr(
                "slam_benchmark_supervisor.write_event: could not write event to run_events_file: {t} {event}".format(
                    t=t, event=str(event)))
            rospy.logerr(traceback.format_exc())

    def end_run(self):
        """
        This function is called after the run has completed, whether the run finished correctly, or there was an exception.
        The only case in which this function is not called is if an exception was raised from self.__init__
        """
        self.execution_time_df.to_csv(self.execution_time_file_path, index=False)
        self.voronoi_distance_df.to_csv(self.voronoi_distance_file_path, index=False)
        self.euclidean_distance_df.to_csv(self.euclidean_distance_file_path, index=False)

    def run_timeout_callback(self, _):
        print_error("terminating supervisor due to timeout, terminating run")
        self.write_event('run_timeout')
        self.write_event('supervisor_finished')
        rospy.signal_shutdown("Signal Shutdown because of run time out.")
        raise RunFailException("run_timeout")

