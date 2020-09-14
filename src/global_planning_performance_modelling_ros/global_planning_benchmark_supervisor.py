#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
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
import pandas as pd
import pyquaternion
import geometry_msgs
import operator

import rospy
import tf2_ros
from collections import OrderedDict
from actionlib import SimpleActionClient
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction, MoveBaseActionGoal
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Quaternion, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from std_srvs.srv import Empty

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


class GlobalPlanningBenchmarkSupervisor:
    def __init__(self):

        # topics, services, actions, entities and frames names
        ground_truth_pose_topic = rospy.get_param('~ground_truth_pose_topic')
        initial_pose_topic = rospy.get_param('~initial_pose_topic')  # /initialpose
        navFn_topic = rospy.get_param('~navFnROS_topic')  # /move_base/NavfnROS/plan
        global_planner_topic = rospy.get_param('~global_planner_topic')  # /move_base/GlobalPlanner/plan
        goal_pose_topic = rospy.get_param('~goal_pose_topic')  # /move_base/goal
        navigate_to_pose_action = rospy.get_param('~navigate_to_pose_action')  # /move_base
        self.fixed_frame = rospy.get_param('~fixed_frame')
        self.child_frame = rospy.get_param('~child_frame')
        self.robot_base_frame = rospy.get_param('~robot_base_frame')
        self.robot_entity_name = rospy.get_param('~robot_entity_name')
        self.robot_radius = rospy.get_param('~robot_radius')

        # file system paths
        self.run_output_folder = rospy.get_param('~run_output_folder')
        self.benchmark_data_folder = path.join(self.run_output_folder, "benchmark_data")
        self.plan_output_folder = path.join(self.benchmark_data_folder, "plan_output")
        self.ground_truth_map_info_path = rospy.get_param('~ground_truth_map_info_path')

        # run parameters
        run_timeout = rospy.get_param('~run_timeout')  # when the robot stucked we can use time out so according to time out we can finish the run
        # self.waypoint_timeout = rospy.get_param('~waypoint_timeout')
        self.ground_truth_map = ground_truth_map.GroundTruthMap(self.ground_truth_map_info_path)

        # run variables
        self.initial_pose = None
        self.goal_sent_count = 0
        self.path_recieve = False
        self.pathCounter = 0
        self.initial_pose_counter = 1
        self.goal_tolerance = 0.2
        self.execution_timer = 0
        self.execution_timer2 = 0

        # prepare folder structure
        if not path.exists(self.benchmark_data_folder):
            os.makedirs(self.benchmark_data_folder)

        if not path.exists(self.plan_output_folder):
            os.makedirs(self.plan_output_folder)

        # file paths for benchmark data    #we are using it for collecting benchmark data #in my case we can just keep path for example
        self.run_events_file_path = path.join(self.benchmark_data_folder, "run_events.csv")  # I keep it for example
        self.initial_pose_file_path = path.join(self.plan_output_folder, "initial_pose.csv")
        self.goal_pose_file_path = path.join(self.plan_output_folder, "goal_pose.csv")
        self.received_plan_file_path = path.join(self.plan_output_folder, "received_plan.csv")
        self.execution_time_file_path = path.join(self.plan_output_folder, "execution_time.csv")
        self.init_run_events_file()
        # self.estimated_poses_file_path = path.join(self.benchmark_data_folder, "estimated_poses.csv")

        self.voronoi_graph_node_finder()  # finding all initial nodes and goal nodes

        # pandas dataframes for benchmark data   #not useful right now #just hold for example
        # self.estimated_poses_df = pd.DataFrame(columns=['t', 'x', 'y', 'theta'])
        # self.estimated_correction_poses_df = pd.DataFrame(columns=['t', 'x', 'y', 'theta', 'cov_x_x', 'cov_x_y', 'cov_y_y', 'cov_theta_theta'])
        # self.ground_truth_poses_df = pd.DataFrame(columns=['t', 'x', 'y', 'theta', 'v_x', 'v_y', 'v_theta'])

        # setup publishers
        self.initial_pose_publisher = rospy.Publisher(initial_pose_topic, PoseWithCovarianceStamped, queue_size=10)
        # self.traversal_path_publisher = rospy.Publisher("~/traversal_path", Path, latch=True, queue_size=1) #closed right now #for visualization

        # setup subscribers
        rospy.Subscriber(initial_pose_topic, PoseWithCovarianceStamped, self.initial_pose_Callback)
        rospy.Subscriber(navFn_topic, Path, self.pathCallback)
        rospy.Subscriber(global_planner_topic, Path, self.pathCallback)
        rospy.Subscriber(goal_pose_topic, MoveBaseActionGoal, self.goal_Callback)
        # you can add your subscribers here
        # you can add subscriber path here

        # setup action clients
        self.navigate_to_pose_action_client = SimpleActionClient(navigate_to_pose_action, MoveBaseAction)  # navigate_to_pose_action => /move_base

        # setup timers
        self.tfTimer = rospy.Timer(rospy.Duration().from_sec(1.0 / 50), self.tfTimerCallback)
        # rospy.Timer(rospy.Duration.from_sec(run_timeout), self.run_timeout_callback)  #callback just holding for example
        # rospy.Timer(rospy.Duration.from_sec(ps_snapshot_period), self.ps_snapshot_timer_callback)  #callback just holding for example

        # tf send
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.transformStamped = geometry_msgs.msg.TransformStamped()

        
        for initial_node_key, goal_node_value in self.initial_goal_dict.items():
            # send initial node and goal node
            self.start_run(initial_node=initial_node_key, goal_node=goal_node_value)
            rospy.sleep(1.0)

        # #only one node send part just for debuging  (close upper for loop)
        # initial_node_key = list(self.initial_goal_dict)[0]
        # goal_node_key = self.initial_goal_dict[initial_node_key]
        # self.start_run(initial_node = initial_node_key , goal_node = goal_node_key)

    # collect all initial nodes and goal nodes in dictionary
    def voronoi_graph_node_finder(self):
        print_info("Entered -> deleaved reduced Voronoi graph from ground truth map")

        # get deleaved reduced Voronoi graph from ground truth map)
        # (in here we are taking a list from voronoi graph we will use later it
        # (but right now we can add only one goal here then we can add volonoi graph after one goal achieved)
        voronoi_graph = self.ground_truth_map.deleaved_reduced_voronoi_graph(
            minimum_radius=2 * self.robot_radius).copy()
        minimum_length_paths = nx.all_pairs_dijkstra_path(voronoi_graph, weight='voronoi_path_distance')
        minimum_length_costs = dict(nx.all_pairs_dijkstra_path_length(voronoi_graph, weight='voronoi_path_distance'))
        costs = defaultdict(dict)

        for i, paths_dict in minimum_length_paths:
            for j in paths_dict.keys():
                if i != j:
                    costs[i][j] = minimum_length_costs[i][j]
                    # print("Costs[{}][{}]: {}".format(i,j,costs[i][j]))

        # in case the graph has multiple unconnected components, remove the components with less than two nodes
        too_small_voronoi_graph_components = list(
            filter(lambda component: len(component) < 2, nx.connected_components(voronoi_graph)))

        for graph_component in too_small_voronoi_graph_components:
            voronoi_graph.remove_nodes_from(graph_component)

        if len(voronoi_graph.nodes) < 2:
            self.write_event('insufficient_number_of_nodes_in_deleaved_reduced_voronoi_graph')
            raise RunFailException(
                "insufficient number of nodes in deleaved_reduced_voronoi_graph, can not generate traversal path")

        self.initial_goal_dict = OrderedDict()  # initial_goal_dict => key: initial node, value: goal node
        self.initial_pose_dict = OrderedDict()  # initial_pose_dict => key: initial node, value: initial pose
        self.goal_pose_dict = OrderedDict()  # goal_pose_dict => key: goal node, value: goal pose

        # find initial and goal nodes and its pose
        for node in list(voronoi_graph.nodes):
            # node our initial poits and we will find farthest node
            farhest_node = max(costs[node].items(), key=operator.itemgetter(1))[0]
            max_node_cost = costs[node][farhest_node]
            # print("Max Costs[{}][{}] = {}".format(node, farhest_node, max_node_cost))

            initial_node_pose = Pose()
            initial_node_pose.position.x, initial_node_pose.position.y = voronoi_graph.nodes[node]['vertex']
            q = pyquaternion.Quaternion(axis=[0, 0, 1], radians=np.random.uniform(-np.pi, np.pi))
            initial_node_pose.orientation = Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)
            # print("node number:", node, "INITIAL:" , initial_node_pose_stamped)

            goal_node_pose = Pose()
            goal_node_pose.position.x, goal_node_pose.position.y = voronoi_graph.nodes[farhest_node]['vertex']
            q = pyquaternion.Quaternion(axis=[0, 0, 1], radians=np.random.uniform(-np.pi, np.pi))
            goal_node_pose.orientation = Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)
            # print("fathest node number:",farhest_node, "GOAL:", goal_node_pose)

            self.initial_goal_dict[node] = farhest_node
            self.initial_pose_dict[node] = initial_node_pose
            self.goal_pose_dict[farhest_node] = goal_node_pose

    def active_cb(self):
        rospy.loginfo("Goal pose " + str(self.goal_sent_count) + " is now being processed by the Action Server...")

    def done_cb(self, status, result):
        if status == 2:
            rospy.loginfo("Goal pose " + str(self.goal_sent_count) + " received a cancel request after it started executing, completed execution!")
            rospy.sleep(1.0)

        if status == 3:
            rospy.loginfo("Goal pose " + str(self.goal_sent_count) + " reached")  # ask question why it is never reach goal???

        if status == 4:
            rospy.loginfo("Goal pose " + str(self.goal_sent_count) + " was aborted by the Action Server")
            rospy.signal_shutdown("Goal pose " + str(self.goal_sent_count) + " aborted, shutting down!")
            return

        if status == 5:
            rospy.loginfo("Goal pose " + str(self.goal_sent_count) + " has been rejected by the Action Server")
            rospy.signal_shutdown("Goal pose " + str(self.goal_sent_count) + " rejected, shutting down!")
            return

        if status == 8:
            rospy.loginfo("Goal pose " + str(
                self.goal_sent_count) + " received a cancel request before it started executing, successfully cancelled!")

    def start_run(self, initial_node, goal_node):
        print_info("preparing to start run")
        self.sended_initial_node = initial_node

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
        print("TF transform OK for node:", initial_node)

        initial_node_pose_stamped.header.stamp = rospy.Time.now()  # before publishing adding time stamp
        self.initial_pose_publisher.publish(initial_node_pose_stamped)

        
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

        # self.navigate_to_pose_action_client.send_goal_and_wait(maxGoalPose,execute_timeout=rospy.Duration.from_sec(1.0)),
        self.execution_timer = rospy.Time.now().to_time()
        self.navigate_to_pose_action_client.send_goal(maxGoalPose, done_cb=self.done_cb, active_cb=self.active_cb)
        self.write_event('target_pose_set')
        # self.navigate_to_pose_action_client.wait_for_result(rospy.Duration.from_sec(1.0))
        self.goal_sent_count += 1
        print("counter:{} sending goal node {} ".format(self.goal_sent_count, goal_node))

        rospy.sleep(0.5)
        while not rospy.is_shutdown():
            if self.path_recieve:
                self.path_recieve = False
                self.navigate_to_pose_action_client.cancel_all_goals()
                self.navigate_to_pose_action_client.wait_for_result(rospy.Duration.from_sec(5.0))
                print_info("PATH RECIEVED")
                break

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

        # if  self.goal_sent_count == len(self.initial_goal_dict):
        #    rospy.signal_shutdown("run_completed")

    def tfTimerCallback(self, event):
        self.transformStamped.header.stamp = rospy.Time.now()
        self.broadcaster.sendTransform(self.transformStamped)
        # print(event)

    def pathCallback(self, pathMessage):
        if pathMessage.header.seq % 2 == 1:
            self.pathCounter += 1
            # print("sendign path message ", pathMessage)
            print("Path message recieved")
            self.execution_timer2 = rospy.Time.now().to_time()
            self.execution_time_Callback(self.sended_initial_node, self.execution_timer,self.execution_timer2)
            self.path_recieve = True
            self.latest_path = pathMessage
            # msg_time = pathMessage.header.stamp.to_sec()
            for pose in pathMessage.poses:
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

    def initial_pose_Callback(self, initial_pose_msg):
        print("WRITINGGG", self.initial_pose_counter)
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

    def goal_Callback(self, goal_msg):
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

    def execution_time_Callback(self, i_point, time_message, time_message2):
        msg_time = time_message2 - time_message
        initial_one = i_point
        goal_one = self.initial_goal_dict[i_point]
        with open(self.execution_time_file_path, 'a') as time_file:
            time_file.write(
                "{i_point}, {g_point}, {t}\n".format(
                    i_point=initial_one,
                    g_point=goal_one,
                    t=msg_time,
                ))

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
        # self.estimated_poses_df.to_csv(self.estimated_poses_file_path, index=False)

    """def run_timeout_callback(self, _):
        print_error("terminating supervisor due to timeout, terminating run")
        self.write_event('run_timeout')
        self.write_event('supervisor_finished')
        raise RunFailException("run_timeout")"""
