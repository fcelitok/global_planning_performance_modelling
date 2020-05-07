#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import traceback

import geometry_msgs
import lifecycle_msgs
import nav_msgs
import numpy as np
import pandas as pd
import pyquaternion
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from lifecycle_msgs.msg import TransitionEvent
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ManageLifecycleNodes
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan

import tf2_ros

import copy
import pickle
import psutil

import os
from os import path

from performance_modelling_py.utils import backup_file_if_exists, print_info, print_error, nanoseconds_to_seconds


def main(args=None):
    rclpy.init(args=args)

    node = LocalizationBenchmarkSupervisor()

    node.start_run()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.ros_shutdown_callback()
    finally:
        node.end_run()


class LocalizationBenchmarkSupervisor(Node):
    def __init__(self):
        super().__init__('localization_benchmark_supervisor', automatically_declare_parameters_from_overrides=True)

        # general parameters
        scan_topic = self.get_parameter('scan_topic').value
        ground_truth_pose_topic = self.get_parameter('ground_truth_pose_topic').value
        estimated_pose_correction_topic = self.get_parameter('estimated_pose_correction_topic').value
        initial_pose_topic = self.get_parameter('initial_pose_topic').value
        localization_node_transition_event_topic = self.get_parameter('localization_node_transition_event_topic').value
        lifecycle_manager_service = self.get_parameter('lifecycle_manager_service').value
        navigate_to_pose_action_name = self.get_parameter('navigate_to_pose_action_name').value
        run_timeout = self.get_parameter('run_timeout').value
        ps_snapshot_period = self.get_parameter('ps_snapshot_period').value
        write_estimated_poses_period = self.get_parameter('write_estimated_poses_period').value

        # run parameters
        self.run_output_folder = self.get_parameter('run_output_folder').value
        self.ps_pid_father = self.get_parameter('pid_father').value
        self.benchmark_data_folder = path.join(self.run_output_folder, "benchmark_data")
        self.ps_output_folder = path.join(self.benchmark_data_folder, "ps_snapshots")

        # run variables
        self.terminate = False
        self.ps_snapshot_count = 0
        self.ps_processes = psutil.Process(self.ps_pid_father).children(recursive=True)  # list of processes children of the benchmark script, i.e., all ros nodes of the benchmark including this one
        self.received_first_scan = False
        self.localization_node_activated = False

        initial_pose_dict = yaml.load("""
        header:
          frame_id: map
        pose:
          pose:
            position:
              x: -1.9485163688659668
              y: -0.47899842262268066
              z: 0.0
            orientation:
              x: 0.0
              y: 0.0
              z: -0.014150828880505449
              w: 0.99989987200819
          covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]
        """)

        self.initial_pose = PoseWithCovarianceStamped()
        self.initial_pose.header.frame_id = initial_pose_dict['header']['frame_id']
        self.initial_pose.pose.pose.position.x = initial_pose_dict['pose']['pose']['position']['x']
        self.initial_pose.pose.pose.position.y = initial_pose_dict['pose']['pose']['position']['y']
        self.initial_pose.pose.pose.position.z = initial_pose_dict['pose']['pose']['position']['z']
        self.initial_pose.pose.pose.orientation.x = initial_pose_dict['pose']['pose']['orientation']['x']
        self.initial_pose.pose.pose.orientation.y = initial_pose_dict['pose']['pose']['orientation']['y']
        self.initial_pose.pose.pose.orientation.z = initial_pose_dict['pose']['pose']['orientation']['z']
        self.initial_pose.pose.covariance = initial_pose_dict['pose']['covariance']

        # prepare folder structure
        if not path.exists(self.benchmark_data_folder):
            os.makedirs(self.benchmark_data_folder)

        if not path.exists(self.ps_output_folder):
            os.makedirs(self.ps_output_folder)

        # file paths for benchmark data
        self.estimated_poses_file_path = path.join(self.benchmark_data_folder, "estimated_poses.csv")
        self.estimated_correction_poses_file_path = path.join(self.benchmark_data_folder, "estimated_correction_poses_with_covariance.csv")
        self.ground_truth_poses_file_path = path.join(self.benchmark_data_folder, "ground_truth_poses.csv")
        self.scans_file_path = path.join(self.benchmark_data_folder, "scans.csv")
        self.run_events_file_path = path.join(self.benchmark_data_folder, "run_events.csv")
        self.init_run_events_file()

        # pandas dataframes for benchmark data
        self.estimated_poses_df = pd.DataFrame(columns=['t', 'x', 'y', 'theta'])
        self.estimated_correction_poses_df = pd.DataFrame(columns=['t', 'x', 'y', 'theta', 'cov_x_x', 'cov_x_y', 'cov_y_y', 'cov_theta_theta'])
        self.ground_truth_poses_df = pd.DataFrame(columns=['t', 'x', 'y', 'theta', 'v_x', 'v_y', 'v_theta'])

        # setup timers
        self.create_timer(run_timeout, self.run_timeout_callback)
        self.create_timer(ps_snapshot_period, self.ps_snapshot_timer_callback)
        self.create_timer(write_estimated_poses_period, self.write_estimated_pose_timer_callback)

        # setup service clients
        self.lifecycle_manager_service_client = self.create_client(ManageLifecycleNodes, lifecycle_manager_service)

        # setup buffers
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # setup publishers
        self.initial_pose_publisher = self.create_publisher(PoseWithCovarianceStamped, initial_pose_topic, 1)

        # setup subscribers
        self.create_subscription(LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, estimated_pose_correction_topic, self.estimated_pose_correction_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, ground_truth_pose_topic, self.ground_truth_pose_callback, qos_profile_sensor_data)
        self.localization_node_transition_event_subscriber = self.create_subscription(TransitionEvent, localization_node_transition_event_topic, self.localization_node_transition_event_callback, qos_profile_sensor_data)

        # setup action clients
        self.navigate_to_pose_action_client = ActionClient(self, NavigateToPose, navigate_to_pose_action_name)
        self.navigate_to_pose_action_goal_future = None
        self.navigate_to_pose_action_result_future = None

    def start_run(self):
        print_info("starting run")

        # wait to receive sensor data from the environment (e.g., a simulator may need time to startup)
        waiting_time = 0.0
        waiting_period = 0.5
        while not self.received_first_scan and rclpy.ok():
            time.sleep(waiting_period)
            rclpy.spin_once(self)
            waiting_time += waiting_period
            if waiting_time > 5.0:
                self.get_logger().warning('still waiting to receive first sensor message from environment')
                waiting_time = 0.0

        # ask lifecycle_manager to startup all its managed nodes
        while not self.lifecycle_manager_service_client.wait_for_service(timeout_sec=5.0) and rclpy.ok():
            self.get_logger().warning('supervisor: still waiting lifecycle_manager_service to become available')

        startup_request = ManageLifecycleNodes.Request(command=ManageLifecycleNodes.Request.STARTUP)

        # the future will complete after the localization node has received the initial pose
        initial_pose_sent = False
        srv_future = self.lifecycle_manager_service_client.call_async(startup_request)
        while not srv_future.done() and rclpy.ok():
            rclpy.spin_once(self)

            # send the initial pose as soon as the localization node is active
            if self.localization_node_activated and not initial_pose_sent:
                initial_pose_sent = True
                self.initial_pose.header.stamp = self.get_clock().now().to_msg()
                self.initial_pose_publisher.publish(self.initial_pose)

        # complete the service request
        try:
            response: ManageLifecycleNodes.Response = srv_future.result()
        except Exception as e:
            self.get_logger().fatal('Service call failed %r' % (e,))
            self.write_event(self.get_clock().now(), 'failed_to_startup_nodes')
            rclpy.shutdown()
            return
        else:
            if not response.success:
                self.get_logger().fatal('Service lifecycle manager could not startup nodes')
                self.write_event(self.get_clock().now(), 'failed_to_startup_nodes')
                rclpy.shutdown()
                return

        self.write_event(self.get_clock().now(), 'run_start')

        self.send_goal()

    def send_goal(self):
        print_info('waiting for navigate_to_pose action server')
        if not self.navigate_to_pose_action_client.wait_for_server(timeout_sec=5.0):
            print_error("navigate_to_pose action server not available")
            self.write_event(self.get_clock().now(), 'failed_to_navigate')
            return  # TODO fail run or try again?

        target_pose_dict = yaml.load("""
        header:
          frame_id: map
        pose:
          position:
            x: 0.0
            y: -0.666
            z: 0.0
          orientation:
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
        """)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = target_pose_dict['header']['frame_id']
        goal_msg.pose.pose.position.x = target_pose_dict['pose']['position']['x']
        goal_msg.pose.pose.position.y = target_pose_dict['pose']['position']['y']
        goal_msg.pose.pose.position.z = target_pose_dict['pose']['position']['z']
        goal_msg.pose.pose.orientation.x = target_pose_dict['pose']['orientation']['x']
        goal_msg.pose.pose.orientation.y = target_pose_dict['pose']['orientation']['y']
        goal_msg.pose.pose.orientation.z = target_pose_dict['pose']['orientation']['z']

        self.navigate_to_pose_action_goal_future = self.navigate_to_pose_action_client.send_goal_async(goal_msg)
        self.navigate_to_pose_action_goal_future.add_done_callback(self.goal_response_callback)
        self.write_event(self.get_clock().now(), 'target_pose_set')

    def ros_shutdown_callback(self):
        print_info("asked to shutdown, terminating run")
        self.write_event(self.get_clock().now(), 'ros_shutdown')
        self.write_event(self.get_clock().now(), 'supervisor_finished')

    def end_run(self):
        self.estimated_poses_df.to_csv(self.estimated_poses_file_path, index=False)
        self.estimated_correction_poses_df.to_csv(self.estimated_correction_poses_file_path, index=False)
        self.ground_truth_poses_df.to_csv(self.ground_truth_poses_file_path, index=False)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print_error('navigation action goal rejected')
            self.write_event(self.get_clock().now(), 'target_pose_rejected')
            return  # TODO terminate run?

        print_info('navigate action goal accepted')
        self.write_event(self.get_clock().now(), 'target_pose_accepted')

        self.navigate_to_pose_action_result_future = goal_handle.get_result_async()
        self.navigate_to_pose_action_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            print_info('navigation action finished')
            self.write_event(self.get_clock().now(), 'target_pose_reached')
        else:
            print_info('navigation action failed with status {}'.format(status))
            self.write_event(self.get_clock().now(), 'target_pose_not_reached')

        self.write_event(self.get_clock().now(), 'run_completed')
        rclpy.shutdown()

    def localization_node_transition_event_callback(self, transition_event_msg: lifecycle_msgs.msg.TransitionEvent):
        if transition_event_msg.goal_state.label == 'active':
            self.localization_node_activated = True

    def run_timeout_callback(self):
        print_info("slam_benchmark_supervisor: terminating supervisor due to timeout, terminating run")
        self.write_event(self.get_clock().now(), 'run_timeout')
        self.write_event(self.get_clock().now(), 'supervisor_finished')
        rclpy.shutdown()

    def scan_callback(self, laser_scan_msg):
        self.received_first_scan = True
        msg_time = nanoseconds_to_seconds(laser_scan_msg.header.stamp.nanosec) + float(laser_scan_msg.header.stamp.sec)
        with open(self.scans_file_path, 'a') as scans_file:
            scans_file.write("{t}, {angle_min}, {angle_max}, {angle_increment}, {range_min}, {range_max}, {ranges}\n".format(
                t=msg_time,
                angle_min=laser_scan_msg.angle_min,
                angle_max=laser_scan_msg.angle_max,
                angle_increment=laser_scan_msg.angle_increment,
                range_min=laser_scan_msg.range_min,
                range_max=laser_scan_msg.range_max,
                ranges=', '.join(map(str, laser_scan_msg.ranges))))

    def write_estimated_pose_timer_callback(self):
        try:
            transform_msg = self.tf_buffer.lookup_transform('map', 'base_link', Time())
            orientation = transform_msg.transform.rotation
            theta, _, _ = pyquaternion.Quaternion(x=orientation.x, y=orientation.y, z=orientation.z, w=orientation.w).yaw_pitch_roll

            self.estimated_poses_df = self.estimated_poses_df.append({
                't': nanoseconds_to_seconds(Time.from_msg(transform_msg.header.stamp).nanoseconds),
                'x': transform_msg.transform.translation.x,
                'y': transform_msg.transform.translation.y,
                'theta': theta
            }, ignore_index=True)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

    def estimated_pose_correction_callback(self, pose_with_covariance_msg: geometry_msgs.msg.PoseWithCovarianceStamped):
        orientation = pose_with_covariance_msg.pose.pose.orientation
        theta, _, _ = pyquaternion.Quaternion(x=orientation.x, y=orientation.y, z=orientation.z, w=orientation.w).yaw_pitch_roll
        covariance_mat = np.array(pose_with_covariance_msg.pose.covariance).reshape(6, 6)

        self.estimated_correction_poses_df = self.estimated_correction_poses_df.append({
            't': nanoseconds_to_seconds(Time.from_msg(pose_with_covariance_msg.header.stamp).nanoseconds),
            'x': pose_with_covariance_msg.pose.pose.position.x,
            'y': pose_with_covariance_msg.pose.pose.position.y,
            'theta': theta,
            'cov_x_x': covariance_mat[0, 0],
            'cov_x_y': covariance_mat[0, 1],
            'cov_y_y': covariance_mat[1, 1],
            'cov_theta_theta': covariance_mat[5, 5]
        }, ignore_index=True)

    def ground_truth_pose_callback(self, odometry_msg: nav_msgs.msg.Odometry):
        orientation = odometry_msg.pose.pose.orientation
        theta, _, _ = pyquaternion.Quaternion(x=orientation.x, y=orientation.y, z=orientation.z, w=orientation.w).yaw_pitch_roll

        self.ground_truth_poses_df = self.ground_truth_poses_df.append({
            't': nanoseconds_to_seconds(Time.from_msg(odometry_msg.header.stamp).nanoseconds),
            'x': odometry_msg.pose.pose.position.x,
            'y': odometry_msg.pose.pose.position.y,
            'theta': theta,
            'v_x': odometry_msg.twist.twist.linear.x,
            'v_y': odometry_msg.twist.twist.linear.y,
            'v_theta': odometry_msg.twist.twist.angular.z,
        }, ignore_index=True)

    def ps_snapshot_timer_callback(self):
        ps_snapshot_file_path = path.join(self.ps_output_folder, "ps_{i:08d}.pkl".format(i=self.ps_snapshot_count))

        processes_dicts_list = list()
        for process in self.ps_processes:
            try:
                process_copy = copy.deepcopy(process.as_dict())  # get all information about the process
            except psutil.NoSuchProcess:  # processes may have died, causing this exception to be raised from psutil.Process.as_dict
                continue
            try:
                # delete uninteresting values
                del process_copy['connections']
                del process_copy['memory_maps']
                del process_copy['environ']

                processes_dicts_list.append(process_copy)
            except KeyError:
                pass
        try:
            with open(ps_snapshot_file_path, 'wb') as ps_snapshot_file:
                pickle.dump(processes_dicts_list, ps_snapshot_file)
        except TypeError:
            print_error(traceback.format_exc())

        self.ps_snapshot_count += 1

    def init_run_events_file(self):
        backup_file_if_exists(self.run_events_file_path)
        try:
            with open(self.run_events_file_path, 'w') as run_events_file:
                run_events_file.write("{t}, {event}\n".format(t='timestamp', event='event'))
        except IOError as e:
            self.get_logger().error("slam_benchmark_supervisor.init_event_file: could not write header to run_events_file")
            self.get_logger().error(e)

    def write_event(self, stamp, event):
        print_info("slam_benchmark_supervisor: t: {t}, event: {event}".format(t=nanoseconds_to_seconds(stamp.nanoseconds), event=str(event)))
        try:
            with open(self.run_events_file_path, 'a') as run_events_file:
                run_events_file.write("{t}, {event}\n".format(t=nanoseconds_to_seconds(stamp.nanoseconds), event=str(event)))
        except IOError as e:
            self.get_logger().error("slam_benchmark_supervisor.write_event: could not write event to run_events_file: {t} {event}".format(t=nanoseconds_to_seconds(stamp.nanoseconds), event=str(event)))
            self.get_logger().error(e)
