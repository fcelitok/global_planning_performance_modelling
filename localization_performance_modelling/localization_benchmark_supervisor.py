#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import traceback

import rclpy
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from geometry_msgs.msg import Twist
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

    try:
        node = LocalizationBenchmarkSupervisor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


class LocalizationBenchmarkSupervisor(Node):
    def __init__(self):
        super().__init__('localization_benchmark_supervisor', automatically_declare_parameters_from_overrides=True)

        # general parameters
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        scan_topic = self.get_parameter('scan_topic').value

        # run parameters
        self.run_timeout = self.get_parameter('run_timeout').value
        self.ps_snapshot_period = self.get_parameter('ps_snapshot_period').value
        self.run_output_folder = self.get_parameter('run_output_folder').value
        self.ps_pid_father = self.get_parameter('pid_father').value
        self.benchmark_data_folder = path.join(self.run_output_folder, "benchmark_data")
        self.ps_output_folder = path.join(self.benchmark_data_folder, "ps_snapshots")

        # run variables
        self.terminate = False
        self.initial_ground_truth_pose = None
        self.map_snapshot_count = 0
        self.ps_snapshot_count = 0
        self.last_map_msg = None
        self.ps_processes = psutil.Process(self.ps_pid_father).children(recursive=True)  # list of processes children of the benchmark script, i.e., all ros nodes of the benchmark including this one

        # prepare folder structure
        if not path.exists(self.benchmark_data_folder):
            os.makedirs(self.benchmark_data_folder)

        if not path.exists(self.ps_output_folder):
            os.makedirs(self.ps_output_folder)

        # file paths for benchmark data
        self.localization_correction_poses_file_path = path.join(self.benchmark_data_folder, "localization_correction_poses")

        self.ground_truth_poses_file_path = path.join(self.benchmark_data_folder, "ground_truth_poses")
        backup_file_if_exists(self.ground_truth_poses_file_path)

        self.cmd_vel_twists_file_path = path.join(self.benchmark_data_folder, "cmd_vel_twists")
        backup_file_if_exists(self.cmd_vel_twists_file_path)

        self.scans_file_path = path.join(self.benchmark_data_folder, "scans")
        backup_file_if_exists(self.scans_file_path)

        self.run_events_file_path = path.join(self.benchmark_data_folder, "run_events.csv")
        self.init_run_events_file()

        # setup timers, buffers and subscribers
        self.create_timer(self.run_timeout, self.run_timeout_callback)
        self.create_timer(self.ps_snapshot_period, self.ps_snapshot_timer_callback)
        self.cmd_vel_twist_subscriber = self.create_subscription(Twist, cmd_vel_topic, self.cmd_vel_callback, 1)
        self.scan_subscriber = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 1)

        self.write_event(self.get_clock().now(), 'run_start')

    def shutdown(self):
        if not self.terminate:
            print_info("slam_benchmark_supervisor: asked to shutdown, terminating run")
            self.write_event(self.get_clock().now(), 'ros_shutdown')
            self.terminate = True

        self.write_event(self.get_clock().now(), 'supervisor_finished')

        sys.exit(0)

    def run_timeout_callback(self):
        print_info("slam_benchmark_supervisor: terminating supervisor due to timeout, terminating run")
        self.write_event(self.get_clock().now(), 'run_timeout')
        self.terminate = True
        self.shutdown()

    def cmd_vel_callback(self, twist_msg):
        with open(self.cmd_vel_twists_file_path, 'a') as cmd_vel_twists_file:
            cmd_vel_twists_file.write("{t}, {v_x}, {v_y}, {v_theta}\n".format(t=nanoseconds_to_seconds(self.get_clock().now().nanoseconds),
                                                                              v_x=twist_msg.linear.x,
                                                                              v_y=twist_msg.linear.y,
                                                                              v_theta=twist_msg.angular.z))

    def map_callback(self, occupancy_grid_msg):
        self.last_map_msg = occupancy_grid_msg

    def scan_callback(self, laser_scan_msg):
        with open(self.scans_file_path, 'a') as scans_file:
            scans_file.write("{t}, {angle_min}, {angle_max}, {angle_increment}, {range_min}, {range_max}, {ranges}\n".format(
                t=nanoseconds_to_seconds(laser_scan_msg.header.stamp.nanoseconds),
                angle_min=laser_scan_msg.angle_min,
                angle_max=laser_scan_msg.angle_max,
                angle_increment=laser_scan_msg.angle_increment,
                range_min=laser_scan_msg.range_min,
                range_max=laser_scan_msg.range_max,
                ranges=', '.join(map(str, laser_scan_msg.ranges))))

    def ps_snapshot_timer_callback(self):
        ps_snapshot_file_path = path.join(self.ps_output_folder, "ps_{i:08d}.pkl".format(i=self.ps_snapshot_count))

        processes_dicts_list = list()
        print("self.ps_processes", self.ps_processes)
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
        except TypeError as e:
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
