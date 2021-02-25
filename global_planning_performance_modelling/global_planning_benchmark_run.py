#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import shutil
import traceback

import yaml
from xml.etree import ElementTree as et
import time
from os import path
# import numpy as np

from performance_modelling_py.utils import backup_file_if_exists, print_info, print_error
from performance_modelling_py.component_proxies.ros2_component import Component, ComponentsLauncher
from localization_performance_modelling.metrics import compute_metrics


class BenchmarkRun(object):
    def __init__(self, run_id, run_output_folder, benchmark_log_path, environment_folder, parameters_combination_dict, benchmark_configuration_dict, show_ros_info, headless):

        # run configuration
        self.run_id = run_id
        self.run_output_folder = run_output_folder
        self.benchmark_log_path = benchmark_log_path
        self.run_parameters = parameters_combination_dict
        self.benchmark_configuration = benchmark_configuration_dict
        self.components_ros_output = 'screen' if show_ros_info else 'log'
        self.headless = headless
        self.use_sim_time = True

        # environment parameters
        self.environment_folder = environment_folder
        self.map_info_file_path = path.join(environment_folder, "data", "map.yaml")

        global_planner_name = self.run_parameters['global_planner_name']

        if global_planner_name == 'NavfnPlanner':
            use_astar = self.run_parameters['use_astar']
        else:
            raise ValueError()

        # run variables
        self.aborted = False

        # prepare folder structure
        run_configuration_path = path.join(self.run_output_folder, "components_configuration")
        run_info_file_path = path.join(self.run_output_folder, "run_info.yaml")
        backup_file_if_exists(self.run_output_folder)
        os.mkdir(self.run_output_folder)
        os.mkdir(run_configuration_path)

        # components original configuration paths
        components_configurations_folder = path.expanduser(self.benchmark_configuration['components_configurations_folder'])
        original_supervisor_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['supervisor'])
        original_nav2_navigation_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['nav2_navigation'])
        self.original_rviz_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['rviz'])
        original_robot_urdf_path = path.join(environment_folder, "gazebo", "robot.urdf")

        # components configuration relative paths
        supervisor_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['supervisor'])
        nav2_navigation_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['nav2_navigation'])
        robot_realistic_urdf_relative_path = path.join("components_configuration", "gazebo", "robot_realistic.urdf")

        # components configuration paths in run folder
        self.supervisor_configuration_path = path.join(self.run_output_folder, supervisor_configuration_relative_path)
        self.nav2_navigation_configuration_path = path.join(self.run_output_folder, nav2_navigation_configuration_relative_path)
        self.robot_realistic_urdf_path = path.join(self.run_output_folder, robot_realistic_urdf_relative_path)

        # copy the configuration of the supervisor to the run folder and update its parameters
        with open(original_supervisor_configuration_path) as supervisor_configuration_file:
            supervisor_configuration = yaml.load(supervisor_configuration_file)
        supervisor_configuration['global_planning_benchmark_supervisor']['ros__parameters']['run_output_folder'] = self.run_output_folder
        supervisor_configuration['global_planning_benchmark_supervisor']['ros__parameters']['pid_father'] = os.getpid()
        supervisor_configuration['global_planning_benchmark_supervisor']['ros__parameters']['use_sim_time'] = self.use_sim_time
        supervisor_configuration['global_planning_benchmark_supervisor']['ros__parameters']['ground_truth_map_info_path'] = self.map_info_file_path
        if not path.exists(path.dirname(self.supervisor_configuration_path)):
            os.makedirs(path.dirname(self.supervisor_configuration_path))
        with open(self.supervisor_configuration_path, 'w') as supervisor_configuration_file:
            yaml.dump(supervisor_configuration, supervisor_configuration_file, default_flow_style=False)

        # copy the configuration of nav2_navigation to the run folder
        with open(original_nav2_navigation_configuration_path) as navigation_configuration_file:
            navigation_configuration = yaml.load(navigation_configuration_file)
        navigation_configuration['planner_server']['ros__parameters']['use_astar'] = use_astar
        if not path.exists(path.dirname(self.nav2_navigation_configuration_path)):
            os.makedirs(path.dirname(self.nav2_navigation_configuration_path))
        with open(self.nav2_navigation_configuration_path, 'w') as navigation_configuration_file:
            yaml.dump(navigation_configuration, navigation_configuration_file, default_flow_style=False)
        #shutil.copyfile(original_nav2_navigation_configuration_path, self.nav2_navigation_configuration_path)

        # copy the configuration of the robot urdf to the run folder and update the link names for realistic data
        robot_realistic_urdf_tree = et.parse(original_robot_urdf_path)
        robot_realistic_urdf_root = robot_realistic_urdf_tree.getroot()
        for link_element in robot_realistic_urdf_root.findall(".//link"):
            link_element.attrib['name'] = "{}_realistic".format(link_element.attrib['name'])
        for joint_link_element in robot_realistic_urdf_root.findall(".//*[@link]"):
            joint_link_element.attrib['link'] = "{}_realistic".format(joint_link_element.attrib['link'])
        if not path.exists(path.dirname(self.robot_realistic_urdf_path)):
            os.makedirs(path.dirname(self.robot_realistic_urdf_path))
        robot_realistic_urdf_tree.write(self.robot_realistic_urdf_path)

        # write run info to file
        run_info_dict = dict()
        run_info_dict["run_id"] = self.run_id
        run_info_dict["run_folder"] = self.run_output_folder
        run_info_dict["environment_folder"] = environment_folder
        run_info_dict["run_parameters"] = self.run_parameters
        run_info_dict["local_components_configuration"] = {
            'supervisor': supervisor_configuration_relative_path,
            'nav2_navigation': nav2_navigation_configuration_relative_path,
            'robot_realistic_urdf': robot_realistic_urdf_relative_path,
        }
        # run_info_dict["local_components_configuration"] = {
        #     'supervisor': supervisor_configuration_relative_path,
        #     'nav2_amcl': nav2_amcl_configuration_relative_path,
        #     'nav2_navigation': nav2_navigation_configuration_relative_path,
        #     'gazebo_world_model': gazebo_world_model_relative_path,
        #     'gazebo_robot_model_sdf': gazebo_robot_model_sdf_relative_path,
        #     'gazebo_robot_model_config': gazebo_robot_model_config_relative_path,
        #     'robot_gt_urdf': robot_gt_urdf_relative_path,
        #     'robot_realistic_urdf': robot_realistic_urdf_relative_path,
        # }

        with open(run_info_file_path, 'w') as run_info_file:
            yaml.dump(run_info_dict, run_info_file, default_flow_style=False)

    def log(self, event):

        if not path.exists(self.benchmark_log_path):
            with open(self.benchmark_log_path, 'a') as output_file:
                output_file.write("timestamp, run_id, event\n")

        t = time.time()

        print_info(f"t: {t}, run: {self.run_id}, event: {event}")
        try:
            with open(self.benchmark_log_path, 'a') as output_file:
                output_file.write(f"{t}, {self.run_id}, {event}\n")
        except IOError as e:
            print_error(f"benchmark_log: could not write event to file: {t}, {self.run_id}, {event}")
            print_error(e)

    def execute_run(self):

        # components parameters

        # environment_params = {
        #     'gazebo_model_path_env_var': self.gazebo_model_path_env_var,
        #     'gazebo_plugin_path_env_var': self.gazebo_plugin_path_env_var,
        #     'world_model_file': self.gazebo_world_model_path,
        #     'robot_gt_urdf_file': self.robot_gt_urdf_path,
        #     'robot_realistic_urdf_file': self.robot_realistic_urdf_path,
        #     'headless': self.headless,
        # }
        # localization_params = {
        #     'params_file': self.nav2_amcl_configuration_path,
        #     'map': self.map_info_file_path,
        #     'use_sim_time': self.use_sim_time,
        # }
        navigation_params = {
            'params_file': self.nav2_navigation_configuration_path,
            'use_sim_time': self.use_sim_time,
            'map': self.map_info_file_path,
            'map_subscribe_transient_local': True,
        }
        supervisor_params = {
            'configuration': self.supervisor_configuration_path,
            'use_sim_time': self.use_sim_time
        }
        rviz_params = {
            'rviz_config_file': self.original_rviz_configuration_path,
        }

        # declare components
        rviz = Component('rviz', 'global_planning_performance_modelling', 'rviz.launch.py', rviz_params)
        navigation = Component('nav2_navigation', 'global_planning_performance_modelling', 'nav2_navigation.launch.py', navigation_params)
        supervisor = Component('supervisor', 'global_planning_performance_modelling', 'global_planning_benchmark_supervisor.launch.py', supervisor_params)

        # recorder = Component('recorder', 'localization_performance_modelling', 'rosbag_recorder.launch.py', recorder_params)
        # environment = Component('gazebo', 'localization_performance_modelling', 'gazebo.launch.py', environment_params)
        # localization = Component('nav2_amcl', 'localization_performance_modelling', 'nav2_amcl.launch.py', localization_params)

        # TODO manage launch exceptions in Component.__init__

        # add components to launcher
        components_launcher = ComponentsLauncher()

        if not self.headless:
            components_launcher.add_component(rviz)
        components_launcher.add_component(navigation)
        components_launcher.add_component(supervisor)

        # recorder.launch()
        # components_launcher.add_component(environment)
        # components_launcher.add_component(localization)


        # launch components and wait for the supervisor to finish
        self.log(event="waiting_supervisor_finish")
        components_launcher.launch()
        self.log(event="supervisor_shutdown")

        # make sure remaining components have shutdown
        components_launcher.shutdown()
        print_info("execute_run: components shutdown completed")

        # TODO: Dont forget metrics computation to open
        # compute all relevant metrics and visualisations
        # noinspection PyBroadException
        # try:
        #     self.log(event="start_compute_metrics")
        #     compute_metrics(self.run_output_folder)
        # except:
        #     print_error("failed metrics computation")
        #     print_error(traceback.format_exc())

        self.log(event="run_end")
        print_info(f"run {self.run_id} completed")
