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
import numpy as np

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
        self.robot_gt_urdf_file = path.join(environment_folder, "gazebo", "robot_gt.urdf")
        self.robot_realistic_urdf_file = path.join(environment_folder, "gazebo", "robot_realistic.urdf")
        self.gazebo_model_path_env_var = ":".join(map(
            lambda p: path.expanduser(p),
            self.benchmark_configuration['gazebo_model_path_env_var'] + [self.run_output_folder]
        ))
        self.gazebo_plugin_path_env_var = ":".join(map(
            lambda p: path.expanduser(p),
            self.benchmark_configuration['gazebo_plugin_path_env_var']
        ))
        beta_1, beta_2, beta_3, beta_4 = self.run_parameters['beta']
        laser_scan_max_range = self.run_parameters['laser_scan_max_range']
        laser_scan_fov_deg = self.run_parameters['laser_scan_fov_deg']
        laser_scan_fov_rad = laser_scan_fov_deg*np.pi/180

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
        supervisor_original_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['supervisor'])
        nav2_amcl_original_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['nav2_amcl'])
        nav2_navigation_original_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['nav2_navigation'])
        gazebo_world_model_original_path = path.join(environment_folder, "gazebo", "gazebo_environment.model")
        gazebo_robot_model_config_original_path = path.join(environment_folder, "gazebo", "robot", "model.config")
        gazebo_robot_model_sdf_original_path = path.join(environment_folder, "gazebo", "robot", "model.sdf")
        self.rviz_original_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['rviz'])

        # components configuration relative paths
        supervisor_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['supervisor'])
        nav2_amcl_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['nav2_amcl'])
        nav2_navigation_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['nav2_navigation'])
        gazebo_world_model_relative_path = path.join("components_configuration", "gazebo", "gazebo_environment.model")
        gazebo_robot_model_config_relative_path = path.join("components_configuration", "gazebo", "robot", "model.config")
        gazebo_robot_model_sdf_relative_path = path.join("components_configuration", "gazebo", "robot", "model.sdf")

        # components configuration paths in run folder
        self.supervisor_configuration_path = path.join(self.run_output_folder, supervisor_configuration_relative_path)
        self.nav2_amcl_configuration_path = path.join(self.run_output_folder, nav2_amcl_configuration_relative_path)
        self.nav2_navigation_configuration_path = path.join(self.run_output_folder, nav2_navigation_configuration_relative_path)
        self.gazebo_world_model_path = path.join(self.run_output_folder, gazebo_world_model_relative_path)
        gazebo_robot_model_config_path = path.join(self.run_output_folder, gazebo_robot_model_config_relative_path)
        gazebo_robot_model_sdf_path = path.join(self.run_output_folder, gazebo_robot_model_sdf_relative_path)

        # copy the configuration of the supervisor to the run folder and update its parameters
        with open(supervisor_original_configuration_path) as supervisor_configuration_file:
            supervisor_configuration = yaml.load(supervisor_configuration_file)
        supervisor_configuration['localization_benchmark_supervisor']['ros__parameters']['run_output_folder'] = self.run_output_folder
        supervisor_configuration['localization_benchmark_supervisor']['ros__parameters']['pid_father'] = os.getpid()
        supervisor_configuration['localization_benchmark_supervisor']['ros__parameters']['use_sim_time'] = self.use_sim_time
        supervisor_configuration['localization_benchmark_supervisor']['ros__parameters']['ground_truth_map_info_path'] = self.map_info_file_path
        if not path.exists(path.dirname(self.supervisor_configuration_path)):
            os.makedirs(path.dirname(self.supervisor_configuration_path))
        with open(self.supervisor_configuration_path, 'w') as supervisor_configuration_file:
            yaml.dump(supervisor_configuration, supervisor_configuration_file, default_flow_style=False)

        # copy the configuration of nav2_amcl to the run folder and update its parameters
        with open(nav2_amcl_original_configuration_path) as nav2_amcl_original_configuration_file:
            nav2_amcl_configuration = yaml.load(nav2_amcl_original_configuration_file)
        nav2_amcl_configuration['amcl']['ros__parameters']['alpha1'] = 2 * beta_1**2
        nav2_amcl_configuration['amcl']['ros__parameters']['alpha2'] = 2 * beta_2**2
        nav2_amcl_configuration['amcl']['ros__parameters']['alpha3'] = 2 * beta_3**2
        nav2_amcl_configuration['amcl']['ros__parameters']['alpha4'] = 2 * beta_4**2
        nav2_amcl_configuration['amcl']['ros__parameters']['laser_max_range'] = laser_scan_max_range
        if not path.exists(path.dirname(self.nav2_amcl_configuration_path)):
            os.makedirs(path.dirname(self.nav2_amcl_configuration_path))
        with open(self.nav2_amcl_configuration_path, 'w') as nav2_amcl_configuration_file:
            yaml.dump(nav2_amcl_configuration, nav2_amcl_configuration_file, default_flow_style=False)

        # copy the configuration of nav2_navigation to the run folder
        if not path.exists(path.dirname(self.nav2_navigation_configuration_path)):
            os.makedirs(path.dirname(self.nav2_navigation_configuration_path))
        shutil.copyfile(nav2_navigation_original_configuration_path, self.nav2_navigation_configuration_path)

        # copy the configuration of the gazebo world model to the run folder and update its parameters
        gazebo_original_world_model_tree = et.parse(gazebo_world_model_original_path)
        gazebo_original_world_model_root = gazebo_original_world_model_tree.getroot()
        gazebo_original_world_model_root.findall(".//model[@name='robot']/include/uri")[0].text = path.join("model://", path.dirname(gazebo_robot_model_sdf_relative_path))
        if not path.exists(path.dirname(self.gazebo_world_model_path)):
            os.makedirs(path.dirname(self.gazebo_world_model_path))
        gazebo_original_world_model_tree.write(self.gazebo_world_model_path)

        # copy the configuration of the gazebo robot sdf model to the run folder and update its parameters
        gazebo_robot_model_sdf_tree = et.parse(gazebo_robot_model_sdf_original_path)
        gazebo_robot_model_sdf_root = gazebo_robot_model_sdf_tree.getroot()
        gazebo_robot_model_sdf_root.findall(".//sensor[@name='lidar_sensor']/ray/scan/horizontal/samples")[0].text = str(int(laser_scan_fov_deg))
        gazebo_robot_model_sdf_root.findall(".//sensor[@name='lidar_sensor']/ray/scan/horizontal/min_angle")[0].text = str(float(-laser_scan_fov_rad/2))
        gazebo_robot_model_sdf_root.findall(".//sensor[@name='lidar_sensor']/ray/scan/horizontal/max_angle")[0].text = str(float(+laser_scan_fov_rad/2))
        gazebo_robot_model_sdf_root.findall(".//sensor[@name='lidar_sensor']/ray/range/max")[0].text = str(float(laser_scan_max_range))
        gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/alpha1")[0].text = str(beta_1)
        gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/alpha2")[0].text = str(beta_2)
        gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/alpha3")[0].text = str(beta_3)
        gazebo_robot_model_sdf_root.findall(".//plugin[@name='turtlebot3_diff_drive']/alpha4")[0].text = str(beta_4)
        if not path.exists(path.dirname(gazebo_robot_model_sdf_path)):
            os.makedirs(path.dirname(gazebo_robot_model_sdf_path))
        gazebo_robot_model_sdf_tree.write(gazebo_robot_model_sdf_path)

        # copy the configuration of the gazebo robot model to the run folder
        if not path.exists(path.dirname(gazebo_robot_model_config_path)):
            os.makedirs(path.dirname(gazebo_robot_model_config_path))
        shutil.copyfile(gazebo_robot_model_config_original_path, gazebo_robot_model_config_path)

        # write run info to file
        run_info_dict = dict()
        run_info_dict["run_id"] = self.run_id
        run_info_dict["run_folder"] = self.run_output_folder
        run_info_dict["environment_folder"] = environment_folder
        run_info_dict["run_parameters"] = self.run_parameters
        run_info_dict["local_components_configuration"] = {
            'supervisor': supervisor_configuration_relative_path,
            'nav2_amcl': nav2_amcl_configuration_relative_path,
            'nav2_navigation': nav2_navigation_configuration_relative_path,
            'gazebo_world_model': gazebo_world_model_relative_path,
            'gazebo_robot_model_sdf': gazebo_robot_model_sdf_relative_path,
            'gazebo_robot_model_config': gazebo_robot_model_config_relative_path,
        }

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
        rviz_params = {
            'rviz_config_file': self.rviz_original_configuration_path,
        }
        environment_params = {
            'gazebo_model_path_env_var': self.gazebo_model_path_env_var,
            'gazebo_plugin_path_env_var': self.gazebo_plugin_path_env_var,
            'world_model_file': self.gazebo_world_model_path,
            'robot_gt_urdf_file': self.robot_gt_urdf_file,
            'robot_realistic_urdf_file': self.robot_realistic_urdf_file,
            'headless': self.headless,
        }
        localization_params = {
            'params_file': self.nav2_amcl_configuration_path,
            'map': self.map_info_file_path,
            'use_sim_time': self.use_sim_time,
        }
        navigation_params = {
            'params_file': self.nav2_navigation_configuration_path,
            'use_sim_time': self.use_sim_time,
            'map_subscribe_transient_local': True,
        }
        supervisor_params = {
            'configuration': self.supervisor_configuration_path,
            'use_sim_time': self.use_sim_time
        }

        # declare components
        rviz = Component('rviz', 'localization_performance_modelling', 'rviz.launch.py', rviz_params)
        # recorder = Component('recorder', 'localization_performance_modelling', 'rosbag_recorder.launch.py', recorder_params)
        environment = Component('gazebo', 'localization_performance_modelling', 'gazebo.launch.py', environment_params)
        localization = Component('nav2_amcl', 'localization_performance_modelling', 'nav2_amcl.launch.py', localization_params)
        navigation = Component('nav2_navigation', 'localization_performance_modelling', 'nav2_navigation.launch.py', navigation_params)
        supervisor = Component('supervisor', 'localization_performance_modelling', 'localization_benchmark_supervisor.launch.py', supervisor_params)

        # TODO manage launch exceptions in Component.__init__

        # launch components
        components_launcher = ComponentsLauncher()

        print_info("execute_run: launching components")
        if not self.headless:
            components_launcher.add_component(rviz)
        # recorder.launch()
        components_launcher.add_component(environment)
        components_launcher.add_component(navigation)
        components_launcher.add_component(localization)
        components_launcher.add_component(supervisor)

        # wait for the supervisor component to finish
        self.log(event="waiting_supervisor_finish")
        components_launcher.launch()
        self.log(event="supervisor_shutdown")

        # make sure remaining components have shutdown
        components_launcher.shutdown()

        print_info("execute_run: components shutdown completed")

        # compute all relevant metrics and visualisations
        # noinspection PyBroadException
        try:
            self.log(event="start_compute_metrics")
            compute_metrics(self.run_output_folder)
        except:
            print_error("failed metrics computation")
            print_error(traceback.format_exc())

        self.log(event="run_end")
        print_info(f"run {self.run_id} completed")
