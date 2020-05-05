#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import shutil
import yaml
import time
from os import path

from performance_modelling_py.utils import backup_file_if_exists, print_info, print_error
from performance_modelling_py.component_proxies.ros2_component import Component
# from performance_modelling_py.metrics.localization_metrics import compute_localization_metrics
# from performance_modelling_py.metrics.navigation_metrics import compute_navigation_metrics
# from performance_modelling_py.metrics.map_metrics import compute_map_metrics
# from performance_modelling_py.metrics.computation_metrics import compute_computation_metrics
# from performance_modelling_py.visualisation.trajectory_visualisation import save_trajectories_plot


class BenchmarkRun(object):
    def __init__(self, run_id, run_output_folder, benchmark_log_path, show_ros_info, headless, environment_folder, component_configuration_file_paths, supervisor_configuration_file_path):

        self.benchmark_log_path = benchmark_log_path

        # components configuration parameters
        self.component_configuration_files = component_configuration_file_paths
        self.supervisor_configuration_file = supervisor_configuration_file_path

        # environment parameters
        self.environment_folder = environment_folder
        self.map_info_file_path = path.join(environment_folder, "map.yaml")
        self.world_model_file = path.join(environment_folder, "gazebo_environment.model")
        self.robot_urdf_file = path.join(environment_folder, "robot.urdf")

        # run parameters
        self.run_id = run_id
        self.run_output_folder = run_output_folder
        self.components_ros_output = 'screen' if show_ros_info else 'log'
        self.headless = headless
        self.use_sim_time = True

        # run variables
        self.aborted = False

        # prepare folder structure
        run_configuration_copy_path = path.join(self.run_output_folder, "components_configuration")
        run_info_file_path = path.join(self.run_output_folder, "run_info.yaml")
        backup_file_if_exists(self.run_output_folder)
        os.mkdir(self.run_output_folder)
        os.mkdir(run_configuration_copy_path)

        # copy the configuration of each component to the run folder and add the run parameters
        component_configuration_copy_relative_paths = dict()
        self.component_configuration_copy_absolute_paths = dict()
        for component_name, configuration_path in self.component_configuration_files.items():
            configuration_copy_relative_path = path.join("components_configuration", "{}_{}".format(component_name, path.basename(configuration_path)))
            configuration_copy_absolute_path = path.join(self.run_output_folder, configuration_copy_relative_path)
            component_configuration_copy_relative_paths[component_name] = configuration_copy_relative_path
            self.component_configuration_copy_absolute_paths[component_name] = configuration_copy_absolute_path

            backup_file_if_exists(configuration_copy_absolute_path)
            shutil.copyfile(configuration_path, configuration_copy_absolute_path)

        supervisor_configuration_copy_relative_path = path.join("components_configuration", "{}_{}".format("supervisor", path.basename(self.supervisor_configuration_file)))
        self.supervisor_configuration_copy_absolute_path = path.join(self.run_output_folder, supervisor_configuration_copy_relative_path)
        backup_file_if_exists(self.supervisor_configuration_copy_absolute_path)
        shutil.copyfile(self.supervisor_configuration_file, self.supervisor_configuration_copy_absolute_path)

        # add run parameters to the configuration of the supervisor
        with open(self.supervisor_configuration_copy_absolute_path) as supervisor_configuration_file:
            supervisor_configuration = yaml.load(supervisor_configuration_file)
        supervisor_configuration['localization_benchmark_supervisor']['ros__parameters']['run_output_folder'] = self.run_output_folder
        supervisor_configuration['localization_benchmark_supervisor']['ros__parameters']['pid_father'] = os.getpid()
        supervisor_configuration['localization_benchmark_supervisor']['ros__parameters']['use_sim_time'] = self.use_sim_time
        with open(self.supervisor_configuration_copy_absolute_path, 'w') as supervisor_configuration_file:
            yaml.dump(supervisor_configuration, supervisor_configuration_file, default_flow_style=False)

        # write info about the run to file
        run_info_dict = dict()
        run_info_dict["original_components_configuration"] = component_configuration_file_paths
        run_info_dict["original_supervisor_configuration"] = supervisor_configuration_file_path
        run_info_dict["environment_folder"] = environment_folder
        run_info_dict["run_folder"] = self.run_output_folder
        run_info_dict["run_id"] = self.run_id
        run_info_dict["local_components_configuration"] = component_configuration_copy_relative_paths
        run_info_dict["local_supervisor_configuration"] = supervisor_configuration_copy_relative_path

        with open(run_info_file_path, 'w') as run_info_file:
            yaml.dump(run_info_dict, run_info_file, default_flow_style=False)

    def log(self, event):

        if not path.exists(self.benchmark_log_path):
            with open(self.benchmark_log_path, 'a') as output_file:
                output_file.write("{t}, {run_id}, {event}\n".format(t="timestamp", run_id="run_id", event="event"))

        t = time.time()

        try:
            with open(self.benchmark_log_path, 'a') as output_file:
                output_file.write("{t}, {run_id}, {event}\n".format(t=t, run_id=self.run_id, event=event))
        except IOError as e:
            print_error("benchmark_log: could not write event to file: {t}, {run_id}, {event}".format(t=t, run_id=self.run_id, event=event))
            print_error(e)

    def execute_run(self):

        # components parameters
        # Component.common_parameters = {'headless': self.headless,
        #                                'output': self.components_ros_output}
        # recorder_params = {'bag_file_path': path.join(self.run_output_folder, "odom_tf_ground_truth.bag")}

        rviz_params = {
            'rviz_config_file': self.component_configuration_files['rviz'],
        }
        environment_params = {
            'world_model_file': self.world_model_file,
            'robot_urdf_file': self.robot_urdf_file,
            'headless': True,  # TODO headless
        }
        localization_params = {
            'params_file': self.component_configuration_files['nav2_amcl'],
            'map': self.map_info_file_path,
            'use_sim_time': self.use_sim_time,
        }
        navigation_params = {
            'params_file': self.component_configuration_files['nav2_navigation'],
            'use_sim_time': self.use_sim_time,
            'map_subscribe_transient_local': True,
        }
        supervisor_params = {
            'configuration': self.supervisor_configuration_copy_absolute_path,
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
        print_info("execute_run: launching components")
        rviz.launch()
        # recorder.launch()
        environment.launch()
        navigation.launch()
        localization.launch()

        # wait for the supervisor component to finish
        print_info("execute_run: waiting for supervisor to finish")
        self.log(event="waiting_supervisor_finish")
        supervisor.launch_and_wait_to_finish()
        print_info("execute_run: supervisor has shutdown")
        self.log(event="supervisor_shutdown")

        # shutdown remaining components
        # recorder.shutdown()
        rviz.shutdown()
        navigation.shutdown()
        localization.shutdown()
        environment.shutdown()
        print_info("execute_run: components shutdown completed")

        # # compute all relevant metrics and visualisations
        # self.log(event="start_compute_map_metrics")
        # compute_map_metrics(self.run_output_folder, self.stage_world_folder)
        #
        # self.log(event="start_compute_localization_metrics")
        # compute_localization_metrics(self.run_output_folder)
        #
        # self.log(event="start_compute_navigation_metrics")
        # compute_navigation_metrics(self.run_output_folder)
        #
        # self.log(event="start_compute_computation_metrics")
        # compute_computation_metrics(self.run_output_folder)
        #
        # print_info("execute_run: metrics computation completed")
        #
        # self.log(event="start_save_trajectories_plot")
        # save_trajectories_plot(self.run_output_folder)
        # print_info("execute_run: saved visualisation files")

        self.log(event="run_end")
