#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import shutil
import traceback

import rospy
import yaml
import time
from os import path

from performance_modelling_py.utils import backup_file_if_exists, print_info, print_error
from performance_modelling_py.component_proxies.ros1_component import Component
from global_planning_performance_modelling_ros.metrics import compute_metrics


class BenchmarkRun(object):
    def __init__(self, run_id, run_output_folder, benchmark_log_path, environment_folder, parameters_combination_dict, benchmark_configuration_dict, show_ros_info, headless):
        # print("PRINT:\n")
        # print(run_id,'\n')                        #exmaple: 0
        # print(run_output_folder,'\n')             #exmaple: /home/furkan/ds/performance_modelling/output/test_planning/session_2020-09-30_17-09-01_964405_run_000000000
        # print(benchmark_log_path,'\n')            #exmaple: /home/furkan/ds/performance_modelling/output/test_planning/benchmark_log.csv
        # print(environment_folder,'\n')            #exmaple: /home/furkan/ds/performance_modelling/test_datasets/dataset/airlab
        # print(parameters_combination_dict,'\n')   #exmaple: {'use_dijkstra': True, 'environment_name': 'airlab', 'global_planner_name': 'GlobalPlanner'}
        # print(benchmark_configuration_dict,'\n')  #exmaple: {'components_configurations_folder': '~/turtlebot3_melodic_ws/src/global_planning_performance_modelling/config/component_configurations',
        #                                           #          'supervisor_component': 'global_planning_benchmark_supervisor', 
        #                                           #          'components_configuration': {'move_base': {'GlobalPlanner': 'move_base/globalPlanner.yaml'}, 
        #                                           #                                       'supervisor': 'global_planning_benchmark_supervisor/global_planning_benchmark_supervisor.yaml', 
        #                                           #                                       'rviz': 'rviz/default_view.rviz'}, 
        #                                           #          'combinatorial_parameters': {'use_dijkstra': [True, False], 'environment_name': ['airlab'], 'global_planner_name': ['GlobalPlanner']}} 
        # print(show_ros_info,'\n')                 #exmaple: False
        # print(headless,'\n')                      #exmaple: False 

        # run configuration
        self.run_id = run_id
        self.run_output_folder = run_output_folder
        self.benchmark_log_path = benchmark_log_path
        self.run_parameters = parameters_combination_dict
        self.benchmark_configuration = benchmark_configuration_dict
        self.components_ros_output = 'screen' if show_ros_info else 'log'
        self.headless = headless

        # environment parameters
        self.environment_folder = environment_folder
        self.map_info_file_path = path.join(environment_folder, "data", "map.yaml")

        # take run parameters from parameters_combination_dictionary
        global_planner_name = self.run_parameters['global_planner_name']

        if global_planner_name == 'GlobalPlanner':
            use_dijkstra = self.run_parameters['use_dijkstra']
            lethal_cost = self.run_parameters['lethal_cost']
        elif global_planner_name == 'SBPLLatticePlanner':
            sbpl_primitives_directory_path = path.expanduser(self.benchmark_configuration['sbpl_primitives_path'])
            sbpl_primitives_name = self.run_parameters['sbpl_primitives_name']
            sbpl_primitives_file_path = path.join(sbpl_primitives_directory_path, sbpl_primitives_name)
            planner_type = self.run_parameters['planner_type']
        elif global_planner_name == 'OmplGlobalPlanner':
            planner_type = self.run_parameters['planner_type']
            lethal_cost = self.run_parameters['lethal_cost']
            # TODO add some parameters for OMPL
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

        # benchmark_configuration_parameters
        self.run_timeout = self.benchmark_configuration['run_timeout']

        # components original configuration paths (inside your workspace path)
        components_configurations_folder = path.expanduser(self.benchmark_configuration['components_configurations_folder'])
        original_supervisor_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['supervisor'])
        original_move_base_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['move_base'])
        if global_planner_name == 'GlobalPlanner':
            original_move_base_global_planner_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['move_base_global_planner'])
        elif global_planner_name == 'SBPLLatticePlanner':
            original_move_base_global_planner_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration[ 'components_configuration']['move_base_sbpl_planner'])
        elif global_planner_name == 'OmplGlobalPlanner':
            original_move_base_global_planner_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['move_base_ompl_planner'])
        else:
            raise ValueError()
        self.original_rviz_configuration_path = path.join(components_configurations_folder, self.benchmark_configuration['components_configuration']['rviz'])
        original_robot_urdf_path = path.join(environment_folder, "gazebo", "robot.urdf")

        # components configuration relative paths
        supervisor_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['supervisor'])
        move_base_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['move_base'])
        if global_planner_name == 'GlobalPlanner':
            move_base_global_planner_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['move_base_global_planner'])
        elif global_planner_name == 'SBPLLatticePlanner':
            move_base_global_planner_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration[ 'components_configuration']['move_base_sbpl_planner'])
        elif global_planner_name == 'OmplGlobalPlanner':
            move_base_global_planner_configuration_relative_path = path.join("components_configuration", self.benchmark_configuration['components_configuration']['move_base_ompl_planner'])
        else:
            raise ValueError()
        robot_realistic_urdf_relative_path = path.join("components_configuration", "gazebo", "robot_realistic.urdf")

        # components configuration paths in run folder (inside ds output file path)
        self.supervisor_configuration_path = path.join(self.run_output_folder, supervisor_configuration_relative_path)
        self.move_base_configuration_path = path.join(self.run_output_folder, move_base_configuration_relative_path)
        self.move_base_global_planner_configuration_path = path.join(self.run_output_folder, move_base_global_planner_configuration_relative_path)
        self.robot_realistic_urdf_path = path.join(self.run_output_folder, robot_realistic_urdf_relative_path)

        # copy the configuration of the supervisor to the run folder and update its parameters
        with open(original_supervisor_configuration_path) as supervisor_configuration_file:
            supervisor_configuration = yaml.load(supervisor_configuration_file)
        supervisor_configuration['run_output_folder'] = self.run_output_folder
        supervisor_configuration['pid_father'] = os.getpid()
        supervisor_configuration['ground_truth_map_info_path'] = self.map_info_file_path
        supervisor_configuration['run_timeout'] = self.run_timeout
        if not path.exists(path.dirname(self.supervisor_configuration_path)):
            os.makedirs(path.dirname(self.supervisor_configuration_path))
        with open(self.supervisor_configuration_path, 'w') as supervisor_configuration_file:
            yaml.dump(supervisor_configuration, supervisor_configuration_file, default_flow_style=False)

        # copy the configuration of move_base to the run folder
        # move_base global planner config
        with open(original_move_base_global_planner_configuration_path) as move_base_global_planner_configuration_file:
            move_base_global_planner_configuration = yaml.load(move_base_global_planner_configuration_file)
            move_base_global_planner_configuration['planner_patience'] = self.run_timeout
            move_base_global_planner_configuration['controller_patience'] = self.run_timeout

        if global_planner_name == 'GlobalPlanner':
            move_base_global_planner_configuration['GlobalPlanner']['use_dijkstra'] = use_dijkstra
            move_base_global_planner_configuration['GlobalPlanner']['use_grid_path'] = not use_dijkstra
            move_base_global_planner_configuration['GlobalPlanner']['lethal_cost'] = lethal_cost
            # todo we can add neutral_cost and cost_factor
        elif global_planner_name == 'SBPLLatticePlanner':
            move_base_global_planner_configuration['SBPLLatticePlanner']['planner_type'] = planner_type
            move_base_global_planner_configuration['SBPLLatticePlanner']['primitive_filename'] = sbpl_primitives_file_path
            # todo
        elif global_planner_name == 'OmplGlobalPlanner':
            move_base_global_planner_configuration['OmplGlobalPlanner']['planner_type'] = planner_type
            move_base_global_planner_configuration['OmplGlobalPlanner']['lethal_cost'] = lethal_cost
        else:
            raise ValueError()

        if not path.exists(path.dirname(self.move_base_global_planner_configuration_path)):
            os.makedirs(path.dirname(self.move_base_global_planner_configuration_path))
        with open(self.move_base_global_planner_configuration_path, 'w') as move_base_global_planner_configuration_file:
            yaml.dump(move_base_global_planner_configuration, move_base_global_planner_configuration_file, default_flow_style=False)

        # move_base general config (costmaps, local_planner, etc)
        if not path.exists(path.dirname(self.move_base_configuration_path)):
            os.makedirs(path.dirname(self.move_base_configuration_path))
        shutil.copyfile(original_move_base_configuration_path, self.move_base_configuration_path)

        # copy the configuration of the robot urdf to the run folder and update the link names for realistic data
        if not path.exists(path.dirname(self.robot_realistic_urdf_path)):
            os.makedirs(path.dirname(self.robot_realistic_urdf_path))
        shutil.copyfile(original_robot_urdf_path, self.robot_realistic_urdf_path)

        # write run info to file
        run_info_dict = dict()
        run_info_dict["run_id"] = self.run_id
        run_info_dict["run_folder"] = self.run_output_folder
        run_info_dict["environment_folder"] = environment_folder
        run_info_dict["run_parameters"] = self.run_parameters
        run_info_dict["local_components_configuration"] = {
            'supervisor': supervisor_configuration_relative_path,
            'move_base': move_base_configuration_relative_path,
            'move_base_global_planner': move_base_global_planner_configuration_relative_path,
            'robot_realistic_urdf': robot_realistic_urdf_relative_path,
        }

        with open(run_info_file_path, 'w') as run_info_file:
            yaml.dump(run_info_dict, run_info_file, default_flow_style=False)

    def log(self, event):  # /home/furkan/ds/performance_modelling/output/test_planning/benchmark_log.csv -> writing benchmark_log.csv related event

        if not path.exists(self.benchmark_log_path):
            with open(self.benchmark_log_path, 'a') as output_file:
                output_file.write("timestamp, run_id, event\n")

        t = time.time()

        print_info("t: {t}, run: {run_id}, event: {event}".format(t=t, run_id=self.run_id, event=event))
        try:
            with open(self.benchmark_log_path, 'a') as output_file:
                output_file.write("{t}, {run_id}, {event}\n".format(t=t, run_id=self.run_id, event=event))
        except IOError as e:
            print_error("benchmark_log: could not write event to file: {t}, {run_id}, {event}".format(t=t, run_id=self.run_id, event=event))
            print_error(e)

    def execute_run(self):

        # components parameters
        rviz_params = {
            'rviz_config_file': self.original_rviz_configuration_path,
            'headless': self.headless,
            'output': "log"
        }
        state_publisher_param = {
            'robot_realistic_urdf_file': self.robot_realistic_urdf_path,
            'output': "log"
        }
        navigation_params = {
            'params_file': self.move_base_configuration_path,
            'global_planner_params_file': self.move_base_global_planner_configuration_path,
            'map_file': self.map_info_file_path,
            'output': "log"
        }
        supervisor_params = {
            'params_file': self.supervisor_configuration_path,
            'output': "screen"
        }
        recorder_params = {
            'bag_file_path': path.join(self.run_output_folder, "benchmark_data.bag"),
            'output': "log"
        }
        # recorder_params2 = {
        #     'bag_file_path': path.join(self.run_output_folder, "benchmark_data2.bag"),
        #     'topics': "/cmd_vel /initialpose /map /map_metadata /map_updates /move_base/DWAPlannerROS/cost_cloud /move_base/DWAPlannerROS/global_plan \
        #                  /move_base/DWAPlannerROS/local_plan /move_base/DWAPlannerROS/parameter_descriptions /move_base/DWAPlannerROS/parameter_updates \
        #                  /move_base/DWAPlannerROS/trajectory_cloud /move_base/(.*)/plan /move_base/SBPLLatticePlanner/sbpl_lattice_planner_stats /move_base/cancel /move_base/current_goal \
        #                  /move_base/feedback /move_base/global_costmap/costmap /move_base/global_costmap/costmap_updates /move_base/global_costmap/footprint \
        #                  /move_base/global_costmap/inflation_layer/parameter_descriptions /move_base/global_costmap/inflation_layer/parameter_updates \
        #                  /move_base/global_costmap/parameter_descriptions /move_base/global_costmap/parameter_updates /move_base/global_costmap/static_layer/parameter_descriptions \
        #                  /move_base/global_costmap/static_layer/parameter_updates /move_base/goal /move_base/local_costmap/costmap /move_base/local_costmap/costmap_updates \
        #                  /move_base/local_costmap/footprint /move_base/local_costmap/inflation_layer/parameter_descriptions /move_base/local_costmap/inflation_layer/parameter_updates \
        #                  /move_base/local_costmap/parameter_descriptions /move_base/local_costmap/parameter_updates /move_base/local_costmap/static_layer/parameter_descriptions \
        #                  /move_base/local_costmap/static_layer/parameter_updates /move_base/parameter_descriptions /move_base/parameter_updates /move_base/result /move_base/status \
        #                  /move_base_simple/goal /odom /rosout /rosout_agg /tf /tf_static",
        #     'output': "log"
        # }


        # declare components
        roscore = Component('roscore', 'global_planning_performance_modelling', 'roscore.launch')
        state_publisher = Component('state_publisher', 'global_planning_performance_modelling', 'robot_state_launcher.launch', state_publisher_param)
        rviz = Component('rviz', 'global_planning_performance_modelling', 'rviz.launch', rviz_params)
        navigation = Component('move_base', 'global_planning_performance_modelling', 'move_base.launch', navigation_params)
        supervisor = Component('supervisor', 'global_planning_performance_modelling', 'global_planning_benchmark_supervisor.launch', supervisor_params)
        recorder = Component('recorder', 'global_planning_performance_modelling', 'rosbag_recorder.launch', recorder_params)
        # recorder2 = Component('recorder', 'global_planning_performance_modelling', 'rosbag_recorder2.launch', recorder_params2)

        # launch roscore and setup a node to monitor ros
        roscore.launch()
        rospy.init_node("benchmark_monitor", anonymous=True)

        # launch components
        rviz.launch()
        navigation.launch()
        supervisor.launch()
        recorder.launch()
        # recorder2.launch()

        # launch components and wait for the supervisor to finish
        self.log(event="waiting_supervisor_finish")
        supervisor.wait_to_finish()
        self.log(event="supervisor_shutdown")

        # check if the rosnode is still ok, otherwise the ros infrastructure has been shutdown and the benchmark is aborted
        if rospy.is_shutdown():
            print_error("execute_run: supervisor finished by ros_shutdown")
            self.aborted = True


        # shut down components
        navigation.shutdown()
        rviz.shutdown()
        roscore.shutdown()
        recorder.shutdown()
        # recorder2.shutdown()
        print_info("execute_run: components shutdown completed")

        # compute all relevant metrics and visualisations
        # try:
        #     self.log(event="start_compute_metrics")
        #     compute_metrics(self.run_output_folder)                      # open here to calculate metric
        # except:
        #     print_error("failed metrics computation")
        #     print_error(traceback.format_exc())

        self.log(event="run_end")
        print_info("run {run_id} completed".format(run_id=self.run_id))
