#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import argparse
import pickle
import sys
import yaml
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import traceback


from performance_modelling_py.utils import print_info, print_error


def get_simple_value(result_path):
    with open(result_path) as result_file:
        return result_file.read()


def get_csv(result_path):
    df_csv = pd.read_csv(result_path, sep=', ', engine='python')
    return df_csv


def get_yaml(yaml_file_path):
    with open(yaml_file_path) as yaml_file:
        return yaml.load(yaml_file)


def collect_data(base_run_folder_path, invalidate_cache=False):

    base_run_folder = path.expanduser(base_run_folder_path)
    cache_file_path = path.join(base_run_folder, "run_data_cache.pkl")

    if not path.isdir(base_run_folder):
        print_error("collect_data: base_run_folder does not exists or is not a directory".format(base_run_folder))
        return None, None

    run_folders = sorted(list(filter(path.isdir, glob.glob(path.abspath(base_run_folder) + '/*'))))

    if invalidate_cache or not path.exists(cache_file_path):
        df = pd.DataFrame()
        parameter_names = set()
        cached_run_folders = set()
    else:
        print_info("collect_data: updating cache")
        with open(cache_file_path, 'rb') as f:
            cache = pickle.load(f)
        df = cache['df']
        parameter_names = cache['parameter_names']
        cached_run_folders = set(df['run_folder'])

    # collect results from runs not already cached
    print_info("collect_data: reading run data")
    no_output = True
    for i, run_folder in enumerate(run_folders):
        try:
            metric_results_folder = path.join(run_folder, "metric_results")
            benchmark_data_folder = path.join(run_folder, "benchmark_data")
            run_info_file_path = path.join(run_folder, "run_info.yaml")

            if not path.exists(metric_results_folder):
                print_error("collect_data: metric_results_folder does not exists [{}]".format(metric_results_folder))
                no_output = False
                continue
            if not path.exists(run_info_file_path):
                print_error("collect_data: run_info file does not exists [{}]".format(run_info_file_path))
                no_output = False
                continue
            if run_folder in cached_run_folders:
                continue

            run_info = get_yaml(run_info_file_path)

            run_record = dict()

            for parameter_name, parameter_value in run_info['run_parameters'].items():
                parameter_names.add(parameter_name)
                if type(parameter_value) == list:
                    parameter_value = tuple(parameter_value)
                run_record[parameter_name] = parameter_value

            # parameter_names.add('environment_name')   # in my run_info.yaml file inside run_parameters we have environment_name
            # run_record['environment_name'] = path.basename(path.abspath(run_info['environment_folder']))
            run_record['environment_name'] = run_info['run_parameters']['environment_name']
            run_record['run_folder'] = run_folder
            run_record['failure_rate'] = 0

            try:
                run_events = pd.read_csv(path.join(benchmark_data_folder, "run_events.csv"))
                metrics_dict = get_yaml(path.join(metric_results_folder, "metrics.yaml"))
            except IOError:
                run_record['failure_rate'] = 1
                df = df.append(run_record, ignore_index=True)
                continue

            feasibility_rate = metrics_dict['feasibility_rate'] if 'feasibility_rate' in metrics_dict else None
            run_record['feasibility_rate'] = feasibility_rate
            if feasibility_rate is None:
                run_record['failure_rate'] = 1
                df = df.append(run_record, ignore_index=True)
                continue

            if 'average_planning_time' in metrics_dict and metrics_dict['average_planning_time'] is not None:
                run_record['average_planning_time'] = metrics_dict['average_planning_time']

            if 'normalised_plan_length' in metrics_dict and metrics_dict['normalised_plan_length'] is not None:
                run_record['normalised_plan_length'] = metrics_dict['normalised_plan_length']

            if 'normalised_planning_time' in metrics_dict and metrics_dict['normalised_planning_time'] is not None:
                run_record['normalised_planning_time'] = metrics_dict['normalised_planning_time']

            run_record['num_target_pose_set'] = len(run_events[run_events["event"] == "target_pose_set"])
            run_record['num_target_pose_reached'] = len(run_events[run_events["event"] == "target_pose_reached"])
            run_record['num_target_pose_aborted'] = len(run_events[run_events["event"] == "target_pose_aborted"])
            run_record['num_initial_pose_true'] = len(run_events[run_events["event"] == "initial_pose_true"])
            run_record['num_away_from_initial_pose'] = len(run_events[run_events["event"] == "away_from_initial_pose"])
            run_record['num_goal_pose_true'] = len(run_events[run_events["event"] == "goal_pose_true"])
            run_record['num_away_from_goal_pose'] = len(run_events[run_events["event"] == "away_from_goal_pose"])

            run_start_events = run_events["event"] == "run_start"
            run_completed_events = run_events["event"] == "run_completed"
            if len(run_start_events) == 0 or len(run_completed_events) == 0:
                run_record['failure_rate'] = 1
                df = df.append(run_record, ignore_index=True)
                continue

            if len(run_events[run_events["event"] == "run_start"]["time"]) == 0:
                print_error("collect_data: run_start event does not exists")
                no_output = False
                run_record['failure_rate'] = 1
                df = df.append(run_record, ignore_index=True)
                continue

            if len(run_events[run_events["event"] == "run_completed"]["time"]) == 0:
                print_error("collect_data: run_completed event does not exists")
                no_output = False
                run_record['failure_rate'] = 1
                df = df.append(run_record, ignore_index=True)
                continue

            run_start_time = float(run_events[run_events["event"] == "run_start"]["time"])
            supervisor_finish_time = float(run_events[run_events["event"] == "run_completed"]["time"])
            run_execution_time = supervisor_finish_time - run_start_time
            run_record['run_execution_time'] = run_execution_time

            # if metrics_dict['cpu_and_memory_usage'] is not None and 'amcl_accumulated_cpu_time' in metrics_dict['cpu_and_memory_usage']:
            #     run_record['normalised_cpu_time'] = metrics_dict['cpu_and_memory_usage']['amcl_accumulated_cpu_time'] / run_execution_time
            #
            # if metrics_dict['cpu_and_memory_usage'] is not None and 'amcl_uss' in metrics_dict['cpu_and_memory_usage']:
            #     run_record['max_memory'] = metrics_dict['cpu_and_memory_usage']['amcl_uss']
            #
            # if metrics_dict['cpu_and_memory_usage'] is not None and 'localization_slam_toolbox_node_accumulated_cpu_time' in metrics_dict['cpu_and_memory_usage']:
            #     run_record['normalised_cpu_time'] = metrics_dict['cpu_and_memory_usage']['localization_slam_toolbox_node_accumulated_cpu_time'] / run_execution_time
            #
            # if metrics_dict['cpu_and_memory_usage'] is not None and 'localization_slam_toolbox_node_uss' in metrics_dict['cpu_and_memory_usage']:
            #     run_record['max_memory'] = metrics_dict['cpu_and_memory_usage']['localization_slam_toolbox_node_uss']

            df = df.append(run_record, ignore_index=True)

            print_info("collect_data: reading run data: {}%".format(int((i + 1)*100/len(run_folders))), replace_previous_line=no_output)
            no_output = True
        except:
            traceback.format_exc()

    # save cache
    if cache_file_path is not None:
        cache = {'df': df, 'parameter_names': parameter_names}
        with open(cache_file_path, 'wb') as f:
            pickle.dump(cache, f, protocol=2)

    return df, parameter_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Execute the analysis of the benchmark results.')

    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder containing the result the runs. Defaults to ~/ds/performance_modelling/output/test_global_planning/',
                        type=str,
                        default="~/ds/performance_modelling/output/test_global_planning",
                        required=False)

    parser.add_argument('-i', dest='invalidate_cache',
                        help='If set, all the data is re-read.',
                        action='store_true',
                        default=False,
                        required=False)

    args = parser.parse_args()

    run_data_df, params = collect_data(args.base_run_folder, args.invalidate_cache)
    run_data_df = pd.DataFrame(run_data_df)
    print(run_data_df)
    print(" ")
    print (params)
    # mean_of_ARAPlanner_normalised_planning_time = run_data_df.where(run_data_df['planner_type'] == 'ARAPlanner')['normalised_planning_time'].mean()
    # mean_of_ADPlanner_normalised_planning_time = run_data_df.where(run_data_df['planner_type'] == 'ADPlanner')['normalised_planning_time'].mean()
    # mean_of_ADPlanner_normalised_plan_length = run_data_df.where(run_data_df['planner_type'] == 'ADPlanner')['normalised_plan_length'].mean()
    # mean_of_ADPlanner_normalised_plan_length = run_data_df.where(run_data_df['planner_type'] == 'ADPlanner')['normalised_plan_length'].mean()
    # print(mean_of_ARAPlanner_normalised_planning_time)
    # print("")
    # print(mean_of_ADPlanner_normalised_planning_time)
    # plt.bar([1, 2, 3, 4], [mean_of_ARAPlanner_normalised_planning_time, mean_of_ADPlanner_normalised_planning_time, mean_of_ADPlanner_normalised_plan_length, mean_of_ADPlanner_normalised_plan_length])
    # plt.show()
    # print(run_data_df.groupby(by=list(params))['run_folder'].count())
