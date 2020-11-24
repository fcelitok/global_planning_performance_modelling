#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import os
from os import path
import yaml
import pandas as pd
import numpy as np
from performance_modelling_py.environment.ground_truth_map import GroundTruthMap

from performance_modelling_py.utils import print_info, print_error, backup_file_if_exists
from performance_modelling_py.metrics.localization_metrics import trajectory_length_metric, absolute_localization_error_metrics, absolute_error_vs_voronoi_radius, absolute_error_vs_scan_range, absolute_error_vs_geometric_similarity, \
    relative_localization_error_metrics
from performance_modelling_py.metrics.computation_metrics import cpu_and_memory_usage_metrics
# from performance_modelling_py.visualisation.trajectory_visualisation import save_trajectories_plot


def compute_metrics(run_output_folder):    # run_output_folder: /home/furkan/ds/performance_modelling/output/test_planning/session_2020-09-30_17-09-01_964405_run_000000000

    metrics_result_dict = dict()

    run_info_path = path.join(run_output_folder, "run_info.yaml")
    if not path.exists(run_info_path) or not path.isfile(run_info_path):
        print_error("run info file does not exists")

    with open(run_info_path) as run_info_file:
        run_info = yaml.safe_load(run_info_file)

    environment_folder = run_info['environment_folder']
    ground_truth_map_info_path = path.join(environment_folder, "data", "map.yaml")
    ground_truth_map = GroundTruthMap(ground_truth_map_info_path)
    # laser_scan_max_range = run_info['run_parameters']['laser_scan_max_range']

    # localization metrics
    execution_time_path = path.join(run_output_folder, "benchmark_data", "plan_output", "execution_time.csv")
    voronoi_distance_path = path.join(run_output_folder, "benchmark_data", "plan_output", "voronoi_distance.csv")
    euclidean_distance_path = path.join(run_output_folder, "benchmark_data", "plan_output", "euclidean_distance.csv")
    feasibility_rate_path = path.join(run_output_folder, "benchmark_data", "plan_output", "feasibility_rate.csv")
    # ground_truth_poses_path = path.join(run_output_folder, "benchmark_data", "ground_truth_poses.csv")
    # scans_file_path = path.join(run_output_folder, "benchmark_data", "scans.csv")

    logs_folder_path = path.join(run_output_folder, "logs")

    metrics_result_folder_path = path.join(run_output_folder, "metric_results")
    metrics_result_file_path = path.join(metrics_result_folder_path, "metrics.yaml")

    if path.exists(metrics_result_file_path):
        print_info("metrics file already exists, not overwriting [{}]".format(metrics_result_file_path))
    else:
        print_info("average planning time")
        metrics_result_dict['average_planning_time'] = average_planning_time(execution_time_path)

        print_info("planning time over voronoi distance")
        metrics_result_dict['planning_time_over_voronoi_distance'] = planning_time_for_voronoi(execution_time_path, voronoi_distance_path)

        print_info("average time over voronoi")
        metrics_result_dict['average_planning_time_over_voronoi'] = average_planning_time_over_voronoi(execution_time_path, voronoi_distance_path)

        print_info("euclidean length over voronoi distance")
        metrics_result_dict['euclidean_length_over_voronoi_distance'] = euclidean_length_over_voronoi_distance(euclidean_distance_path, voronoi_distance_path)

        print_info("average euclidean length over voronoi distance")
        metrics_result_dict['average_euclidean_length_over_voronoi_distance'] = average_euclidean_length_over_voronoi_distance(euclidean_distance_path, voronoi_distance_path)

        print_info("feasibility rate of map")
        metrics_result_dict['feasibility_rate'] = feasibility_rate(feasibility_rate_path)

        # print_info("trajectory_length")
        # metrics_result_dict['trajectory_length'] = trajectory_length_metric(ground_truth_poses_path)
        #
        # print_info("relative_localization_correction_error")
        # metrics_result_dict['relative_localization_correction_error'] = relative_localization_error_metrics(path.join(logs_folder_path, "relative_localisation_correction_error"), estimated_correction_poses_path, ground_truth_poses_path)
        #
        # print_info("relative_localization_error")
        # metrics_result_dict['relative_localization_error'] = relative_localization_error_metrics(path.join(logs_folder_path, "relative_localisation_error"), estimated_poses_path, ground_truth_poses_path)
        #
        # print_info("absolute_localization_correction_error")
        # metrics_result_dict['absolute_localization_correction_error'] = absolute_localization_error_metrics(estimated_correction_poses_path, ground_truth_poses_path)
        #
        # print_info("absolute_localization_error")
        # metrics_result_dict['absolute_localization_error'] = absolute_localization_error_metrics(estimated_poses_path, ground_truth_poses_path)
        #
        # computation metrics
        # print_info("cpu_and_memory_usage")
        # ps_snapshots_folder_path = path.join(run_output_folder, "benchmark_data", "ps_snapshots")
        # metrics_result_dict['cpu_and_memory_usage'] = cpu_and_memory_usage_metrics(ps_snapshots_folder_path)

        # write metrics
        if not path.exists(metrics_result_folder_path):
            os.makedirs(metrics_result_folder_path)
        with open(metrics_result_file_path, 'w') as metrics_result_file:
            yaml.dump(metrics_result_dict, metrics_result_file, default_flow_style=False)

    # absolute_error_vs_voronoi_radius_df = absolute_error_vs_voronoi_radius(estimated_poses_path, ground_truth_poses_path, ground_truth_map)
    # backup_file_if_exists(path.join(metrics_result_folder_path, "abs_err_vs_voronoi_radius.csv"))
    # absolute_error_vs_voronoi_radius_df.to_csv(path.join(metrics_result_folder_path, "abs_err_vs_voronoi_radius.csv"))
    #
    # absolute_error_vs_scan_range_df = absolute_error_vs_scan_range(estimated_poses_path, ground_truth_poses_path, scans_file_path)
    # backup_file_if_exists(path.join(metrics_result_folder_path, "absolute_error_vs_scan_range.csv"))
    # absolute_error_vs_scan_range_df.to_csv(path.join(metrics_result_folder_path, "absolute_error_vs_scan_range.csv"))
    #
    # absolute_error_vs_geometric_similarity_df = absolute_error_vs_geometric_similarity(estimated_poses_path, ground_truth_poses_path, ground_truth_map, horizon_length=laser_scan_max_range, max_iterations=5, samples_per_second=1)
    # backup_file_if_exists(path.join(metrics_result_folder_path, "absolute_error_vs_geometric_similarity.csv"))
    # absolute_error_vs_geometric_similarity_df.to_csv(path.join(metrics_result_folder_path, "absolute_error_vs_geometric_similarity.csv"))

    # # visualisation
    # print_info("visualisation")
    # visualisation_output_folder = path.join(run_output_folder, "visualisation")
    # save_trajectories_plot(visualisation_output_folder, estimated_poses_path, estimated_correction_poses_path, ground_truth_poses_path)


def feasibility_rate(feasibility_rate_path):
    with open(feasibility_rate_path, 'r') as path:
        feasibility_rate = float(path.read())
    return feasibility_rate


def average_planning_time(time_path):
    time_data_frame = pd.read_csv(time_path)
    each_plan_time_df = time_data_frame["time"].mean()
    # print ("Mean of time: ", float(each_plan_time_df))
    return float(each_plan_time_df)


def planning_time_for_voronoi(time_path, voronoi_path):
    time_voronoi_metric_dict = dict()

    time_df = pd.read_csv(time_path)
    voronoi_df = pd.read_csv(voronoi_path)
    each_time_list = list(time_df["time"].values)
    each_voronoi_list = list(voronoi_df["voronoi_distance"].values)
    size_of_time_df = time_df.shape[0]
    size_of_voronoi_df = voronoi_df.shape[0]

    if size_of_time_df == size_of_voronoi_df:       # TODO double for loop can solve problem think about it
        for i in range(size_of_time_df):
            if (time_df["i_x"].values[i] == voronoi_df["i_x"].values[i]) & (
                    time_df["i_y"].values[i] == voronoi_df["i_y"].values[i]) & (
                    time_df["g_x"].values[i] == voronoi_df["g_x"].values[i]) & (
                    time_df["g_y"].values[i] == voronoi_df["g_y"].values[i]):
                key = str(time_df["i_x"].values[i]) + ',' + str(time_df["i_y"].values[i]) + ',' + str(
                    time_df["g_x"].values[i]) + ',' + str(time_df["g_y"].values[i])
                time_voronoi_metric_dict[key] = float(each_time_list[i]/each_voronoi_list[i])
            else:
                print_error("Time and Voronoi values did not matched")
                # TODO break is not a good solution think about it
                # break
    else:
        print_error("Time and Voronoi length did not matched")

    return time_voronoi_metric_dict


def average_planning_time_over_voronoi(time_path, voronoi_path):
    sum = 0

    time_df = pd.read_csv(time_path)
    voronoi_df = pd.read_csv(voronoi_path)
    each_time_list = list(time_df["time"].values)
    each_voronoi_list = list(voronoi_df["voronoi_distance"].values)
    size_of_time_df = time_df.shape[0]
    size_of_voronoi_df = voronoi_df.shape[0]

    if size_of_time_df == size_of_voronoi_df:             # TODO double for loop can solve problem think about it
        for i in range(size_of_time_df):
            if (time_df["i_x"].values[i] == voronoi_df["i_x"].values[i]) & (
                    time_df["i_y"].values[i] == voronoi_df["i_y"].values[i]) & (
                    time_df["g_x"].values[i] == voronoi_df["g_x"].values[i]) & (
                    time_df["g_y"].values[i] == voronoi_df["g_y"].values[i]):
                sum = float(each_time_list[i] / each_voronoi_list[i]) + sum
            else:
                print_error("Time and Voronoi values did not matched")
                # TODO break is not a good solution think about it
                # break
    else:
        print_error("Time and Voronoi length did not matched")

    average = sum / size_of_time_df
    return average

def euclidean_length_over_voronoi_distance(euclidean_path, voronoi_path):
    euclidean_voronoi_metric_dict = dict()

    euclidean_df = pd.read_csv(euclidean_path)
    voronoi_df = pd.read_csv(voronoi_path)
    each_euclidean_list = list(euclidean_df["euclidean_distance"].values)
    each_voronoi_list = list(voronoi_df["voronoi_distance"].values)
    size_of_euclidean_df = euclidean_df.shape[0]
    size_of_voronoi_df = voronoi_df.shape[0]

    if size_of_euclidean_df == size_of_voronoi_df:          # TODO double for loop can solve problem think about it
        for i in range(size_of_euclidean_df):
            if (euclidean_df["i_x"].values[i] == voronoi_df["i_x"].values[i]) & (
                    euclidean_df["i_y"].values[i] == voronoi_df["i_y"].values[i]) & (
                    euclidean_df["g_x"].values[i] == voronoi_df["g_x"].values[i]) & (
                    euclidean_df["g_y"].values[i] == voronoi_df["g_y"].values[i]):
                key = str(euclidean_df["i_x"].values[i]) + ',' + str(euclidean_df["i_y"].values[i]) + ',' + str(
                    euclidean_df["g_x"].values[i]) + ',' + str(euclidean_df["g_y"].values[i])
                euclidean_voronoi_metric_dict[key] = float(each_euclidean_list[i] / each_voronoi_list[i])
            else:
                print_error("Euclidean and Voronoi values did not matched")
                # TODO break is not a good solution think about it
                # break
    else:
        print_error("Euclidean and Voronoi length did not matched")

    return euclidean_voronoi_metric_dict

def average_euclidean_length_over_voronoi_distance(euclidean_path, voronoi_path):
    sum = 0

    euclidean_df = pd.read_csv(euclidean_path)
    voronoi_df = pd.read_csv(voronoi_path)
    each_euclidean_list = list(euclidean_df["euclidean_distance"].values)
    each_voronoi_list = list(voronoi_df["voronoi_distance"].values)
    size_of_euclidean_df = euclidean_df.shape[0]
    size_of_voronoi_df = voronoi_df.shape[0]

    if size_of_euclidean_df == size_of_voronoi_df:  # TODO double for loop can solve problem think about it
        for i in range(size_of_euclidean_df):
            if (euclidean_df["i_x"].values[i] == voronoi_df["i_x"].values[i]) & (
                    euclidean_df["i_y"].values[i] == voronoi_df["i_y"].values[i]) & (
                    euclidean_df["g_x"].values[i] == voronoi_df["g_x"].values[i]) & (
                    euclidean_df["g_y"].values[i] == voronoi_df["g_y"].values[i]):
                sum = float(each_euclidean_list[i] / each_voronoi_list[i]) + sum
            else:
                print_error("Euclidean and Voronoi values did not matched")
                # TODO break is not a good solution think about it
                # break
    else:
        print_error("Euclidean and Voronoi length did not matched")

    average = sum / size_of_euclidean_df
    return average


if __name__ == '__main__':
    import traceback
    run_folders = sorted(list(filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling/output/test_planning/*")))))

    for progress, run_folder in enumerate(run_folders):
        print_info("main: compute_metrics {}% {}".format((progress + 1)*100/len(run_folders), run_folder))
        # noinspection PyBroadException
        try:
            compute_metrics(path.expanduser(run_folder))
        except KeyboardInterrupt:
            print_info("aborting due to KeyboardInterrupt")
            break
        except:
            print(traceback.format_exc())
            print_error("failed computing metrics for run folder [{}]".format(run_folder))

    # run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/elysium/performance_modelling/output/localization/*"))))
    # # run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling/output/test_localization/*"))))
    # last_run_folder = sorted(run_folders, key=lambda x: path.getmtime(x))[-1]
    # print("last run folder:", last_run_folder)
    # compute_metrics(last_run_folder)
