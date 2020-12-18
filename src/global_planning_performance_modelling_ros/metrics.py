#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import os
from os import path
import yaml
import pandas as pd
import numpy as np
from PIL import Image
from skimage.draw import line
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from performance_modelling_py.environment.ground_truth_map import GroundTruthMap

from performance_modelling_py.utils import print_info, print_error, backup_file_if_exists
#from performance_modelling_py.metrics.localization_metrics import trajectory_length_metric, absolute_localization_error_metrics, absolute_error_vs_voronoi_radius, absolute_error_vs_scan_range, absolute_error_vs_geometric_similarity, \
#    relative_localization_error_metrics
# from performance_modelling_py.metrics.computation_metrics import cpu_and_memory_usage_metrics
# from performance_modelling_py.visualisation.trajectory_visualisation import save_trajectories_plot


def compute_metrics(run_output_folder):    # run_output_folder: /home/furkan/ds/performance_modelling/output/test_planning/session_2020-09-30_17-09-01_964405_run_000000000

    metrics_result_dict = dict()

    run_info_path = path.join(run_output_folder, "run_info.yaml")
    if not path.exists(run_info_path) or not path.isfile(run_info_path):
        print_error("run info file does not exists")

    with open(run_info_path) as run_info_file:
        run_info = yaml.safe_load(run_info_file)

    environment_folder = run_info['environment_folder']

    # localization metrics
    execution_time_path = path.join(run_output_folder, "benchmark_data", "plan_output", "execution_time.csv")
    voronoi_distance_path = path.join(run_output_folder, "benchmark_data", "plan_output", "voronoi_distance.csv")
    euclidean_distance_path = path.join(run_output_folder, "benchmark_data", "plan_output", "euclidean_distance.csv")
    feasibility_rate_path = path.join(run_output_folder, "benchmark_data", "plan_output", "feasibility_rate.csv")
    mean_passage_width_path = path.join(run_output_folder, "benchmark_data", "plan_output", "mean_passage_width.csv")
    mean_normalized_passage_width_path = path.join(run_output_folder, "benchmark_data", "plan_output", "mean_normalized_passage_width.csv")
    minimum_passage_width_path = path.join(run_output_folder, "benchmark_data", "plan_output", "minimum_passage_width.csv")


    # logs_folder_path = path.join(run_output_folder, "logs")

    metrics_result_folder_path = path.join(run_output_folder, "metric_results")
    metrics_result_file_path = path.join(metrics_result_folder_path, "metrics.yaml")

    if path.exists(metrics_result_file_path):
        print_info("metrics file already exists, not overwriting [{}]".format(metrics_result_file_path))
    else:
        print_info("average planning time")
        metrics_result_dict['average_planning_time'] = average_planning_time(execution_time_path)

        print_info("planning time over voronoi distance")
        metrics_result_dict['planning_time_over_voronoi_distance'] = planning_time_over_voronoi_distance(execution_time_path, voronoi_distance_path)

        print_info("normalised planning time")
        metrics_result_dict['normalised_planning_time'] = normalised_planning_time(execution_time_path, voronoi_distance_path)

        print_info("euclidean length over voronoi distance")
        metrics_result_dict['euclidean_length_over_voronoi_distance'] = euclidean_length_over_voronoi_distance(euclidean_distance_path, voronoi_distance_path)

        print_info("normalised plan length")
        metrics_result_dict['normalised_plan_length'] = normalised_plan_length(euclidean_distance_path, voronoi_distance_path)

        print_info("feasibility rate of map")
        metrics_result_dict['feasibility_rate'] = feasibility_rate(feasibility_rate_path)

        print_info("mean passage width")
        metrics_result_dict['mean_passage_width'] = mean_passage_width(mean_passage_width_path)

        print_info("mean normalized passage width")
        metrics_result_dict['mean_normalized_passage_width'] = mean_normalized_passage_width(mean_normalized_passage_width_path)

        print_info("minimum passage width")
        metrics_result_dict['minimum_passage_width'] = minimum_passage_width(minimum_passage_width_path)

        print_info("number_of_walls_traversed")
        metrics_result_dict['number_of_walls_traversed'] = number_of_walls_traversed(feasibility_rate_path, environment_folder)

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


def number_of_walls_traversed(feasibility_rate_path, environment_folder):
    draw_map = 1            # If you do not want to see maps make draw_map = 0

    feasibility_rate_df = pd.read_csv(feasibility_rate_path)
    ground_truth_map_info_path = path.join(environment_folder, "data", "map.yaml")
    ground_truth_map_info = GroundTruthMap(ground_truth_map_info_path)

    ground_truth_map_png_path = path.join(environment_folder, "data", "map.pgm")
    ground_truth_map_png = Image.open(ground_truth_map_png_path)

    ground_truth_map_pixels = ground_truth_map_png.load()

    number_of_walls_traversed_list = list()

    for index, row in feasibility_rate_df.iterrows():
        if row['path_feasibility'] == 1:
            number_of_walls_traversed_dict = dict()

            p_i_x, p_i_y = ground_truth_map_info.map_frame_to_image_coordinates([row['i_x'], row['i_y']])
            p_g_x, p_g_y = ground_truth_map_info.map_frame_to_image_coordinates([row['g_x'], row['g_y']])

            map_line = np.array(line(p_i_x, p_i_y, p_g_x, p_g_y)).transpose()

            ground_truth_free_cell_count = 0
            ground_truth_occupied_cell_count = 0
            ground_truth_unknown_cell_count = 0

            list_of_occupied = list()
            for item in map_line:
                # print(item)
                i = int(item[0])
                j = int(item[1])
                ground_truth_free_cell_count += ground_truth_map_pixels[i, j] == (255, 255, 255)
                ground_truth_occupied_cell_count += ground_truth_map_pixels[i, j] == (0, 0, 0)
                ground_truth_unknown_cell_count += ground_truth_map_pixels[i, j] == (205, 205, 205)
                if ground_truth_map_pixels[i, j] == (0, 0, 0):
                    list_of_occupied.append([i, j])
            number_of_walls_traversed = ground_truth_occupied_cell_count / len(map_line)
            list_of_occupied_np = np.array(list_of_occupied)

            number_of_walls_traversed_dict["i_x"] = float(row['i_x'])
            number_of_walls_traversed_dict["i_y"] = float(row['i_y'])
            number_of_walls_traversed_dict["g_x"] = float(row['g_x'])
            number_of_walls_traversed_dict["g_y"] = float(row['g_y'])
            number_of_walls_traversed_dict["number_of_walls_traversed"] = float(number_of_walls_traversed)

            number_of_walls_traversed_list.append(number_of_walls_traversed_dict)

        if draw_map == 1:
            image = mpimg.imread(ground_truth_map_png_path)
            plt.imshow(image)
            plt.scatter(map_line[:, 0], map_line[:, 1], marker=".", color="black", s=1)
            if list_of_occupied_np.size >= 1:
                plt.scatter(list_of_occupied_np[:, 0], list_of_occupied_np[:, 1], marker=".", color="red", s=1)
            plt.show()
    return number_of_walls_traversed_list


def mean_passage_width(mean_passage_width_path):
    mean_passage_width_list = list()
    mean_passage_width_df = pd.read_csv(mean_passage_width_path)

    for i in range(mean_passage_width_df.shape[0]):
        mean_passage_width_dict = dict()
        mean_passage_width_dict["i_x"] = float(mean_passage_width_df["i_x"].values[i])
        mean_passage_width_dict["i_y"] = float(mean_passage_width_df["i_y"].values[i])
        mean_passage_width_dict["g_x"] = float(mean_passage_width_df["g_x"].values[i])
        mean_passage_width_dict["g_y"] = float(mean_passage_width_df["g_y"].values[i])
        mean_passage_width_dict["mean_passage_width_of_path"] = float(mean_passage_width_df["mean_passage_width"].values[i])

        mean_passage_width_list.append(mean_passage_width_dict)
    return mean_passage_width_list


def mean_normalized_passage_width(mean_normalized_passage_width_path):
    mean_normalized_passage_width_list = list()
    mean_normalized_passage_width_df = pd.read_csv(mean_normalized_passage_width_path)

    for i in range(mean_normalized_passage_width_df.shape[0]):
        mean_normalized_passage_width_dict = dict()
        mean_normalized_passage_width_dict["i_x"] = float(mean_normalized_passage_width_df["i_x"].values[i])
        mean_normalized_passage_width_dict["i_y"] = float(mean_normalized_passage_width_df["i_y"].values[i])
        mean_normalized_passage_width_dict["g_x"] = float(mean_normalized_passage_width_df["g_x"].values[i])
        mean_normalized_passage_width_dict["g_y"] = float(mean_normalized_passage_width_df["g_y"].values[i])
        mean_normalized_passage_width_dict["mean_normalized_passage_width_of_path"] = float(
            mean_normalized_passage_width_df["mean_normalized_passage_width"].values[i])

        mean_normalized_passage_width_list.append(mean_normalized_passage_width_dict)
    return mean_normalized_passage_width_list

def minimum_passage_width(minimum_passage_width_path):
    minimum_passage_width_list = list()
    mminimum_passage_width_df = pd.read_csv(minimum_passage_width_path)

    for i in range(mminimum_passage_width_df.shape[0]):
        minimum_passage_width_dict = dict()
        minimum_passage_width_dict["i_x"] = float(mminimum_passage_width_df["i_x"].values[i])
        minimum_passage_width_dict["i_y"] = float(mminimum_passage_width_df["i_y"].values[i])
        minimum_passage_width_dict["g_x"] = float(mminimum_passage_width_df["g_x"].values[i])
        minimum_passage_width_dict["g_y"] = float(mminimum_passage_width_df["g_y"].values[i])
        minimum_passage_width_dict["minimum_passage_width_of_path"] = float(mminimum_passage_width_df["minimum_passage_width"].values[i])

        minimum_passage_width_list.append(minimum_passage_width_dict)
    return minimum_passage_width_list


def feasibility_rate(feasibility_rate_path):

    feasibility_rate_list = list()
    feasibility_rate_df = pd.read_csv(feasibility_rate_path)

    for i in range(feasibility_rate_df.shape[0]):
        feasibility_rate_dict = dict()
        feasibility_rate_dict["i_x"] = float(feasibility_rate_df["i_x"].values[i])
        feasibility_rate_dict["i_y"] = float(feasibility_rate_df["i_y"].values[i])
        feasibility_rate_dict["g_x"] = float(feasibility_rate_df["g_x"].values[i])
        feasibility_rate_dict["g_y"] = float(feasibility_rate_df["g_y"].values[i])
        feasibility_rate_dict["feasibility_rate_of_path"] = float(feasibility_rate_df["path_feasibility"].values[i])

        feasibility_rate_list.append(feasibility_rate_dict)
    return feasibility_rate_list


def average_planning_time(time_path):
    time_data_frame = pd.read_csv(time_path)
    each_plan_time_df = time_data_frame["time"].mean()
    # print ("Mean of time: ", float(each_plan_time_df))
    return float(each_plan_time_df)


def planning_time_over_voronoi_distance(time_path, voronoi_path):

    time_voronoi_list = list()
    time_df = pd.read_csv(time_path)
    voronoi_df = pd.read_csv(voronoi_path)
    each_time_list = list(time_df["time"].values)
    each_voronoi_list = list(voronoi_df["voronoi_distance"].values)
    size_of_time_df = time_df.shape[0]
    size_of_voronoi_df = voronoi_df.shape[0]

    for i in range(size_of_time_df):
        for j in range(size_of_voronoi_df):
            if (time_df["i_x"].values[i] == voronoi_df["i_x"].values[j]) & (
                time_df["i_y"].values[i] == voronoi_df["i_y"].values[j]) & (
                time_df["g_x"].values[i] == voronoi_df["g_x"].values[j]) & (
                    time_df["g_y"].values[i] == voronoi_df["g_y"].values[j]):
                time_voronoi_metric_dict = dict()
                time_voronoi_metric_dict["i_x"] = float(time_df["i_x"].values[i])
                time_voronoi_metric_dict["i_y"] = float(time_df["i_y"].values[i])
                time_voronoi_metric_dict["g_x"] = float(time_df["g_x"].values[i])
                time_voronoi_metric_dict["g_y"] = float(time_df["g_y"].values[i])
                time_voronoi_metric_dict["normalized_planning_time_for_each_path"] = float(each_time_list[i]/each_voronoi_list[j])
                time_voronoi_list.append(time_voronoi_metric_dict)

    return time_voronoi_list


def normalised_planning_time(time_path, voronoi_path):
    sum = 0

    time_df = pd.read_csv(time_path)
    voronoi_df = pd.read_csv(voronoi_path)
    each_time_list = list(time_df["time"].values)
    each_voronoi_list = list(voronoi_df["voronoi_distance"].values)
    size_of_time_df = time_df.shape[0]
    size_of_voronoi_df = voronoi_df.shape[0]

    # TODO double for loop can solve problem think about it
    for i in range(size_of_time_df):
        for j in range(size_of_voronoi_df):
            if (time_df["i_x"].values[i] == voronoi_df["i_x"].values[j]) & (
                    time_df["i_y"].values[i] == voronoi_df["i_y"].values[j]) & (
                    time_df["g_x"].values[i] == voronoi_df["g_x"].values[j]) & (
                    time_df["g_y"].values[i] == voronoi_df["g_y"].values[j]):
                sum = float(each_time_list[i] / each_voronoi_list[j]) + sum
    if size_of_time_df <= 0:
        print_error("size_of_time_df is equal or lover zero.")
        average = 0
    else:
        average = sum / size_of_time_df
    return average


def euclidean_length_over_voronoi_distance(euclidean_path, voronoi_path):

    euclidean_voronoi_list = list()
    euclidean_df = pd.read_csv(euclidean_path)
    voronoi_df = pd.read_csv(voronoi_path)
    each_euclidean_list = list(euclidean_df["euclidean_distance"].values)
    each_voronoi_list = list(voronoi_df["voronoi_distance"].values)
    size_of_euclidean_df = euclidean_df.shape[0]
    size_of_voronoi_df = voronoi_df.shape[0]

    # TODO double for loop can solve problem think about it # check for join in data frame (innerjoin)
    for i in range(size_of_euclidean_df):
        for j in range(size_of_voronoi_df):
            if (euclidean_df["i_x"].values[i] == voronoi_df["i_x"].values[j]) & (
                    euclidean_df["i_y"].values[i] == voronoi_df["i_y"].values[j]) & (
                    euclidean_df["g_x"].values[i] == voronoi_df["g_x"].values[j]) & (
                    euclidean_df["g_y"].values[i] == voronoi_df["g_y"].values[j]):
                euclidean_voronoi_metric_dict = dict()
                euclidean_voronoi_metric_dict["i_x"] = float(euclidean_df["i_x"].values[i])
                euclidean_voronoi_metric_dict["i_y"] = float(euclidean_df["i_y"].values[i])
                euclidean_voronoi_metric_dict["g_x"] = float(euclidean_df["g_x"].values[i])
                euclidean_voronoi_metric_dict["g_y"] = float(euclidean_df["g_y"].values[i])
                euclidean_voronoi_metric_dict["normalised_plan_length_for_each_path"] = float(each_euclidean_list[i] / each_voronoi_list[j])
                euclidean_voronoi_list.append(euclidean_voronoi_metric_dict)
    return euclidean_voronoi_list


def normalised_plan_length(euclidean_path, voronoi_path):
    sum = 0

    euclidean_df = pd.read_csv(euclidean_path)
    voronoi_df = pd.read_csv(voronoi_path)
    each_euclidean_list = list(euclidean_df["euclidean_distance"].values)
    each_voronoi_list = list(voronoi_df["voronoi_distance"].values)
    size_of_euclidean_df = euclidean_df.shape[0]
    size_of_voronoi_df = voronoi_df.shape[0]

    for i in range(size_of_euclidean_df):
        for j in range(size_of_voronoi_df):
            if (euclidean_df["i_x"].values[i] == voronoi_df["i_x"].values[j]) & (
                    euclidean_df["i_y"].values[i] == voronoi_df["i_y"].values[j]) & (
                    euclidean_df["g_x"].values[i] == voronoi_df["g_x"].values[j]) & (
                    euclidean_df["g_y"].values[i] == voronoi_df["g_y"].values[j]):
                sum = float(each_euclidean_list[i] / each_voronoi_list[j]) + sum
    if size_of_euclidean_df <= 0:
        print_error("size_of_euclidean_df is equal or lover zero.")
        average = 0
    else:
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
