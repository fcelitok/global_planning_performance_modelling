#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

# import sys

# print("Python version:", sys.version_info)
# if sys.version_info.major < 3:
#     print("Python version less than 3")
#     sys.exit()

import glob
import argparse
import pickle
import yaml
import pandas as pd
from os import path

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


def get_yaml_by_path(yaml_dict, keys):
    assert(isinstance(keys, list))
    try:
        if len(keys) > 1:
            return get_yaml_by_path(yaml_dict[keys[0]], keys[1:])
        elif len(keys) == 1:
            return yaml_dict[keys[0]]
        else:
            return None
    except (KeyError, TypeError):
        return None


def collect_data(base_run_folder_path, invalidate_cache=False):

    base_run_folder = path.expanduser(base_run_folder_path)
    cache_file_path = path.join(base_run_folder, "run_data_per_waypoint_cache.pkl")

    if not path.isdir(base_run_folder):
        print_error("collect_data: base_run_folder does not exists or is not a directory".format(base_run_folder))
        return None, None

    run_folders = sorted(list(filter(path.isdir, glob.glob(path.abspath(base_run_folder) + '/*'))))

    record_list = list()
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
        metric_results_folder = path.join(run_folder, "metric_results")
        run_info_file_path = path.join(run_folder, "run_info.yaml")
        metrics_file_path = path.join(metric_results_folder, "metrics.yaml")

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

        try:
            metrics_dict = get_yaml(metrics_file_path)
        except IOError:
            print_error("metric_results could not be read: {}".format(metrics_file_path))
            no_output = False
            continue

        run_record = dict()

        for parameter_name, parameter_value in run_info['run_parameters'].items():
            parameter_names.add(parameter_name)
            if type(parameter_value) == list:
                parameter_value = tuple(parameter_value)
            run_record[parameter_name] = parameter_value

        # parameter_names.add('environment_name')  # environment name is already inside
        run_record['environment_name'] = path.basename(path.abspath(run_info['environment_folder']))
        run_record['run_folder'] = run_folder


        # collect per waypoint metric results

        euclidean_length_over_voronoi_distance_per_waypoint_dict = dict()
        euclidean_length_over_voronoi_distance_per_waypoint_list = get_yaml_by_path(metrics_dict, ['euclidean_length_over_voronoi_distance'])
        if euclidean_length_over_voronoi_distance_per_waypoint_list is not None:
            for euclidean_length_over_voronoi_distance_per_waypoint in euclidean_length_over_voronoi_distance_per_waypoint_list:
                if euclidean_length_over_voronoi_distance_per_waypoint is not None and 'i_x' and 'i_y' and 'g_x' and 'g_y' in euclidean_length_over_voronoi_distance_per_waypoint:
                    euclidean_length_over_voronoi_distance_per_waypoint_dict[
                        euclidean_length_over_voronoi_distance_per_waypoint[
                            'i_x'], euclidean_length_over_voronoi_distance_per_waypoint[
                            'i_y'], euclidean_length_over_voronoi_distance_per_waypoint[
                            'g_x'], euclidean_length_over_voronoi_distance_per_waypoint[
                            'g_y']] = euclidean_length_over_voronoi_distance_per_waypoint

        planning_time_over_voronoi_distance_per_waypoint_dict = dict()
        planning_time_over_voronoi_distance_per_waypoint_list = get_yaml_by_path(metrics_dict, ['planning_time_over_voronoi_distance'])
        if planning_time_over_voronoi_distance_per_waypoint_list is not None:
            for planning_time_over_voronoi_distance_per_waypoint in planning_time_over_voronoi_distance_per_waypoint_list:
                if planning_time_over_voronoi_distance_per_waypoint is not None and 'i_x' and 'i_y' and 'g_x' and 'g_y' in planning_time_over_voronoi_distance_per_waypoint:
                    planning_time_over_voronoi_distance_per_waypoint_dict[
                        planning_time_over_voronoi_distance_per_waypoint[
                            'i_x'], planning_time_over_voronoi_distance_per_waypoint[
                            'i_y'], planning_time_over_voronoi_distance_per_waypoint[
                            'g_x'], planning_time_over_voronoi_distance_per_waypoint[
                            'g_y']] = planning_time_over_voronoi_distance_per_waypoint

        feasibility_rate_per_waypoint_dict = dict()
        feasibility_rate_per_waypoint_list = get_yaml_by_path(metrics_dict, ['feasibility_rate'])
        if feasibility_rate_per_waypoint_list is not None:
            for feasibility_rate_per_waypoint in feasibility_rate_per_waypoint_list:
                if feasibility_rate_per_waypoint is not None and 'i_x' and 'i_y' and 'g_x' and 'g_y' in feasibility_rate_per_waypoint:
                    feasibility_rate_per_waypoint_dict[
                        feasibility_rate_per_waypoint[
                            'i_x'], feasibility_rate_per_waypoint[
                            'i_y'], feasibility_rate_per_waypoint[
                            'g_x'], feasibility_rate_per_waypoint[
                            'g_y']] = feasibility_rate_per_waypoint

        mean_passage_width_per_waypoint_dict = dict()
        mean_passage_width_per_waypoint_list = get_yaml_by_path(metrics_dict, ['mean_passage_width'])
        if mean_passage_width_per_waypoint_list is not None:
            for mean_passage_width_per_waypoint in mean_passage_width_per_waypoint_list:
                if mean_passage_width_per_waypoint is not None and 'i_x' and 'i_y' and 'g_x' and 'g_y' in mean_passage_width_per_waypoint:
                    mean_passage_width_per_waypoint_dict[
                        mean_passage_width_per_waypoint[
                            'i_x'], mean_passage_width_per_waypoint[
                            'i_y'], mean_passage_width_per_waypoint[
                            'g_x'], mean_passage_width_per_waypoint[
                            'g_y']] = mean_passage_width_per_waypoint

        mean_normalized_passage_width_per_waypoint_dict = dict()
        mean_normalized_passage_width_per_waypoint_list = get_yaml_by_path(metrics_dict, ['mean_normalized_passage_width'])
        if mean_normalized_passage_width_per_waypoint_list is not None:
            for mean_normalized_passage_width_per_waypoint in mean_normalized_passage_width_per_waypoint_list:
                if mean_normalized_passage_width_per_waypoint is not None and 'i_x' and 'i_y' and 'g_x' and 'g_y' in mean_normalized_passage_width_per_waypoint:
                    mean_normalized_passage_width_per_waypoint_dict[
                        mean_normalized_passage_width_per_waypoint[
                            'i_x'], mean_normalized_passage_width_per_waypoint[
                            'i_y'], mean_normalized_passage_width_per_waypoint[
                            'g_x'], mean_normalized_passage_width_per_waypoint[
                            'g_y']] = mean_normalized_passage_width_per_waypoint

        minimum_passage_width_per_waypoint_dict = dict()
        minimum_passage_width_per_waypoint_list = get_yaml_by_path(metrics_dict, ['minimum_passage_width'])
        if minimum_passage_width_per_waypoint_list is not None:
            for minimum_passage_width_per_waypoint in minimum_passage_width_per_waypoint_list:
                if minimum_passage_width_per_waypoint is not None and 'i_x' and 'i_y' and 'g_x' and 'g_y' in minimum_passage_width_per_waypoint:
                    minimum_passage_width_per_waypoint_dict[
                        minimum_passage_width_per_waypoint[
                            'i_x'], minimum_passage_width_per_waypoint[
                            'i_y'], minimum_passage_width_per_waypoint[
                            'g_x'], minimum_passage_width_per_waypoint[
                            'g_y']] = minimum_passage_width_per_waypoint

        for waypoint_position in minimum_passage_width_per_waypoint_dict.keys():
            run_record_per_waypoint = run_record.copy()

            run_record_per_waypoint['i_x'] = waypoint_position[0]   # position i_x
            run_record_per_waypoint['i_y'] = waypoint_position[1]   # position i_y
            run_record_per_waypoint['g_x'] = waypoint_position[2]   # position g_x
            run_record_per_waypoint['g_y'] = waypoint_position[3]   # position g_y

            all_euclidean_length_over_voronoi_distance_metrics = get_yaml_by_path(euclidean_length_over_voronoi_distance_per_waypoint_dict, [waypoint_position])
            if all_euclidean_length_over_voronoi_distance_metrics is not None:
                for euclidean_length_over_voronoi_distance_metric_name, euclidean_length_over_voronoi_distance_metric_value in all_euclidean_length_over_voronoi_distance_metrics.items():
                    run_record_per_waypoint['euclidean_length_over_voronoi_distance_' + euclidean_length_over_voronoi_distance_metric_name] = euclidean_length_over_voronoi_distance_metric_value

            all_planning_time_over_voronoi_distance_metrics = get_yaml_by_path(planning_time_over_voronoi_distance_per_waypoint_dict, [waypoint_position])
            if all_planning_time_over_voronoi_distance_metrics is not None:
                for planning_time_over_voronoi_distance_metric_name, planning_time_over_voronoi_distance_metric_value in all_planning_time_over_voronoi_distance_metrics.items():
                    run_record_per_waypoint['planning_time_over_voronoi_distance_' + planning_time_over_voronoi_distance_metric_name] = planning_time_over_voronoi_distance_metric_value

            all_feasibility_rate_metrics = get_yaml_by_path(feasibility_rate_per_waypoint_dict, [waypoint_position])
            if all_feasibility_rate_metrics is not None:
                for feasibility_rate_metric_name, feasibility_rate_metric_value in all_feasibility_rate_metrics.items():
                    run_record_per_waypoint['feasibility_rate_' + feasibility_rate_metric_name] = feasibility_rate_metric_value

            all_mean_passage_width_metrics = get_yaml_by_path(mean_passage_width_per_waypoint_dict, [waypoint_position])
            if all_mean_passage_width_metrics is not None:
                for mean_passage_width_metric_name, mean_passage_width_metric_value in all_mean_passage_width_metrics.items():
                    run_record_per_waypoint['mean_passage_width_' + mean_passage_width_metric_name] = mean_passage_width_metric_value

            all_mean_normalized_passage_width_metrics = get_yaml_by_path(mean_normalized_passage_width_per_waypoint_dict, [waypoint_position])
            if all_mean_normalized_passage_width_metrics is not None:
                for mean_normalized_passage_width_metric_name, mean_normalized_passage_width_metric_value in all_mean_normalized_passage_width_metrics.items():
                    run_record_per_waypoint[
                        'mean_normalized_passage_width_' + mean_normalized_passage_width_metric_name] = mean_normalized_passage_width_metric_value

            minimum_passage_width_metrics = get_yaml_by_path(minimum_passage_width_per_waypoint_dict, [waypoint_position])
            if minimum_passage_width_metrics is not None:
                for minimum_passage_width_metrics_name, minimum_passage_width_metric_value in minimum_passage_width_metrics.items():
                    run_record_per_waypoint['minimum_passage_width_' + minimum_passage_width_metrics_name] = minimum_passage_width_metric_value

            record_list.append(run_record_per_waypoint)

        print_info("collect_data: reading run data: {}% {}/{} {}".format(int((i + 1)*100/len(run_folders)), i, len(run_folders), path.basename(run_folder)), replace_previous_line=no_output)
        no_output = True

    df = pd.DataFrame(record_list)

    # save cache
    if cache_file_path is not None:
        cache = {'df': df, 'parameter_names': parameter_names}
        with open(cache_file_path, 'wb') as f:
            pickle.dump(cache, f, protocol=2)

    return df, parameter_names


if __name__ == '__main__':
    default_base_run_folder = "~/ds/performance_modelling/output/test_planning"

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Execute the analysis of the benchmark results.')
    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder containing the result the runs. Defaults to {}'.format(default_base_run_folder),
                        type=str,
                        default=default_base_run_folder,
                        required=False)

    parser.add_argument('-i', dest='invalidate_cache',
                        help='If set, all the data is re-read.',
                        action='store_true',
                        default=True,  # TODO does nothing, repair or remove
                        required=False)

    args = parser.parse_args()
    import time
    s = time.time()
    run_data_df, params = collect_data(args.base_run_folder, args.invalidate_cache)
    print(run_data_df)
    print(run_data_df.columns)
    print("collect_data: ", time.time() - s, "s")

