#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import argparse
from os import path

from global_planning_performance_modelling_ros.global_planning_benchmark_run import BenchmarkRun
from performance_modelling_py.benchmark_execution.grid_benchmarking import execute_grid_benchmark

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Execute the benchmark')  # this module makes it easy to write user-friendly command-line interfaces.

    parser.add_argument('-e', dest='environment_dataset_folders',
                        help='Dataset folders contain the environment data. Use wildcards to select multiple folders. Only folders are selected, files are ignored.',
                        type=str,
                        default="~/ds/performance_modelling/test_datasets/dataset/airlab",
                        # when used ~/ds/performance_modelling/test_datasets/dataset/* it will take all environments
                        required=False)

    parser.add_argument('-c', dest='grid_benchmark_configuration',
                        help='Yaml file with the configuration of the benchmark.',
                        type=str,
                        default="~/turtlebot3_melodic_ws/src/global_planning_performance_modelling/config/benchmark_configurations/global_planning_grid_benchmark_sbpl.yaml",
                        required=False)

    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder in which the result of each run will be placed.',
                        type=str,
                        default="~/ds/performance_modelling/output/test_planning/",
                        required=False)

    parser.add_argument('-n', '--num-runs', dest='num_runs',
                        help='Number of runs to be executed for each combination of configurations.',
                        type=int,
                        default=1,
                        required=False)

    parser.add_argument('--ignore-previous-runs', dest='ignore_previous_runs',
                        help='When set the the previous runs in base_run_folder are ignored when counting the param combinations already executed.',
                        action='store_true',
                        required=False)

    parser.add_argument('--no-shuffle', dest='no_shuffle',
                        help='When set the order of the combinations is not randomized.',
                        action='store_true',
                        required=False)

    parser.add_argument('-g', '--headless', dest='headless',
                        help='When set the components are run with GUI (opposite of headless).',
                        action='store_true',
                        required=False)

    parser.add_argument('-s', '--show-ros-info', dest='show_ros_info',
                        help='When set the component nodes are launched with output="screen".',
                        action='store_true',
                        required=False)

    args = parser.parse_args()
    base_run_folder = path.expanduser(args.base_run_folder)
    environment_folders = sorted(filter(path.isdir, glob.glob(path.expanduser(args.environment_dataset_folders))))
    grid_benchmark_configuration = path.expanduser(args.grid_benchmark_configuration)

    execute_grid_benchmark(benchmark_run_object=BenchmarkRun,
                           grid_benchmark_configuration=grid_benchmark_configuration,
                           environment_folders=environment_folders,
                           base_run_folder=base_run_folder,
                           num_runs=args.num_runs,
                           ignore_executed_params_combinations=args.ignore_previous_runs,
                           shuffle=not args.no_shuffle,
                           headless=args.headless,
                           show_ros_info=args.show_ros_info)
