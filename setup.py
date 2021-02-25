import os
from glob import glob
from setuptools import setup

package_name = 'global_planning_performance_modelling'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Include our package.xml file
        (os.path.join('share', package_name), ['package.xml']),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Enrico Piazza',
    maintainer_email='enrico.piazza@polimi.it',
    description='Execute benchmark of ROS localization packages to build a performance model',
    license='BSD License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'global_planning_benchmark_supervisor = global_planning_performance_modelling.global_planning_benchmark_supervisor:main',
            'execute_grid_benchmark = global_planning_performance_modelling.execute_grid_benchmark:main',
        ],
    },
)
