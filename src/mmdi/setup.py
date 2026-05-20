from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'mmdi'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mike Hagenow',
    maintainer_email='wisconsinhci@gmail.com',
    description='Multi-modal Demonstration Interface for ROS 2',
    license='WTFPL',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arduino_handler = mmdi.arduino_handler:main',
            'mode_handler = mmdi.mode_handler:main',
            'natural_handler = mmdi.natural_handler:main',
            'probe_tracker = mmdi.probe_tracker:main',
            'ft_calibrator = mmdi.ft_calibrator:main',
	    'wrench_env_sensor = mmdi.wrench_env_sensor:main',
        ],
    },
)
