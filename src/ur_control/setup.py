from setuptools import find_packages, setup

package_name = 'ur_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'viser',
        'robot_descriptions',
        'yourdfpy',
    ],
    zip_safe=True,
    maintainer='rt2',
    maintainer_email='rt2@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'position_controller = ur_control.position_controller:main',
        'sm_teleop = ur_control.sm_teleop:main',
        'compliance_controller = ur_control.compliance_controller:main',
        'freedrive_controller = ur_control.freedrive_controller:main',
        'viser_viewer = ur_control.viser_viewer:main',
    ],
},

)
