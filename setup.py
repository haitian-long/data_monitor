#!/usr/bin/env python3

from glob import glob
import os

from setuptools import find_packages, setup

package_name = "data_monitor"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*_launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Realtime odometry plotting and range monitoring tool.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "odometry_monitor = odometry_monitor.odometry_monitor_node:main",
        ],
    },
)
