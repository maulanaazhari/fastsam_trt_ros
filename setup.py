"""
Setup for fastsam_trt_ros
"""
import os
from glob import glob

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(packages=['fastsam_trt_ros'], package_dir={'': 'src'})

setup(**d)