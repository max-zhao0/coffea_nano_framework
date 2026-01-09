"""
    Setup script for the coffea-nano-framework
"""
from setuptools import setup, find_packages

setup(
    name='coffea-nano-framework',
    version='0.0.1',
    package_dir={'': 'src'},  # Tells setuptools that packages are under src
    packages=find_packages(where='src'),  # Looks for packages in src directory
)
