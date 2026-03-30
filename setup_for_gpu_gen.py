import os
from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='sbvr',
    packages=['sbvr', 'sbvr.kernels'],
    install_requires=requirements,
)