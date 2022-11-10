from setuptools import find_packages,setup
from typing import List
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Tomato disease detection model',
    version='1.0',
    author="prikshit singh",
    author_email="prikshitsingh792gmail.com",
    install_requires=requirements,
)