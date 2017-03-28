from setuptools import setup, find_packages
import pydl

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pydl',
    version=pydl.__version__,
    description='Deep Learning Algorithms in Python',
    author='Rafael Gonzalez',
    author_email='rthomazigonzalez@gmail.com',
    license='MIT',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['pydl=cli.main:main'],
    },
    install_requires=required
)
