from setuptools import setup, find_packages
import pydl

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pydl',
    version=pydl.__version__,
    description='Deep Learning Algorithms in Python',
    author=pydl.__author__,
    author_email='rthomazigonzalez@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=required
)
