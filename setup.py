from setuptools import setup

setup(
    name = 'pydl',
    version = '0.0.1',
    description = 'pydl cli',
    author = 'Rafael Gonzalez',
    author_email='rthomazigonzalez@gmail.com',
    license = 'MIT',
    packages=['cli'],
    scripts=['cli/run.py'],
    entry_points = {'console_scripts': ['pydl = cli.run',],},
)