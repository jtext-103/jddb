from setuptools import setup, find_packages

setup(
    name='JDDB',
    version='0.1.0',
    description='J-TEXT Disruption Database Python package',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas'
    ],
)
