from setuptools import setup, find_packages

setup(
    name="xtuner_custom",
    version="0.1",
    packages=find_packages(include=["map_fn", "map_fn.*"]),
)