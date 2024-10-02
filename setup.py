from setuptools import setup, find_packages
import pathlib

requirements = pathlib.Path('requirements.txt').read_text().splitlines()

setup(
    name="cvskpd",
    version="0.1.1",
    packages=find_packages(),
    install_requires=requirements,
)