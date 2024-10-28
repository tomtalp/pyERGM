from setuptools import setup, find_packages

setup(
    name="pyERGM",
    version="0.1.0",
    description="Exponential Random Graph Models in Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tomtalp/pyERGM",
    packages=find_packages(),
    python_requires=">=3.6",
)