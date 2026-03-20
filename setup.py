from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pu4c",
    version="1.4.0",
    packages=find_packages(exclude=["tests"]),
    author="castle945",
    author_email="castle945@njust.edu.cn",
    url="https://github.com/castle945",
    description="A python utils package for castle945",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'rpyc',
        'rich',
        'numpy',
    ],
)