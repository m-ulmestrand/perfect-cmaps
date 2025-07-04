from setuptools import setup, find_packages
import re


# Read the version directly from __init__.py
with open("perfect_cmaps/__init__.py", "r") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)


setup(
    name="perfect_cmaps",
    version=version,
    author="Mattias Ulmestrand",
    author_email="m.ulmestrand@gmail.com",
    description="Create amazing perceptually uniform colormaps with ease!",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/m-ulmestrand/perfect-cmaps",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "numba",
        "pandas",
        "matplotlib",
        "colour-science",
        "scipy",
        "opencv-python-headless",
        "importlib_resources",
        "appdirs",
        "tqdm"
    ],
)