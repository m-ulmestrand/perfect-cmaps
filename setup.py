from setuptools import setup, find_packages

setup(
    name="perfect_cmaps",
    version="1.0.0",
    author="Mattias Ulmestrand",
    author_email="m.ulmestrand@gmail.com",
    description="Create amazing perceptually uniform colormaps with ease!",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/m-ulmestrand/perfect-cmaps",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License, Version 2.0",
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
    ],
)