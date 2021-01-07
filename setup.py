"""This file contains rudimentary setup code for the project"""

import setuptools

with open("README.md", "r") as rm:
    long_description = rm.read()

setuptools.setup(
    name="Ouroboros",
    version="0.0.1",
    author="Flawnson & Kevin",
    author_email="flawnsontong1@gmail.com & kshen3778@gmail.com",
    description="Introverted NNs for world domination",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)