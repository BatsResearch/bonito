import os
from setuptools import setup, find_packages

requirements = [
    "transformers",
    "datasets",
    "vllm",
]

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bonito-llm",
    version="0.1.0",
    author="Nihal Nayak, Yiyang Nan, Avi Trost, and Stephen Bach",
    author_email="nnayak2@cs.brown.edu",
    license="BSD-3-Clause",
    url="https://github.com/BatsResearch/bonito",
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description=(
        "A lightweight library for generating synthetic instruction tuning "
        "datasets for your data without GPT."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
)
