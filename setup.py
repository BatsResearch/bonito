from setuptools import find_packages, setup

requirements = [
    "transformers",
    "datasets",
    "vllm",
]

dev_requirements = [
    "pytest",
]

setup(
    name="bonito",
    version="0.0.3",
    url="https://github.com/BatsResearch/bonito",
    python_requires=">=3.9",
    install_requires=requirements,
    extra_requires={"dev": dev_requirements},
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    description="A lightweight wrapper for the Bonito model.",
    packages=find_packages(),
)
