from setuptools import find_packages, setup

setup(
    name="LCA-normative-modelling",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        include=[
            "LCA*",
            "modelling*",
            "preprocess*",
            "discover*",
        ],
    ),
)