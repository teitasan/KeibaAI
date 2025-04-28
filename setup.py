from setuptools import setup, find_packages

setup(
    name="keiba_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "requests",
    ],
)
