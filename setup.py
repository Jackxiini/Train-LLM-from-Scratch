from setuptools import setup, find_packages

setup(
    name="llm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "datasets",
        "wandb",  # optional, for logging
    ],
    python_requires=">=3.8",
) 