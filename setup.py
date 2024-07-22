"""MLROSe setup file."""

# Author: Genevieve Hayes
# License: BSD 3 clause
# Forked and optimized by Kyle Nakamura

from setuptools import setup


setup(
    name="mlrose_torch",
    version="0.1.0",
    description="A highly optimized fork of 'MLROSe: Machine Learning, Randomized Optimization and Search', "
                "now with PyTorch and advanced NumPy vectorization.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/knakamura13/mlrose-torch",
    author="Kyle Nakamura",
    license="BSD",
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=["mlrose_torch"],
    install_requires=["numpy", "scipy", "scikit-learn", "six", "torch", "setuptools"],
    python_requires=">=3.10",
    zip_safe=False,
    project_urls={
        'Original Project': 'https://github.com/gkhayes/mlrose',
    }
)
