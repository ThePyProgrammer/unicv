# src/unicv/__init__.py

"""
This is the unicv package.

It provides model implementations and core neural-network building blocks
for heterogeneous computer vision tasks.
"""

from unicv import models, nn, utils

__version__ = "0.1.0"

__all__ = ["models", "nn", "utils", "__version__"]