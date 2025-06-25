# src/physics/__init__.py

# This file makes 'physics' a Python package.
# You can also import specific functions to make them directly accessible
# from the 'physics' package.

from .boundary_conditions_applicator import apply_boundary_conditions
from .boundary_data_parser import identify_boundary_nodes