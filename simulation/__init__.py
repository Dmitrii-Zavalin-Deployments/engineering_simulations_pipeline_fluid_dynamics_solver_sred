# simulation/__init__.py

from .setup import Simulation
from .stepper import run
from .cfl_utils import calculate_max_cfl

Simulation.run = run
Simulation.calculate_max_cfl = calculate_max_cfl



