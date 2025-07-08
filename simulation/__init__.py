# simulation/__init__.py

from .setup import Simulation
from .stepper import run
from .cfl_utils import calculate_max_cfl

Simulation.run = run
Simulation.calculate_max_cfl = calculate_max_cfl

# ✅ Alias for pressure field to avoid attribute errors and clarify intent
@property
def pressure_field(self):
    return self.p

Simulation.pressure_field = pressure_field



