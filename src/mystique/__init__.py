from .kirby import Kirby
from .synthetic_control import SyntheticControl
from . import utils

__all__ = [
    "Kirby",
    "SyntheticControl",
    "utils",
]

import colorful as colors

colors.use_8_ansi_colors()
