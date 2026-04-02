"""Documentation for HeMAC (Heterogeneous Multi-Agent Challenge) environments."""
from __future__ import annotations

from .SimpleFleet import HEMAC_SIMPLE_FLEET_HTML, get_simple_fleet_html
from .Fleet import HEMAC_FLEET_HTML, get_fleet_html
from .ComplexFleet import HEMAC_COMPLEX_FLEET_HTML, get_complex_fleet_html

__all__ = [
    "HEMAC_SIMPLE_FLEET_HTML",
    "get_simple_fleet_html",
    "HEMAC_FLEET_HTML",
    "get_fleet_html",
    "HEMAC_COMPLEX_FLEET_HTML",
    "get_complex_fleet_html",
]
