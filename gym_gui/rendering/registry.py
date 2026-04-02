"""Renderer registry responsible for instantiating strategies by render mode."""

from __future__ import annotations

from typing import Callable, Dict

from qtpy import QtWidgets

from gym_gui.core.enums import RenderMode
from gym_gui.rendering.interfaces import RendererContext, RendererStrategy

RendererFactory = Callable[[QtWidgets.QWidget | None], RendererStrategy]


class RendererRegistry:
    """Service that creates renderer strategies on demand."""

    def __init__(self) -> None:
        self._factories: Dict[RenderMode, RendererFactory] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, mode: RenderMode, factory: RendererFactory) -> None:
        self._factories[mode] = factory

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def create(self, mode: RenderMode, parent: QtWidgets.QWidget | None = None) -> RendererStrategy:
        factory = self._factories.get(mode)
        if factory is None:
            raise KeyError(f"No renderer strategy registered for mode '{mode.value}'")
        strategy = factory(parent)
        return strategy

    def is_registered(self, mode: RenderMode) -> bool:
        return mode in self._factories

    def supported_modes(self) -> tuple[RenderMode, ...]:
        return tuple(self._factories.keys())


def create_default_renderer_registry() -> RendererRegistry:
    """Instantiate a registry pre-populated with supported strategies."""

    from gym_gui.rendering.strategies.grid import GridRendererStrategy
    from gym_gui.rendering.strategies.rgb import RgbRendererStrategy

    registry = RendererRegistry()
    registry.register(RenderMode.GRID, lambda parent=None: GridRendererStrategy(parent=parent))
    registry.register(RenderMode.RGB_ARRAY, lambda parent=None: RgbRendererStrategy(parent=parent))

    # Optional: MosaicMalmo renderer (experimental)
    try:
        from gym_gui.rendering.strategies.mosaic_malmo import MosaicMalmoRendererStrategy  # pragma: no cover
        registry.register(RenderMode.MOSAIC_MALMO, lambda parent=None: MosaicMalmoRendererStrategy(parent=parent))
    except ImportError:  # pragma: no cover
        pass  # MosaicMalmo renderer is optional/experimental

    return registry


__all__ = ["RendererRegistry", "create_default_renderer_registry", "RendererContext", "RendererStrategy"]
