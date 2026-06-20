"""
Expose main classes and functions for easy import from the core package
"""
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .canvas import (ImageGraphicsView, ImageStorage, ImageWrapper,
                         MemoryEfficientImageCache, MetaData,
                         ReferenceGraphicsView)
    from .cell_intensity import CellIntensity
    from .cell_layer_alignment import (CellLayerAligner,
                                       calculate_alignment_metrics,
                                       combine_transforms,
                                       create_preprocessing_matrix,
                                       extract_complete_transformation,
                                       morph_open, register_combination)
    from .dataframe_utils import METADATA_COLUMNS, get_marker_columns
    from .registeration import Register, TileMap
    from .stardist import StarDist
    from .Worker import Worker

_LAZY_IMPORTS = {
    "MemoryEfficientImageCache": ("core.canvas", "MemoryEfficientImageCache"),
    "ImageStorage": ("core.canvas", "ImageStorage"),
    "ImageWrapper": ("core.canvas", "ImageWrapper"),
    "ReferenceGraphicsView": ("core.canvas", "ReferenceGraphicsView"),
    "ImageGraphicsView": ("core.canvas", "ImageGraphicsView"),
    "MetaData": ("core.canvas", "MetaData"),
    "CellIntensity": ("core.cell_intensity", "CellIntensity"),
    "CellLayerAligner": ("core.cell_layer_alignment", "CellLayerAligner"),
    "calculate_alignment_metrics": ("core.cell_layer_alignment", "calculate_alignment_metrics"),
    "register_combination": ("core.cell_layer_alignment", "register_combination"),
    "morph_open": ("core.cell_layer_alignment", "morph_open"),
    "create_preprocessing_matrix": ("core.cell_layer_alignment", "create_preprocessing_matrix"),
    "combine_transforms": ("core.cell_layer_alignment", "combine_transforms"),
    "extract_complete_transformation": ("core.cell_layer_alignment", "extract_complete_transformation"),
    "Register": ("core.registeration", "Register"),
    "TileMap": ("core.registeration", "TileMap"),
    "Worker": ("core.Worker", "Worker"),
    "StarDist": ("core.stardist", "StarDist"),
    "METADATA_COLUMNS": ("core.dataframe_utils", "METADATA_COLUMNS"),
    "get_marker_columns": ("core.dataframe_utils", "get_marker_columns"),
}

__all__ = [
    "MemoryEfficientImageCache",
    "ImageStorage",
    "ImageWrapper",
    "ReferenceGraphicsView",
    "ImageGraphicsView",
    "MetaData",
    "CellIntensity",
    "CellLayerAligner",
    "calculate_alignment_metrics",
    "register_combination",
    "morph_open",
    "create_preprocessing_matrix",
    "combine_transforms",
    "extract_complete_transformation",
    "Register",
    "TileMap",
    "Worker",
    "StarDist",
    "METADATA_COLUMNS",
    "get_marker_columns",
]


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name, package="core")
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(__all__) + list(_LAZY_IMPORTS.keys()))
