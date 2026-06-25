.. MIST-Explorer documentation master file, created by
   sphinx-quickstart on Thu Mar 27 14:55:41 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MIST-Explorer
===========================

MIST-Explorer is a tool for researchers and scientists working in single-cell proteomics. It lets you load, process, visualize, and analyze protein distribution at the single-cell level — from raw multi-cycle microscopy images through to statistical and spatial analysis of individual cells.

Key Features
------------
- **Extract Tab**: Crop, rotate, flip, register and align images. Segment cells with StarDist or CellProfiler, then quantify per-cell protein intensities.
- **View Tab**: Visualize multi-channel protein expression data as layered overlays with independent opacity, contrast, and color tinting per channel. Export to PNG or multi-channel TIFF.
- **Analysis Tab**: Draw regions of interest (rectangle, circle, polygon) and generate box plots, spatial autocorrelation heatmaps, spatial heatmaps, pie charts, histograms, and UMAP projections.

Requirements
------------
- **Operating System**: Windows, macOS (tested)
- **Minimum Hardware**: 8 GB RAM, 2 GHz processor

.. toctree::
   :maxdepth: 2

   install
   tutorial
   extract
   view
   analysis

