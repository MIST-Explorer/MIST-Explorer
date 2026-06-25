Tutorial
========

This tutorial walks through the end-to-end MIST-Explorer workflow — from loading raw images to generating visualizations and statistical analysis of protein expression data.

Getting Started
---------------

1. **Installation**: Ensure the application is installed. See the :doc:`install` page for instructions.

2. **Loading Data**: Launch the application and load your microscopy images via **File → Open** or dropping them onto the canvas/sidebar.

Extracting and Processing Your Images
--------------------------------------

The Extract tab prepares raw multi-cycle images for visualization and analysis.

1. **Crop and Rotate**: Use the Transform sub-tab to crop to your region of interest and correct image orientation. You can also flip images if needed.

2. **Manual Registeration**: Before running automatic registration, use the manual alignment tool in the Transform sub-tab to roughly position your images. This step is optional but strongly recommended.

3. **Automatic Registration**: Run the registration pipeline to register images automatically. You will have a chance to manually register the output of the registeration pipeline in the post registeration modal.

4. **Brightfield Alignment**: Use the Alignment sub-tab if you need to align a tiff to reference brightfield, applies same transformation across all channels.

5. **Cell Segmentation**: In the Segmentation sub-tab, run StarDist (or CellProfiler) to identify individual cells.

6. **Quantification**: Run Cell Intensity Extraction in the Quantification sub-tab to produce a per-cell CSV of protein expression values.

For detailed information on all options and parameters, see the :doc:`extract` page.

Visualizing Protein Expression
-------------------------------

After saving cell segmentation mask and cell data, use the View tab to explore protein expression patterns.

1. **Loading Data**: Load your segmentation image and the protein expression CSV produced by Quantification.

2. **Adding Layers**: Click **Add Layer** to add protein channels of interest. Each channel appears as an independent layer.

3. **Adjusting Display**: Use the per-layer controls to set opacity, contrast, and tint color. Assign complementary colors to co-localizing proteins.

4. **Region Selection**: Draw regions of interest (rectangle, circle, or polygon) directly on the canvas to prepare for analysis.

5. **Exporting**: Export the current view as a PNG or multi-channel TIFF using the Export controls.

For comprehensive information on visualization features, refer to the :doc:`view` page.

Analyzing Regions of Interest
------------------------------

Once you have drawn regions of interest, switch to the Analysis tab to explore the data.

1. **Select Proteins**: Use the multi-select dropdown to choose which proteins to include. Click **Apply**.

2. **Explore Visualizations**: Step through the available plot types — box plots, spatial autocorrelation heatmaps, spatial heatmaps, pie charts, histograms, and UMAP.

3. **Compare Regions**: Use the **Back** and **Next** buttons to navigate between your drawn ROIs and compare expression patterns.

4. **Full Image Analysis**: For a global view, use the full image analysis mode to run UMAP.

5. **Save Results**: Use **Save Plot** to export visualizations for reporting or publication.

For in-depth information on analysis capabilities, see the :doc:`analysis` page.

Video Tutorials
---------------

The following videos walk through common workflows in MIST-Explorer.

How to Manually Register Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Covers the full registration workflow: using manual alignment in the Transform sub-tab to remove gross offsets, running automatic registration, and fine-tuning the result in the post-registration modal.

.. raw:: html

   <!-- TODO: Replace with YouTube embed once video is published
   <iframe width="560" height="315"
     src="https://www.youtube.com/embed/VIDEO_ID_1"
     title="How to Manually Register Images"
     frameborder="0"
     allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
     allowfullscreen>
   </iframe>
   -->

   <p><em>Video coming soon.</em></p>

How to Visualize Generated Cell Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Covers loading the Quantification CSV output, adding protein layers in the View tab, adjusting contrast and tint colors, and using complementary colors for co-localization.

.. raw:: html

   <!-- TODO: Replace with YouTube embed once video is published
   <iframe width="560" height="315"
     src="https://www.youtube.com/embed/VIDEO_ID_2"
     title="How to Visualize Generated Cell Data"
     frameborder="0"
     allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
     allowfullscreen>
   </iframe>
   -->

   <p><em>Video coming soon.</em></p>

How to Run UMAP
^^^^^^^^^^^^^^^^

Covers launching the UMAP visualizer, interpreting the 2D projection, and identifying cell clusters based on protein expression profiles.

.. raw:: html

   <!-- TODO: Replace with YouTube embed once video is published
   <iframe width="560" height="315"
     src="https://www.youtube.com/embed/VIDEO_ID_3"
     title="How to Run UMAP"
     frameborder="0"
     allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
     allowfullscreen>
   </iframe>
   -->

   <p><em>Video coming soon.</em></p>

End-to-End Workflow
^^^^^^^^^^^^^^^^^^^^

A full pipeline walkthrough: loading images → aligning and registering → segmenting cells → quantifying intensity → visualizing layers → drawing ROIs → analyzing expression patterns.

.. raw:: html

   <!-- TODO: Replace with YouTube embed once video is published
   <iframe width="560" height="315"
     src="https://www.youtube.com/embed/VIDEO_ID_4"
     title="End-to-End Workflow"
     frameborder="0"
     allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
     allowfullscreen>
   </iframe>
   -->

   <p><em>Video coming soon.</em></p>

Cell Segmentation with StarDist and CellProfiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Covers selecting the right channel for segmentation, when to use Gaussian blur, and how to tune the probability and overlap threshold parameters for different cell types and image qualities.

.. raw:: html

   <!-- TODO: Replace with YouTube embed once video is published
   <iframe width="560" height="315"
     src="https://www.youtube.com/embed/VIDEO_ID_5"
     title="Cell Segmentation with StarDist"
     frameborder="0"
     allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
     allowfullscreen>
   </iframe>
   -->

   <p><em>Video coming soon.</em></p>

Drawing ROIs and Comparing Regions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Covers using rectangle, circle, and polygon lasso tools to define regions of interest, navigating between multiple ROIs in the Analysis tab, and comparing box plots and heatmaps across regions.

.. raw:: html

   <!-- TODO: Replace with YouTube embed once video is published
   <iframe width="560" height="315"
     src="https://www.youtube.com/embed/VIDEO_ID_6"
     title="Drawing ROIs and Comparing Regions"
     frameborder="0"
     allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
     allowfullscreen>
   </iframe>
   -->

   <p><em>Video coming soon.</em></p>

Exporting Results
^^^^^^^^^^^^^^^^^^

Covers exporting the canvas view as PNG or multi-channel TIFF from the View tab, and saving individual analysis plots from the Analysis tab.

.. raw:: html

   <!-- TODO: Replace with YouTube embed once video is published
   <iframe width="560" height="315"
     src="https://www.youtube.com/embed/VIDEO_ID_7"
     title="Exporting Results"
     frameborder="0"
     allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
     allowfullscreen>
   </iframe>
   -->

   <p><em>Video coming soon.</em></p>
