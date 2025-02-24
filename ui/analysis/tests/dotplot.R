

"
A plot that shows the percent expression within all clusters.
This plot is useful for determining which protein belong within which cluster and helping 
with distinguishing cell types per cluster. The input data must be subtracted by the
background values to ensure no excessive expression representation in the plots. There must
be 0's within the data to show zero expression, otherwise all plots will be over saturated.
"
  
dot_markers <- FindAllMarkers(data_umap, only.pos=TRUE)

# pct.1 = percent of expression within its own cluster
# pct.2 = percent of expression within global clusters
top_dot <- dot_markers %>% group_by(cluster) %>% top_n(10, wt=pct.2)
DotPlot(data_umap, features=unique(top_dot$gene)) + RotatedAxis() + coord_flip() + theme(text = element_text(size = 7))

library(ggplot2)
library(Seurat)
library(dplyr)
library(RColorBrewer)

# Find top markers
dot_markers <- FindAllMarkers(data_umap, only.pos=TRUE)

# Select top markers for each cluster
top_dot <- dot_markers %>% group_by(cluster) %>% top_n(10, wt=pct.2)

# Create a DotPlot with a diverging color scale
dp <-DotPlot(data_umap, features=unique(top_dot$gene)) + 
  RotatedAxis() + 
  coord_flip() + 
  theme(text = element_text(size = 7)) +
  scale_color_distiller(palette = "RdYlBu", direction = 1) +  # Use a diverging color scale
  scale_size(range = c(1, 5))  # Adjust point sizes for better visibility

print(dp)