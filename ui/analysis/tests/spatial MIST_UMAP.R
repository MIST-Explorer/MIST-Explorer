rm(list=ls())



library(dplyr)
library(Seurat)
library(SeuratObject)
library(patchwork)
library(openxlsx)
library(ggplot2)


library(umap)

# Reading the excel file into a data frame
input_path = "/Users/clark/Desktop/wang/protein_visualization_app/ui/analysis/graphing/Grouped Cells Biopsy Data.xlsx"
data_origin = data.frame(read.xlsx(input_path, sheet='Sheet1'))
#baseline = data.frame(read.xlsx(input_path, sheet='0 cell'))

# Subset the data to only include protein expression values
data_1 = data_origin[,4:length(data_origin)]
#baseline_1 = baseline[,2:length(baseline)]
#print("Data Read")

#baseline_std_mean = colMeans(baseline_1, dims=1) + 2* sapply(baseline_1, sd) 
## Clustering(UMAP) ----

#data_background_removal = sweep(data_1, 2, baseline_std_mean)

"
Clustering groups the cells together dependent on the protein expression values of each cell.
This allows us to visualize which cells can be identified by a particular cell type or 
how similar the functionality of the cell types are (data-dependent)
"


data_2 = t(data.matrix(data_1))

colnames(data_2) = paste0("Cell", 1:length(data_2[1,]))
data_3 = data_2 # we do this to create variable features by removing background noise.
#seq data are normally sparse data with a lot of zeroes
data_3[data_3 < 1] = 0
#data_3 = log2(data_3+1) # don't normalize.
# Initialize Seurat Object
data = CreateSeuratObject(data_3)

# Visualize the data values
# data@assays$RNA@data  # <- run this code if needed

# Visualize Features
#VlnPlot(data, features = c("nFeature_RNA", "nCount_RNA"), ncol=2)
#FeatureScatter(data, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")

# QC for outliers and normalization
#data = subset(data, subset = nCount_RNA<600000)  # need to change nCount_RNA to appropriate value

data = NormalizeData(data, normalization.method = "LogNormalize")  # can change method between "RC" and "LogNormalize"


# Find Highly Variable Features:
# Pick out proteins that exhibit high cell-to-cell variation
# i.e. they are highly expressed in some cells, and lowly expressed in others
# nfeatures = the number of proteins to choose
allData = FindVariableFeatures(data, selection.method = "vst", nfeatures=ncol(data)) #nfeatures=ncol(data)
#allData

# Identify the 10 most highly variable genes
top20 <- head(VariableFeatures(allData), 6)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(allData)
plot2 <- LabelPoints(plot = plot1, points = top20, repel = TRUE)
plot2

# Scaling the Data and running dimensional reduction (PCA)
allProteins = rownames(allData)
data = ScaleData(allData, features=allProteins)
data = RunPCA(data, features=VariableFeatures(allData))

#pbmc_jack <- JackStraw(data)  
#pbmc_score <- ScoreJackStraw(pbmc_jack, dims = 1:15)
#ElbowPlot(pbmc_score) 


# Run UMAP clustering
data_nn = FindNeighbors(data, dims=1:5) # dim = number of PCA dimensions to include
data_cluster = FindClusters(data_nn, resolution=0.1) # resolution = how large the community is

data_umap = RunUMAP(data_cluster,  n.neighbors=15L, min.dist=0.1,dims=1:5)
print("Clustering Completed")
DimPlot(data_umap, reduction='umap', label=TRUE, pt.size=1, label.color = "black",
        group.by=c("seurat_clusters"), ncol=1) 


# annotate cell types - 4 cell types

## Heatmap ----

"
Heatmaps visualize the top x number of proteins per group (i.e. seurat_clusters, 
conditions, samples, etc.) by representing the protein's logFC (log fold change; 
main weight parameter) value on the heatmap. Parameters can be change to find
downregulated proteins, number of proteins per group, type of group, different
weight parameters)

Colors: yellow = highly expressed; purple = poorly expressed
"

# Find top 10 markers of the seurat_clusters labeled groups using logFC
heat_markers = FindAllMarkers(data_umap, only.pos=TRUE)
top = heat_markers %>% group_by(cluster) %>% top_n(n=20, wt=avg_log2FC)
DoHeatmap(data_umap, features=unique(top$gene)) + NoLegend() + theme(text = element_text(size = 7))




## FeaturePlot ----

"
A plot that shows the location of the expression of a single protein for all proteins.
This plot is useful for determining which protein belong within which cluster and helping 
with distinguishing cell types per cluster. The input data must be subtracted by the
background values to ensure no excessive expression representation in the plots. There must
be 0's within the data to show zero expression, otherwise all plots will be over saturated.
"

# Features= can be changed to a list of specific proteins for convenience
# This function will take a very long time to load if there are >50 proteins chosen
# The resulting image must also be saved with large dimensions (i.e. 3000x6000 for 30 proteins)
FeaturePlot(data_umap, features='Nestin')
FeaturePlot(data_umap, features='Vimentin')
FeaturePlot(data_umap, features='Nestin')
FeaturePlot(data_umap, features='Calbindin')
# FeaturePlot(data_umap, features='Aqp2')




