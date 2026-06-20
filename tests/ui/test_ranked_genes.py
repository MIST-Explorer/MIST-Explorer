from ui.analysis.graphing import UMAPPlot

def test_ranked_genes_renaming(mock_adata, qtbot):
    """
    Verify that renaming clusters updates the dropdown text but preserves
    the original internal data keys used for AnnData lookups.
    """
    view = UMAPPlot.RankedGenesView()
    qtbot.addWidget(view)
    
    # 1. Set Data
    cluster_key = "leiden"
    view.set_data(mock_adata, cluster_key)
    
    # Check initial state
    assert view.cluster_combo.count() == 3
    assert view.cluster_combo.itemText(0) == "0"
    assert view.cluster_combo.itemData(0) == "0"
    
    # 2. Rename Cluster 0 -> "Tumor", Cluster 2 -> "Stroma"
    renames = {0: "Tumor", 2: "Stroma"}
    view.update_cluster_names(renames)
    
    # 3. Verify Updates
    # Item 0
    assert view.cluster_combo.itemText(0) == "Tumor"
    assert view.cluster_combo.itemData(0) == "0" # Should preserve original key
    
    # Item 1 (Unchanged)
    assert view.cluster_combo.itemText(1) == "1"
    assert view.cluster_combo.itemData(1) == "1"
    
    # Item 2
    assert view.cluster_combo.itemText(2) == "Stroma"
    assert view.cluster_combo.itemData(2) == "2"

def test_on_update_uses_original_key(mock_adata, qtbot):
    """
    Ensure that when selection changes, the code uses the itemData (original key)
    rather than the itemText (display name) to query scanpy.
    """
    view = UMAPPlot.RankedGenesView()
    qtbot.addWidget(view)
    view.set_data(mock_adata, "leiden")
    
    # Rename
    renames = {0: "Tumor"}
    view.update_cluster_names(renames)
    
    # Select the renamed item (Index 0, "Tumor")
    view.cluster_combo.setCurrentIndex(0)
    
    # We can't easily mock scanpy calls inside the class without heavy mocking,
    # but we can verify what `currentData()` returns, which is what the fix relies on.
    assert view.cluster_combo.currentText() == "Tumor"
    assert view.cluster_combo.currentData() == "0"
