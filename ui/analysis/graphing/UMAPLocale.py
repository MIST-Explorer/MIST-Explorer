class Locale:
    """
    Central repository for all GUI text.
    Edit values here to change text across the entire application.
    """

    STRINGS = {
        # --- Main Window ---
        "APP_TITLE": "MIST Explorer",
        "HEADER": "UMAP Analysis Dashboard",
        "THEME_BTN": "Switch Theme",
        "EXPORT_BTN": "Export Cluster Labels",
        "BTN_EXPORT_VIEW": "Export to View Tab",
        "STATUS_READY": "Ready",
        "STATUS_BUSY": "Processing...",
        "STATUS_ERR": "Error",
        # --- Tabs ---
        "TAB_UMAP": "UMAP Visualization",
        "TAB_SEG": "Spatial Segmentation",
        "TAB_HEATMAP": "Protein Expression Heatmap",
        # --- Data Controls ---
        "GRP_DATA": "Data Preprocessing",
        "LBL_NORM": "Normalization Method:",
        "PLACEHOLDER_SEARCH": "Search proteins (e.g., CD4, CD8a)...",
        "BTN_SCAN": "Search",
        "BTN_ALL": "Select All",
        "BTN_CLR": "Deselect All",
        "GRP_PCA": "PCA Parameters",
        "LBL_COMPONENTS": "Principal Components [{:02d}]",
        "BTN_RECALC_PCA": "Recalculate PCA",
        "GRP_UMAP": "UMAP Settings",
        "LBL_NEIGHBORS": "Neighbors [{:03d}]",
        "LBL_MINDIST": "Minimum Distance [{:.2f}]",
        "GRP_CLUSTER": "Dynamic Clustering (Leiden)",
        "LBL_RES": "Resolution [{:.2f}]",
        "BTN_EXECUTE": "Run Analysis",
        "BTN_PROCESSING": "Calculating...",
        "BTN_CANCEL": "Cancel",
        # --- Views ---
        "PLOT_NO_SIGNAL": "No data to display",
        "PLOT_PROJECTION": "Color by: {}",
        "SEG_STATUS_OFF": "Status: Not Loaded",
        "SEG_CLUSTER_FILTER": "Filter by Cluster",
        "SEG_ACTIVE_COUNT": "Selected Clusters: {}",
        "HM_WAITING": "Select genes to generate heatmap...",
        "HM_VOID": "No cells in current selection",
        # --- Alerts ---
        "ALERT_SCAN_TITLE": "Search Results",
        "ALERT_SCAN_MSG": "Found {} matching genes.",
        "ALERT_ERR_TITLE": "Input Error",
        "ALERT_MIN_FEAT": "Minimum features required: 3",
        "ALERT_SUCCESS": "Analysis updated successfully.",
    }

    @classmethod
    def get(cls, key, *args):
        """
        Retrieves text. If args are provided, performs string formatting.
        Example: Locale.get("LBL_NEIGHBORS", 15) -> "NEIGHBORS [015]"
        """
        template = cls.STRINGS.get(key, f"MISSING_KEY:[{key}]")
        if args:
            try:
                return template.format(*args)
            except IndexError:
                return template
        return template
