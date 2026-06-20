import logging
import re

import numpy as np
import pandas as pd
import scanpy as sc

# Setup logger for this module
logger = logging.getLogger(__name__)


class DataModel:
    def __init__(self, data=None, normalization="none"):
        if data is None:
            raise ValueError("No data provided")

        # Store the immutable raw data
        self.raw_df = clean_panel_df(data)
        self.normalization_method = normalization

        # Define available features from the raw data immediately
        exclude_cols = ["Cell_ID", "Global_X", "Global_Y"]
        self.all_features = [
            col
            for col in self.raw_df.columns
            if col not in exclude_cols and not col.startswith("N/A")
        ]

        # Default: Use all features
        self.selected_features = self.all_features.copy()

        self.thresholds = {feature: 1.0 for feature in self.all_features}
        self.threshold_mode = "pre"
        self.adata = None
        self.adata_clean = None
        self.normalized_pre_threshold = None
        self.raw_pre_threshold = None
        self.hvg_rankings = None

        # Initial Build
        try:
            self.reprocess_data(self.normalization_method, self.selected_features)
        except Exception as e:
            logger.error(f"Error during initial data processing: {e}")

    def get_all_features(self):
        """Returns list of all available columns from raw dataframe"""
        return self.all_features

    def reprocess_data(self, normalization, features):
        logger.info("=== START reprocess_data ===")
        logger.info("Step 1: Filter columns")
        cols_to_use = (
            ["Cell_ID"] + features if "Cell_ID" in self.raw_df.columns else features
        )
        subset_df = self.raw_df[cols_to_use].copy()

        logger.info("Step 2: Remove NaNs")
        mask = ~subset_df.isna().any(axis=1)
        clean_df = subset_df.loc[mask].reset_index(drop=True)

        logger.info(f"Step 3: Create data array - shape: {clean_df.shape}")

        self.normalization_method = normalization
        self.selected_features = features

        # 1. Filter Raw DF by Selected Features
        # Always include metadata like Cell ID
        cols_to_use = (
            ["Cell_ID"] + features if "Cell_ID" in self.raw_df.columns else features
        )
        subset_df = self.raw_df[cols_to_use].copy()

        # 2. Remove NaNs
        mask = ~subset_df.isna().any(axis=1)
        clean_df = subset_df.loc[mask].reset_index(drop=True)

        # 3. Create AnnData
        if "Cell_ID" in clean_df.columns:
            labels = clean_df["Cell_ID"].values
            data_values = clean_df[features].values.astype(np.float64)
        else:
            labels = np.arange(1, len(clean_df) + 1)
            data_values = clean_df.values.astype(np.float64)

        # Store raw pre-threshold values
        self.raw_pre_threshold = data_values.copy()

        # 6. Apply Customizable Thresholds (Pre-normalization)
        if self.threshold_mode == "pre":
            for j, feature in enumerate(features):
                thresh = self.thresholds.get(feature, 1.0)
                data_values[:, j][data_values[:, j] < thresh] = 0.0

        self.adata = sc.AnnData(X=data_values)
        self.adata.obs["cell_id"] = labels
        self.adata.var_names = features

        # 4. Standard Filtering (No longer filtering cells in-place to preserve all-zero cells for UI mapping)
        xmin = float(self.adata.X.min())
        xmax = float(self.adata.X.max())
        logger.info(f"Data range before counts cast: min={xmin}, max={xmax}")

        if xmin < 0:
            logger.warning(
                "Counts contain negatives; 'counts' layer should be non-negative."
            )
        if xmax <= 65535 and xmin >= 0:
            self.adata.layers["counts"] = self.adata.X.astype(np.uint16).copy()
        else:
            # safest: preserve exact integers for HVG
            self.adata.layers["counts"] = self.adata.X.astype(np.uint32).copy()
        raw_adata = self.adata.copy()
        sc.pp.log1p(raw_adata)
        self.adata.raw = raw_adata

        # 5. Apply Selected Normalization
        self._apply_normalization()

        # Store post-normalization, pre-threshold values
        self.normalized_pre_threshold = self.adata.X.copy()

        # 6. Apply Customizable Thresholds (Post-normalization)
        if self.threshold_mode == "post":
            for j, feature in enumerate(features):
                thresh = self.thresholds.get(feature, 1.0)
                self.adata.X[:, j][self.adata.X[:, j] < thresh] = 0.0

        # 7. Identify cells that are all-zeros post-threshold
        all_zero_mask = np.all(self.adata.X == 0.0, axis=1)
        self.adata.obs["is_filtered"] = all_zero_mask

        # Subset to non-zero cells for PCA, UMAP, and Clustering
        self.adata_clean = self.adata[~all_zero_mask].copy()

        # 8. Recompute PCA
        self._compute_hvg_and_pca()

        logger.info(f"Data reprocessed. Shape: {self.adata.shape}, Clean: {self.adata_clean.shape}")

    def get_normalization_methods(self):
        """Returns the list of available normalization methods"""
        return ["RC", "lognorm", "znorm", "none"]

    def _apply_normalization(self):
        """Applies the specific normalization logic defined in original code"""
        assert isinstance(self.adata, sc.AnnData), "adata is not an AnnData instance"
        if self.normalization_method == "RC":
            sc.pp.scale(self.adata, max_value=10)
        elif self.normalization_method == "lognorm":
            logger.info("Applying log Normalization")
            sc.pp.log1p(self.adata)
            sc.pp.scale(self.adata, max_value=10)
            # adata = self.adata
            # adata.layers["counts"] = adata.X.copy()
            # adata.layers["cpm"] = adata.X.copy()
            # sc.pp.log1p(adata)
            # adata.layers["data"] = adata.X.copy()
            # sc.pp.highly_variable_genes(
            #     adata,
            #     flavor="seurat_v3",
            #     layer="counts",
            #     n_top_genes=2500,
            # )
            # sc.pp.regress_out(adata, ["pct_counts_mt"])
        elif self.normalization_method == "none":
            logger.info("No normalization applied.")
        elif self.normalization_method == "znorm":
            X = self.adata.X.astype(float)
            means = X.mean(axis=0)
            stds = X.std(axis=0)
            self.adata.X = (X - means) / stds
            # sc.pp.scale(self.adata)
            logger.info("Z-normalization applied.")

    def _compute_hvg_and_pca(self):
        assert isinstance(self.adata_clean, sc.AnnData)

        # PCA on all features, safely capped at min(n_obs, n_vars) to avoid ValueError
        n_comps = min(self.adata_clean.n_obs, self.adata_clean.n_vars)
        if n_comps > 0:
            sc.pp.pca(self.adata_clean, n_comps=n_comps, random_state=0, svd_solver="auto")

        # Mimic "all variables are highly variable"
        self.adata_clean.var["highly_variable"] = True

        # Optional: fake dispersion fields (some UI code expects them)
        self.adata_clean.var["means"] = self.adata_clean.X.mean(axis=0)
        self.adata_clean.var["variances"] = self.adata_clean.X.var(axis=0)
        # based on varriances
        self.adata_clean.var["highly_variable_rank"] = (
            np.argsort(-self.adata_clean.var["variances"]).astype(np.int32) + 1
        )  # 1-based ranking

        self.hvg_rankings = self.adata_clean.var["highly_variable_rank"].copy()

    # --- Public Accessors for UI ---
    def get_num_cells(self):
        assert self.adata is not None, "adata is not initialized"
        return self.adata.n_obs

    def get_num_features(self):
        assert self.adata is not None, "adata is not initialized"
        return self.adata.n_vars

    def get_max_pcs(self):
        """Returns the maximum number of PCs available for UMAP selection"""
        if self.adata_clean is not None and "PCs" in self.adata_clean.varm:
            return self.adata_clean.varm["PCs"].shape[1]
        return 0

    def get_pca_variance_ratio(self):
        """Returns the variance ratio array for the 'Variance Threshold' UI logic"""
        assert self.adata_clean is not None, "adata_clean is not initialized"
        return self.adata_clean.uns["pca"]["variance_ratio"]

    # --- Heavy Calculation Methods (To be called by Worker Thread) ---

    def _merge_clean_back_to_full(self):
        if self.adata_clean is None or self.adata is None:
            return

        import pandas as pd

        # 1. Merge Leiden clustering labels
        if "leiden" in self.adata_clean.obs:
            clean_cats = list(self.adata_clean.obs["leiden"].cat.categories)
            has_filtered = np.any(self.adata.obs["is_filtered"])

            if has_filtered and "Filtered" not in clean_cats:
                clean_cats.append("Filtered")

            # Create array of labels filled with "Filtered"
            full_labels = np.full(self.adata.n_obs, "Filtered", dtype=object)
            clean_indices = np.where(~self.adata.obs["is_filtered"])[0]

            # Place clean labels
            full_labels[clean_indices] = self.adata_clean.obs["leiden"].astype(str).values

            # Assign category mapping to self.adata.obs
            self.adata.obs["leiden"] = pd.Categorical(full_labels, categories=clean_cats)

            # 2. Merge Leiden cluster colors
            if "leiden_colors" in self.adata_clean.uns:
                clean_colors = list(self.adata_clean.uns["leiden_colors"])
                if has_filtered and len(clean_colors) < len(clean_cats):
                    # Dedicate a gray color for filtered cells
                    clean_colors.append("#555555")
                self.adata.uns["leiden_colors"] = np.array(clean_colors)

        # 3. Merge UMAP coordinates
        if "X_umap" in self.adata_clean.obsm:
            full_umap = np.full((self.adata.n_obs, 2), np.nan)
            clean_indices = np.where(~self.adata.obs["is_filtered"])[0]
            full_umap[clean_indices] = self.adata_clean.obsm["X_umap"]
            self.adata.obsm["X_umap"] = full_umap

    def run_clustering_only(self, resolution):
        """Runs only Leiden clustering (fast update for slider)"""
        logger.info(f"Re-running clustering at resolution {resolution}")
        assert isinstance(self.adata_clean, sc.AnnData), "adata_clean is not initialized"
        key = "leiden"
        sc.tl.leiden(
            self.adata_clean,
            key_added=key,
            resolution=resolution,
            flavor="igraph",
            n_iterations=2,
            random_state=0,
            directed=False,
        )
        self._merge_clean_back_to_full()
        return self.adata, key

    def run_umap_pipeline(
        self, n_neighbors, min_dist, n_components, random_state, resolution
    ):
        """
        Runs Neighbors -> UMAP -> Leiden.
        Replicates run_umap logic but is designed to be atomic for the thread.
        """
        logger.info(
            f"Running UMAP Pipeline: neighbors={n_neighbors}, dist={min_dist}, pcs={n_components}"
        )
        assert isinstance(self.adata_clean, sc.AnnData), "adata_clean is not initialized"

        # 1. Compute Neighbors
        sc.pp.neighbors(
            self.adata_clean,
            random_state=random_state,
            n_neighbors=n_neighbors,
            n_pcs=n_components,
            metric="cosine",
            use_rep="X_pca",
        )
        logger.info("Neighbors computed.")
        # 2. Run UMAP
        sc.tl.umap(
            self.adata_clean,
            min_dist=min_dist,
            random_state=random_state,
            maxiter=None,
        )
        logger.info("UMAP embedding computed.")
        # 3. Initial Clustering
        adata, key = self.run_clustering_only(resolution)
        logger.info("Leiden clustering computed.")

        return adata, key


def clean_panel_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # strip whitespace
    df.columns = [c.strip() for c in df.columns]

    # normalize names to avoid "IL 10" / "Granzyme B" weirdness in downstream lookups
    # (optional, but makes life easier)
    def norm(c: str) -> str:
        c = c.strip()
        c = re.sub(r"\s+", "_", c)  # spaces -> _
        c = c.replace(".", "_")  # dots -> _
        return c

    df.rename(columns={c: norm(c) for c in df.columns}, inplace=True)

    # ensure uniqueness if duplicates exist
    if df.columns.duplicated().any():
        counts = {}
        new_cols = []
        for c in df.columns:
            if c not in counts:
                counts[c] = 0
                new_cols.append(c)
            else:
                counts[c] += 1
                new_cols.append(f"{c}__{counts[c]}")
        df.columns = new_cols

    # coerce everything except Global coords / Cell ID
    meta = {"Cell_ID", "Global_X", "Global_Y"}
    feat_cols = [c for c in df.columns if c not in meta]

    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df
