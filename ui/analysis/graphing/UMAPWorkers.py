import logging
import traceback

import matplotlib
from PyQt6.QtCore import QThread, pyqtSignal
from ui.analysis.graphing.UMAPDataModel import DataModel

matplotlib.use("QtAgg")

logger = logging.getLogger("MIST_UMAP")


class AnalysisWorker(QThread):
    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)

    def __init__(
        self,
        model: DataModel,
        n_neighbors,
        min_dist,
        n_components,
        random_state,
        resolution,
    ):
        super().__init__()
        self.model = model
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.random_state = random_state
        self.resolution = resolution
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            logger.info("Worker: Starting UMAP pipeline calculation...")
            adata, key = self.model.run_umap_pipeline(
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                n_components=self.n_components,
                random_state=self.random_state,
                resolution=self.resolution,
            )
            if self._cancelled:
                logger.info("Worker: Cancelled, discarding results.")
                return
            logger.info("Worker: Pipeline calculation successful.")
            self.finished.emit(adata, key)

        except Exception as e:
            if self._cancelled:
                logger.info("Worker: Cancelled during exception.")
                return
            logger.exception("Worker failed during UMAP pipeline.")
            tb = traceback.format_exc()
            self.error.emit(f"Worker Error: {str(e)}\n\n{tb}")


class ClusteringWorker(QThread):
    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)

    def __init__(self, model: DataModel, resolution):
        super().__init__()
        self.model = model
        self.resolution = resolution
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self.model.adata is None:
                raise ValueError("No data loaded in model.")

            logger.info(
                f"Worker: Starting Re-clustering (Resolution={self.resolution})..."
            )

            adata, key = self.model.run_clustering_only(self.resolution)

            if self._cancelled:
                logger.info("Worker: Cancelled, discarding results.")
                return

            logger.info("Worker: Re-clustering successful.")
            self.finished.emit(self.model.adata, key)

        except Exception as e:
            if self._cancelled:
                logger.info("Worker: Cancelled during exception.")
                return
            logger.exception("Worker failed during clustering.")
            tb = traceback.format_exc()
            self.error.emit(f"Clustering Error: {str(e)}\n\n{tb}")
