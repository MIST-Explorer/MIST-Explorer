import os
import platform

import cv2 as cv
import numpy as np
import tensorflow as tf
from csbdeep.utils import normalize
from matplotlib import colormaps
from PIL import Image
from pyclesperanto_prototype import dilate_labels
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog
from stardist.models import StarDist2D

from core import ImageWrapper

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class StarDist(QThread):
    stardist_done = pyqtSignal(ImageWrapper, bool, str)
    # sendGrayScale = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int, str)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.protein_channels = None
        self.np_image = None
        self.params = {
            "channel": "Channel 2",
            "model": "2D_versatile_fluo",
            "percentile_low": 3,
            "percentile_high": 99.80,
            "prob_threshold": 0.48,
            "nms_threshold": 0.3,
            "n_tiles": 0,
            "radius": 5,
        }
        self.aligned = False

    def load_cell_image(self, arr):
        self.cell_image = arr
        self.aligned = True

    def run_stardist(self):
        if self.protein_channels is None and self.np_image is None:
            self.error_signal.emit("please load image first")  # emit error message
            return
        elif self.protein_channels and self.np_image:
            self.error_signal.emit(
                "unknown error, canvas has both single channel image and multi-channel image initiated"
            )  # emit error message
            return

        system = platform.system()
        print("system: ", system)
        print("tensorflow version: ", tf.__version__)
        gpu = len(tf.config.list_physical_devices("GPU")) > 0
        if gpu:
            device_name = tf.test.gpu_device_name()
            print("gpu name: ", device_name)
        else:
            device_name = "/CPU:0"

        with tf.device(device_name):
            try:
                self.run()
            except Exception as e:
                self._critical_error(f"StarDist Error: {str(e)}")
                return
            self.finished.connect(self.quit)
            self.finished.connect(self.deleteLater)

        print("here")

    def __get_cell_image(self):
        if self.aligned:
            return self.cell_image
        elif self.protein_channels is None and self.np_image:
            return self.np_image
        elif self.protein_channels and self.np_image is None:
            return self.protein_channels[self.params["channel"]].data

    def _critical_error(self, message):
        self.error_signal.emit(message)
        self.progress.emit(100, "Error")
        self.terminate()

    def run(self):
        cell_image = self.__get_cell_image()
        if cell_image is None:
            self._fatal_error_message("No image to process")
            return
        assert isinstance(cell_image, np.ndarray), "cell_image must be a numpy array"

        if cell_image is None:
            self._critical_error("No cell image available for processing")
            return

        # adjusted = cv.convertScaleAbs(cell_image, alpha=(255.0/65535.0))

        # alpha = 5 # Contrast control
        # beta = 15 # Brightness control
        # adjusted = cv.convertScaleAbs(adjusted, alpha=alpha, beta=beta)
        # cv.imshow('Image Window',adjusted)

        # cv.waitKey(0)

        # cv.destroyAllWindows()

        self.progress.emit(0, "Starting StarDist")
        model = StarDist2D.from_pretrained(str(self.params["model"]))
        assert model is not None, "Failed to load model"
        self.progress.emit(25, "Training model")

        print("here2")
        guess_tiles = self.params["n_tiles"]
        if guess_tiles == 0:
            guess_tiles = model._guess_n_tiles(cell_image)
            # total_tiles = int(guess_tiles[0] * guess_tiles[1])
            # self.setNumberTiles(n_tiles)
            stardist_labels, _ = model.predict_instances(
                normalize(
                    cell_image,
                    self.params["percentile_low"],
                    self.params["percentile_high"],
                ),
                prob_thresh=self.params["prob_threshold"],
                nms_thresh=self.params["nms_threshold"],
                n_tiles=guess_tiles,
            )  # type: ignore

        else:

            stardist_labels, _ = model.predict_instances(
                normalize(
                    cell_image,
                    self.params["percentile_low"],
                    self.params["percentile_high"],
                ),
                prob_thresh=self.params["prob_threshold"],
                nms_thresh=self.params["nms_threshold"],
                n_tiles=(self.params["n_tiles"], (self.params["n_tiles"])),
            )  # type: ignore

        # dilate
        print("here3")
        radius = self.params["radius"]
        self.progress.emit(95, "Dilating")
        # If error is platform not found, ask user to install run "sudo apt install pocl-opencl-icd"
        try:
            self.stardist_labels_grayscale = np.array(
                dilate_labels(stardist_labels, radius=radius), dtype=np.uint16
            )
        except Exception as e:
            self._fatal_error_message(
                f"Error during dilation: {e}. You may need to install pocl-opencl-icd(wsl2 users)."
            )
            return
        print("here 4")
        self.progress.emit(100, "Stardist Done")
        stardist_result = ImageWrapper(self.stardist_labels_grayscale, name="Channel 1")
        self.stardist_done.emit(
            stardist_result, True, "StarDist Labels"
        )  # emit signal with result, saves to sidebar

    def cancel(self):
        self.terminate()

    def save_image(self):
        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)"
        )
        if not self.stardist_labels_grayscale is None:
            Image.fromarray(self.stardist_labels_grayscale).save(file_name)
        else:
            self.error_signal.emit("Cannot save. No stardist labels available")

    # @pyqtSlot(int)
    # def updateProgress(self, num):
    #     self.progress.emit(num, f"Generating Tile {num}")

    # only uint8
    # @pyqtSlot(ImageWrapper)
    # def on_stardist_completed(self, stardist_result):
    #     self.stardistDone.emit(stardist_result)

    def change_cmap(self):
        pass

    def generate_lut(self, cmap: str):
        label_range = np.linspace(0, 1, 256)
        return np.uint8(colormaps[cmap](label_range)[:, 2::-1] * 256).reshape(256, 1, 3)

    def label2rgb(self, labels, lut):
        return cv.LUT(cv.merge((labels, labels, labels)), lut)

    def update_channels(self, protein_channels, _):
        self.np_image = None
        self.protein_channels = protein_channels

    def set_protein_image(self, protein_channels):
        self.protein_channels = protein_channels
        self.np_image = None

    def set_image_to_process(self, np_image):
        self.protein_channels = None
        self.np_image = np_image

    def set_channel(self, channel):
        self.params["channel"] = channel

    def set_model(self, model):
        self.params["model"] = model

    def set_percentile_low(self, value):
        self.params["percentile_low"] = value

    def set_percentile_high(self, value):
        self.params["percentile_high"] = value

    def set_prob_thresh(self, value):
        self.params["prob_threshold"] = value

    def set_number_tiles(self, value):
        self.params["n_tiles"] = value

    def set_dilation_radius(self, value):
        self.params["radius"] = value

    def set_nms_thresh(self, value):
        self.params["nms_threshold"] = value

    def set_num_tiles(self, value):
        self.params["n_tiles"] = value

    def set_dialation_radisu(self, value):
        self.params["radius"] = value

    def _fatal_error_message(self, msg):
        self.error_signal.emit(msg)
        self.progress.emit(100, "")
