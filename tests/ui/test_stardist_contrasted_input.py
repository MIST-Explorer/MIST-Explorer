import cv2 as cv
import numpy as np

from core.canvas import ImageWrapper
from core.image_utils import create_lut, scale_adjust
from core.stardist import StarDist


def test_resolve_segmentation_input_raw_mode():
    model = StarDist()
    wrapper = ImageWrapper(
        np.array([[10, 20], [30, 40]], dtype=np.uint16), name="Channel 1"
    )
    wrapper.contrast_min = 50
    wrapper.contrast_max = 150
    model.update_channels({"Channel 1": wrapper}, False)
    model.set_channel("Channel 1")
    model.set_use_contrasted_image(False)

    resolved = model._resolve_segmentation_input()

    assert resolved is wrapper.data


def test_resolve_segmentation_input_contrasted_mode():
    model = StarDist()
    raw = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    wrapper = ImageWrapper(raw, name="Channel 1")
    wrapper.contrast_min = 64
    wrapper.contrast_max = 192
    model.update_channels({"Channel 1": wrapper}, False)
    model.set_channel("Channel 1")
    model.set_use_contrasted_image(True)

    resolved = model._resolve_segmentation_input()

    expected = (
        np.clip(cv.LUT(scale_adjust(raw), create_lut(64, 192)), 0, 254, dtype=np.uint8)
        .astype(np.uint16)
        * 257
    )
    assert resolved.dtype == np.uint16
    assert np.array_equal(resolved, expected)


def test_resolve_segmentation_input_contrasted_mode_falls_back_for_np_image():
    model = StarDist()
    raw = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    model.set_image_to_process(raw)
    model.set_use_contrasted_image(True)

    resolved = model._resolve_segmentation_input()

    assert resolved is raw


def test_resolve_segmentation_input_uses_selected_channel_contrast():
    model = StarDist()
    ch1 = ImageWrapper(np.array([[0, 32], [64, 96]], dtype=np.uint8), name="Channel 1")
    ch1.contrast_min = 0
    ch1.contrast_max = 64

    ch2_raw = np.array([[0, 128], [192, 255]], dtype=np.uint8)
    ch2 = ImageWrapper(ch2_raw, name="Channel 2")
    ch2.contrast_min = 128
    ch2.contrast_max = 255

    model.update_channels({"Channel 1": ch1, "Channel 2": ch2}, False)
    model.set_channel("Channel 2")
    model.set_use_contrasted_image(True)

    resolved = model._resolve_segmentation_input()

    expected = (
        np.clip(
            cv.LUT(scale_adjust(ch2_raw), create_lut(128, 255)), 0, 254, dtype=np.uint8
        ).astype(np.uint16)
        * 257
    )
    assert np.array_equal(resolved, expected)


def test_run_passes_scale_to_stardist_predictor(monkeypatch):
    model = StarDist()
    model.set_image_to_process(np.array([[1, 2], [3, 4]], dtype=np.uint16))
    model.set_scale(1.7)

    class _FakeConfig:
        axes = "YX"
        n_channel_in = 1

    class _FakeModel:
        config = _FakeConfig()

        def __init__(self):
            self.last_kwargs = None

        def _guess_n_tiles(self, _image):
            return (1, 1)

        def _predict_instances_generator(self, _img, **kwargs):
            self.last_kwargs = kwargs
            yield (np.array([[1, 0], [0, 2]], dtype=np.uint16),)

    fake_model = _FakeModel()
    model.model = fake_model
    model.current_model = str(model.params["model"])
    monkeypatch.setattr("core.stardist.dilate_labels", lambda labels, radius: labels)

    model.run()

    assert fake_model.last_kwargs["scale"] == 1.7


def test_run_cellprofiler_like_method_skips_stardist_model_load(monkeypatch):
    model = StarDist()
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:36, 20:36] = 1.0
    model.set_image_to_process(image)
    model.set_segmentation_method("CellProfiler-like")
    model.set_min_size(5)
    model.set_max_size(80)

    def _fail_model_load(*_args, **_kwargs):
        raise AssertionError("StarDist model should not be loaded")

    monkeypatch.setattr("core.stardist.StarDist2D.from_openvino", _fail_model_load)

    emitted = []
    model.stardist_done.connect(lambda wrapper, *_: emitted.append(wrapper))

    model.run()

    assert emitted
    assert emitted[0].data.dtype == np.int32
    assert int(emitted[0].data.max()) >= 1
