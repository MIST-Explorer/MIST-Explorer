import uuid
from types import SimpleNamespace

import numpy as np
from PyQt6.QtCore import QRect

import core.canvas as canvas_module
from core.canvas import ImageGraphicsView, ImageWrapper


def _make_channels(offset: int = 0):
    return {
        "Channel 1": ImageWrapper(
            np.array([[1, 2], [3, 4]], dtype=np.uint16) + offset, "Channel 1"
        ),
        "Channel 2": ImageWrapper(
            np.array([[10, 20], [30, 40]], dtype=np.uint16) + offset, "Channel 2"
        ),
    }


def _make_channels_with_stardist():
    labels = np.array([[0, 1], [2, 0]], dtype=np.uint16)
    return {
        "Channel 1": ImageWrapper(np.array([[1, 2], [3, 4]], dtype=np.uint16), "Channel 1"),
        "Channel 2": ImageWrapper(
            np.array([[10, 20], [30, 40]], dtype=np.uint16), "Channel 2"
        ),
        "Channel 3": ImageWrapper(labels, name="Segmentation", cmap="label_image"),
    }


def _make_view():
    controller = SimpleNamespace(model_stardist=SimpleNamespace(params={}))
    view = ImageGraphicsView(controller)
    view.set_pixmap = lambda *_args, **_kwargs: None
    view._schedule_caching_task = lambda *_args, **_kwargs: None
    return view


def _load_uuid_on_channel(view: ImageGraphicsView, image_uuid: uuid.UUID, channel: str):
    item = view.storage.get_data(str(image_uuid))
    assert item is not None
    view.set_uuid(image_uuid)
    view._replace_canvas_multichannel(item["data"], target_channel=channel)


def test_auto_contrast_uses_active_channel_after_uuid_switch(qapp):
    view = _make_view()
    first_uuid = uuid.uuid4()
    second_uuid = uuid.uuid4()
    view.storage.add_data(str(first_uuid), {"name": "first", "data": _make_channels(0)})
    view.storage.add_data(
        str(second_uuid), {"name": "second", "data": _make_channels(100)}
    )

    _load_uuid_on_channel(view, second_uuid, "Channel 2")
    assert view.current_channel == 1

    view.working_channels["Channel 1"].contrast_min = 11
    view.working_channels["Channel 1"].contrast_max = 22
    view.working_channels["Channel 2"].contrast_min = 33
    view.working_channels["Channel 2"].contrast_max = 44
    view.image_wrapper = view.working_channels["Channel 2"]

    view.auto_contrast()

    assert view.current_channel == 1
    assert view.image_wrapper is view.working_channels["Channel 2"]
    assert (
        view.working_channels["Channel 1"].contrast_min,
        view.working_channels["Channel 1"].contrast_max,
    ) == (11, 22)
    assert (
        view.working_channels["Channel 2"].contrast_min,
        view.working_channels["Channel 2"].contrast_max,
    ) != (33, 44)


def test_cmap_update_keeps_active_channel_after_uuid_switch(qapp):
    view = _make_view()
    image_uuid = uuid.uuid4()
    view.storage.add_data(str(image_uuid), {"name": "sample", "data": _make_channels(0)})

    _load_uuid_on_channel(view, image_uuid, "Channel 2")
    view.update_image(cmap_text="viridis")

    assert view.current_channel == 1
    assert view.image_wrapper is view.working_channels["Channel 2"]
    assert view.working_channels["Channel 2"].cmap == "viridis"
    assert view.working_channels["Channel 1"].cmap == "gray"


def test_swap_to_stardist_virtual_channel_keeps_non_label_cmap(qapp):
    view = _make_view()
    image_uuid = uuid.uuid4()
    view.storage.add_data(
        str(image_uuid), {"name": "sample", "data": _make_channels_with_stardist()}
    )

    _load_uuid_on_channel(view, image_uuid, "Channel 1")
    view.update_image(cmap_text="viridis")

    view.swap_channel(2)

    assert view.current_channel == 2
    assert view.image_wrapper is view.working_channels["Channel 3"]
    assert view.working_channels["Channel 3"].cmap == "viridis"


def test_rotate_and_flip_keep_active_channel(qapp):
    view = _make_view()
    image_uuid = uuid.uuid4()
    view.storage.add_data(str(image_uuid), {"name": "sample", "data": _make_channels(0)})

    _load_uuid_on_channel(view, image_uuid, "Channel 2")
    view.flip_horizontal()
    view.flip_vertical()

    assert view.current_channel == 1
    assert view.image_wrapper is view.working_channels["Channel 2"]

    rotated_result = {}
    for channel_name, wrapper in view.working_channels.items():
        updated = wrapper.copy()
        updated.data = wrapper.data + 1
        rotated_result[channel_name] = updated

    view.on_rotation_completed((image_uuid, rotated_result))

    assert view.current_channel == 1
    assert view.image_wrapper is view.working_channels["Channel 2"]


def test_crop_new_image_opens_on_active_channel(qapp, monkeypatch):
    view = _make_view()
    image_uuid = uuid.uuid4()
    view.storage.add_data(str(image_uuid), {"name": "sample", "data": _make_channels(0)})
    _load_uuid_on_channel(view, image_uuid, "Channel 2")

    class DummyDialog:
        confirm_crop = True

        def __init__(self, *_args, **_kwargs):
            pass

        def exec(self):
            return 0

    class DummySignal:
        def connect(self, *_args, **_kwargs):
            return None

    class DummyWorker:
        def __init__(self, fn, *args):
            self.fn = fn
            self.args = args
            self.signal = DummySignal()
            self.error = DummySignal()
            self.finished = DummySignal()

        def quit(self):
            return None

        def deleteLater(self):
            return None

        def start(self):
            return None

    monkeypatch.setattr(canvas_module, "ImageDialog", DummyDialog)
    monkeypatch.setattr(canvas_module, "Worker", DummyWorker)

    view.crop(QRect(0, 0, 2, 2))

    # crop() now calls add_to_canvas() directly; image_worker holds target_channel as last arg
    assert isinstance(view.image_worker, DummyWorker)
    assert view.image_worker.args[-1] == "Channel 2"


def test_uuid_reload_restores_stardist_overlay_state(qapp, monkeypatch):
    # Patch Worker to run synchronously so overlay build completes inline.
    class SyncWorker:
        def __init__(self, fn, *args, **kwargs):
            self._fn = fn
            self._args = args
            self._result = None
            self._signal_cb = None
            self._error_cb = None

        class _CB:
            def connect(self, cb):
                pass

        finished = _CB()
        error = _CB()

        @property
        def signal(self):
            return self

        def connect(self, cb):
            self._signal_cb = cb

        def isRunning(self):
            return False

        def deleteLater(self):
            pass

        def start(self):
            try:
                result = self._fn(*self._args)
                if self._signal_cb:
                    self._signal_cb(result)
            except Exception as exc:
                if self._error_cb:
                    self._error_cb(str(exc))

    monkeypatch.setattr(canvas_module, "Worker", SyncWorker)

    view = _make_view()
    image_uuid = uuid.uuid4()
    view.storage.add_data(
        str(image_uuid), {"name": "sample", "data": _make_channels_with_stardist()}
    )

    _load_uuid_on_channel(view, image_uuid, "Channel 1")

    assert view.segmentation_labels is not None
    assert view.stardist_virtual_channel == "Channel 3"
    assert view.stardist_source_channel == "Channel 1"

    view.set_stardist_overlay_enabled(True)

    assert view.stardist_overlay_enabled is True

    # Verify that overlay is applied to Channel 1 (non-virtual channel)
    test_img = np.ones((2, 2), dtype=np.uint16) * 1000
    view.current_channel = 0  # Channel 1
    view.image_wrapper = view.working_channels["Channel 1"]
    result_ch1 = view._apply_stardist_overlay(test_img)
    assert result_ch1.ndim == 3
    assert result_ch1.shape[2] == 3

    # Verify that overlay is also applied to Channel 2 (another non-virtual channel)
    view.current_channel = 1  # Channel 2
    view.image_wrapper = view.working_channels["Channel 2"]
    result_ch2 = view._apply_stardist_overlay(test_img)
    assert result_ch2.ndim == 3
    assert result_ch2.shape[2] == 3

    # Verify that overlay is NOT applied on Channel 3 (the virtual segmentation channel itself)
    view.current_channel = 2  # Channel 3
    view.image_wrapper = view.working_channels["Channel 3"]
    result_ch3 = view._apply_stardist_overlay(test_img)
    assert np.array_equal(result_ch3, test_img)

