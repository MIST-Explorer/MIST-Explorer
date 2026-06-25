import uuid
from datetime import datetime

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from core.canvas import ImageStorage, ImageWrapper
from models.workspace import ProjectMetadata
from ui.image_manager import ImageManager


@pytest.fixture(autouse=True)
def _clear_image_storage():
    storage = ImageStorage()
    storage.clear_data()
    yield
    storage.clear_data()


def _add_image(manager: ImageManager, name: str, channels: dict[str, ImageWrapper]):
    image_uuid = uuid.uuid4()
    manager.storage.add_data(str(image_uuid), {"name": name, "data": channels})
    manager.add_item(image_uuid)
    return image_uuid


def _multi_channels():
    return {
        "Channel 1": ImageWrapper(
            np.array([[1, 2], [3, 4]], dtype=np.uint16), name="Channel 1"
        ),
        "Channel 2": ImageWrapper(
            np.array([[5, 6], [7, 8]], dtype=np.uint16), name="Channel 2"
        ),
    }


def _three_channels():
    return {
        "Channel 1": ImageWrapper(
            np.array([[1, 2], [3, 4]], dtype=np.uint16), name="Channel 1"
        ),
        "Channel 2": ImageWrapper(
            np.array([[10, 20], [30, 40]], dtype=np.uint16), name="Channel 2"
        ),
        "Channel 3": ImageWrapper(
            np.array([[100, 200], [300, 400]], dtype=np.uint16), name="Channel 3"
        ),
    }


def test_allowed_exports_for_child_layer_is_tif_and_png(qapp):
    manager = ImageManager()
    _add_image(manager, "multi_layer", _multi_channels())
    root_index = manager.image_tree_model.index(0, 0)
    child_index = manager.image_tree_model.index(0, 0, root_index)

    assert manager.image_tree_view._allowed_export_formats(child_index) == [
        "tif",
        "png",
    ]


def test_allowed_exports_for_stardist_child_is_tif_and_png(qapp):
    manager = ImageManager()
    channels = _multi_channels()
    channels["Channel 3"] = ImageWrapper(
        np.array([[0, 1], [1, 2]], dtype=np.uint16),
        name="Segmentation",
        cmap="label_image",
    )
    _add_image(manager, "multi_with_stardist", channels)
    root_index = manager.image_tree_model.index(0, 0)
    stardist_child_index = manager.image_tree_model.index(2, 0, root_index)

    assert manager.image_tree_view._allowed_export_formats(stardist_child_index) == [
        "tif",
        "png",
    ]


def test_allowed_exports_for_single_layer_root_is_tif_and_png(qapp):
    manager = ImageManager()
    _add_image(
        manager,
        "single_layer",
        {
            "Channel 1": ImageWrapper(
                np.array([[10, 20], [30, 40]], dtype=np.uint16), name="Channel 1"
            )
        },
    )
    root_index = manager.image_tree_model.index(0, 0)

    assert manager.image_tree_view._allowed_export_formats(root_index) == [
        "tif",
        "png",
    ]


@pytest.mark.parametrize(
    "image_name", ["single_layer.tif", "single_layer.png", "single_layer.jpg"]
)
def test_single_layer_adds_channel_one_child_when_new_channel_added(qapp, image_name):
    manager = ImageManager()
    image_uuid = _add_image(
        manager,
        image_name,
        {
            "Channel 1": ImageWrapper(
                np.array([[10, 20], [30, 40]], dtype=np.uint16), name="Channel 1"
            )
        },
    )
    root_index = manager.image_tree_model.index(0, 0)
    root_item = manager.image_tree_model.itemFromIndex(root_index)
    assert root_item is not None
    assert root_item.rowCount() == 0

    storage_item = manager.storage.get_data(image_uuid)
    assert storage_item is not None
    storage_item["data"]["Channel 2"] = ImageWrapper(
        np.array([[0, 1], [1, 2]], dtype=np.uint16),
        name="Segmentation",
        cmap="label_image",
    )

    manager.set_channel_icon(image_uuid, "Channel 2")

    root_item = manager.image_tree_model.itemFromIndex(root_index)
    assert root_item is not None
    child_channels = [
        root_item.child(i).data(Qt.ItemDataRole.WhatsThisRole)
        for i in range(root_item.rowCount())
    ]
    assert child_channels == ["Channel 1", "Channel 2"]


def test_allowed_exports_for_multi_layer_root_is_tif_only(qapp):
    manager = ImageManager()
    _add_image(manager, "multi_layer", _multi_channels())
    root_index = manager.image_tree_model.index(0, 0)

    assert manager.image_tree_view._allowed_export_formats(root_index) == ["tif"]


def test_save_as_png_uses_selected_single_channel(qapp, monkeypatch, tmp_path):
    manager = ImageManager()
    channel_1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    channel_2 = np.array([[11, 12], [13, 14]], dtype=np.uint8)
    _add_image(
        manager,
        "multi_layer",
        {
            "Channel 1": ImageWrapper(channel_1, name="Channel 1"),
            "Channel 2": ImageWrapper(channel_2, name="Channel 2"),
        },
    )
    root_index = manager.image_tree_model.index(0, 0)
    child_index = manager.image_tree_model.index(1, 0, root_index)

    monkeypatch.setattr(
        "ui.image_manager.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(tmp_path),
    )

    saved = {}

    class _FakeImage:
        def save(self, path):
            saved["path"] = path

    def _fake_fromarray(array):
        saved["array"] = np.array(array, copy=True)
        return _FakeImage()

    monkeypatch.setattr("ui.image_manager.Image.fromarray", _fake_fromarray)

    manager.image_tree_view.save_as(child_index, "png", single_channel="Channel 2")

    assert np.array_equal(saved["array"], channel_2)
    assert not np.array_equal(saved["array"], channel_1)
    assert saved["path"].endswith("Channel 2.png")


def test_save_as_tif_uses_non_imagej_for_int32_channel(qapp, monkeypatch, tmp_path):
    manager = ImageManager()
    _add_image(
        manager,
        "int32_labels",
        {
            "Channel 1": ImageWrapper(
                np.array([[0, 70000], [1, 2]], dtype=np.int32),
                name="Segmentation",
                cmap="label_image",
            )
        },
    )
    root_index = manager.image_tree_model.index(0, 0)

    monkeypatch.setattr(
        "ui.image_manager.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(tmp_path),
    )

    captured = {}

    def _fake_imwrite(path, data, **kwargs):
        captured["path"] = path
        captured["data"] = np.array(data, copy=True)
        captured["kwargs"] = kwargs

    monkeypatch.setattr("ui.image_manager.tifffile.imwrite", _fake_imwrite)

    manager.image_tree_view.save_as(root_index, "tif", single_channel="Channel 1")

    assert captured["path"].endswith("int32_labels.tif")
    assert captured["data"].dtype == np.int32
    assert captured["kwargs"]["imagej"] is False


def test_delete_non_stardist_child_reindexes_channels(qapp):
    manager = ImageManager()
    image_uuid = _add_image(manager, "three_layer", _three_channels())

    root_index = manager.image_tree_model.index(0, 0)
    channel_2_index = manager.image_tree_model.index(1, 0, root_index)
    manager.image_tree_view.delete_item(channel_2_index)

    storage_item = manager.storage.get_data(image_uuid)
    assert storage_item is not None
    remaining = storage_item["data"]
    assert list(remaining.keys()) == ["Channel 1", "Channel 2"]
    assert np.array_equal(
        remaining["Channel 2"].data,
        np.array([[100, 200], [300, 400]], dtype=np.uint16),
    )
    assert remaining["Channel 2"].name == "Channel 2"

    root_item = manager.image_tree_model.itemFromIndex(root_index)
    assert root_item is not None
    child_channels = [
        root_item.child(i).data(Qt.ItemDataRole.WhatsThisRole)
        for i in range(root_item.rowCount())
    ]
    assert child_channels == ["Channel 1", "Channel 2"]


def test_delete_channel_repairs_root_default_channel(qapp):
    manager = ImageManager()
    _add_image(manager, "multi_layer", _multi_channels())

    root_index = manager.image_tree_model.index(0, 0)
    root_item = manager.image_tree_model.itemFromIndex(root_index)
    assert root_item is not None
    root_item.setData("Channel 2", Qt.ItemDataRole.WhatsThisRole)

    channel_2_index = manager.image_tree_model.index(1, 0, root_index)
    manager.image_tree_view.delete_item(channel_2_index)

    root_item = manager.image_tree_model.itemFromIndex(root_index)
    assert root_item is not None
    assert root_item.data(Qt.ItemDataRole.WhatsThisRole) == "Channel 1"


def test_temp_project_skips_auto_persist_on_add_and_update(qapp, monkeypatch, tmp_path):
    manager = ImageManager()
    project_path = tmp_path / "temp_session.mist"
    project_path.mkdir()

    metadata = ProjectMetadata(
        name="Session 1",
        path=project_path,
        created_at=datetime.now(),
        modified_at=datetime.now(),
        is_temporary=True,
    )
    monkeypatch.setattr("ui.image_manager.ProjectManager.load_project", lambda *_: metadata)

    save_calls = []
    monkeypatch.setattr(
        "ui.image_manager.ProjectManager.save_image",
        lambda **kwargs: save_calls.append(kwargs),
    )

    manager.set_project_path(project_path)
    image_uuid = _add_image(
        manager,
        "single_layer",
        {
            "Channel 1": ImageWrapper(
                np.array([[1, 2], [3, 4]], dtype=np.uint16), name="Channel 1"
            )
        },
    )
    assert save_calls == []

    storage_item = manager.storage.get_data(image_uuid)
    assert storage_item is not None
    storage_item["data"]["Channel 2"] = ImageWrapper(
        np.array([[5, 6], [7, 8]], dtype=np.uint16), name="Channel 2"
    )
    manager.set_channel_icon(image_uuid, "Channel 2")
    assert save_calls == []


def test_temp_project_skips_auto_persist_on_delete(qapp, monkeypatch, tmp_path):
    manager = ImageManager()
    project_path = tmp_path / "temp_session.mist"
    project_path.mkdir()

    metadata = ProjectMetadata(
        name="Session 2",
        path=project_path,
        created_at=datetime.now(),
        modified_at=datetime.now(),
        is_temporary=True,
    )
    monkeypatch.setattr("ui.image_manager.ProjectManager.load_project", lambda *_: metadata)

    delete_calls = []
    monkeypatch.setattr(
        "ui.image_manager.ProjectManager.delete_image",
        lambda *args, **kwargs: delete_calls.append((args, kwargs)) or True,
    )

    manager.set_project_path(project_path)
    _add_image(
        manager,
        "single_layer",
        {
            "Channel 1": ImageWrapper(
                np.array([[1, 2], [3, 4]], dtype=np.uint16), name="Channel 1"
            )
        },
    )

    root_index = manager.image_tree_model.index(0, 0)
    manager.image_tree_view.delete_item(root_index)
    assert delete_calls == []
    assert manager.image_tree_model.rowCount() == 0


def test_set_as_segmentation_label_moves_channel_to_parent_image(qapp, monkeypatch):
    manager = ImageManager()

    # Add two images
    img1_uuid = _add_image(
        manager,
        "SourceImage",
        {
            "Channel 1": ImageWrapper(
                np.array([[10, 20], [30, 40]], dtype=np.uint16), name="Channel 1"
            ),
            "Channel 2": ImageWrapper(
                np.array([[0, 1], [1, 2]], dtype=np.uint16), name="Channel 2"
            )
        }
    )
    img2_uuid = _add_image(
        manager,
        "TargetImage",
        {
            "Channel 1": ImageWrapper(
                np.array([[100, 200], [300, 400]], dtype=np.uint16), name="Channel 1"
            )
        }
    )

    # We want to set SourceImage's Channel 2 as a segmentation label, moving it to TargetImage.
    # The dialog should list TargetImage.
    # We mock QInputDialog.getItem to select "TargetImage (xxxx)"
    mocked_getItem_calls = []
    def mock_getItem(parent, title, label, items, current, editable):
        mocked_getItem_calls.append(items)
        return items[0], True # TargetImage (xxx) is the only other image

    monkeypatch.setattr("PyQt6.QtWidgets.QInputDialog.getItem", mock_getItem)

    # Channel index of "Channel 2" in img1 is 1.
    manager.image_tree_view.set_as_segmentation_label(img1_uuid, 1)

    # Check that getItem was called with TargetImage
    assert len(mocked_getItem_calls) == 1
    assert "TargetImage" in mocked_getItem_calls[0][0]

    # Verify channel was removed from img1
    img1_entry = manager.storage.get_data(img1_uuid)
    assert "Channel 2" not in img1_entry["data"]
    assert len(img1_entry["data"]) == 1

    # Verify channel was moved to img2 and marked as a segmentation channel
    img2_entry = manager.storage.get_data(img2_uuid)
    assert "Channel 2" in img2_entry["data"]
    moved_wrapper = img2_entry["data"]["Channel 2"]
    from core.project_naming import is_segmentation_channel
    assert is_segmentation_channel(moved_wrapper)
    assert moved_wrapper.cmap == "label_image"

    # Verify seg_label_uuid in storage points to target
    seg_label = manager.storage.get_data("seg_label_uuid")
    assert seg_label is not None
    assert str(seg_label["value"]) == str(img2_uuid)
    assert seg_label["channel"] == 1 # Channel 2 is at index 1


def test_set_as_segmentation_label_deletes_source_if_empty(qapp, monkeypatch):
    manager = ImageManager()

    img1_uuid = _add_image(
        manager,
        "SourceImage",
        {
            "Channel 1": ImageWrapper(
                np.array([[10, 20], [30, 40]], dtype=np.uint16), name="Channel 1"
            )
        }
    )
    img2_uuid = _add_image(
        manager,
        "TargetImage",
        {
            "Channel 1": ImageWrapper(
                np.array([[100, 200], [300, 400]], dtype=np.uint16), name="Channel 1"
            )
        }
    )

    def mock_getItem(parent, title, label, items, current, editable):
        return items[0], True

    monkeypatch.setattr("PyQt6.QtWidgets.QInputDialog.getItem", mock_getItem)

    # Channel index of "Channel 1" in img1 is 0.
    manager.image_tree_view.set_as_segmentation_label(img1_uuid, 0)

    # Verify SourceImage is deleted from storage and model
    assert manager.storage.get_data(img1_uuid) is None
    assert manager.image_tree_model.rowCount() == 1 # Only TargetImage left

    # Verify TargetImage has the moved segmentation channel
    img2_entry = manager.storage.get_data(img2_uuid)
    assert "Channel 2" in img2_entry["data"]

