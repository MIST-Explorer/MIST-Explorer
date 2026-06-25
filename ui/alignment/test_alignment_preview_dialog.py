import numpy as np
import pytest
from PyQt6.QtGui import QTransform
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog
from ui.alignment.alignment_preview_dialog import (
    AlignmentPreviewDialog,
    readable_matrix_string,
    colorize_grayscale,
    transform_to_matrix,
)


@pytest.fixture
def dialog(qtbot):
    snapshot_data = {
        "target_image": np.zeros((100, 100), dtype=np.uint8),
        "aligned_image": np.zeros((100, 100), dtype=np.uint8),
    }
    dialog = AlignmentPreviewDialog(snapshot_data, can_edit=True)
    dialog.show()
    qtbot.addWidget(dialog)
    return dialog


def test_readable_matrix_string():
    matrix = np.array([[1, 0, 10], [0, 1, 20]], dtype=np.float32)
    s = readable_matrix_string(matrix)
    assert "Translation: (10.00, 20.00)" in s
    assert "Rotation: 0.00" in s
    assert "Scale: (x: 1.00, y: 1.00)" in s


def test_colorize_grayscale(qtbot):
    gray_img = np.zeros((10, 10), dtype=np.uint8)
    gray_img[5, 5] = 255
    pixmap = colorize_grayscale(gray_img, "red")
    assert not pixmap.isNull()
    img = pixmap.toImage()
    assert img.pixelColor(5, 5).red() == 255
    assert img.pixelColor(5, 5).alpha() == 255
    assert img.pixelColor(0, 0).alpha() == 0


def test_transform_to_matrix():
    t = QTransform()
    t.translate(10, 20)
    matrix = transform_to_matrix(t)
    assert np.array_equal(matrix, np.array([[1, 0, 10], [0, 1, 20]], dtype=np.float32))


def test_dialog_creation(dialog):
    assert isinstance(dialog, QDialog)
    assert dialog.windowTitle().startswith("Alignment Preview")


def test_editable_controls(dialog):
    assert dialog.dx_input.isVisible()
    assert dialog.dy_input.isVisible()
    assert dialog.rotation_input.isVisible()
    assert dialog.scale_input.isVisible()
    assert dialog.apply_trans_button.isVisible()
    assert dialog.rotate_button.isVisible()
    assert dialog.scale_button.isVisible()
    assert dialog.reset_button.isVisible()
    assert dialog.confirm_button.isVisible()
    assert dialog.cancel_button.isVisible()


def test_view_only_controls(qtbot):
    """AlignmentViewDialog (arrays-preview base) has no editing controls."""
    from ui.alignment.alignment_view_dialog import AlignmentViewDialog
    snapshot_data = {
        "target_image": np.zeros((100, 100), dtype=np.uint8),
        "aligned_image": np.zeros((100, 100), dtype=np.uint8),
    }
    dialog = AlignmentViewDialog(snapshot_data)
    qtbot.addWidget(dialog)
    assert not hasattr(dialog, "dx_input")
    assert hasattr(dialog, "confirm_button")
    assert not hasattr(dialog, "replace_button")


def test_contrast_checkbox(dialog):
    assert dialog.enhance_contrast_checkbox.isChecked()
    dialog.enhance_contrast_checkbox.setChecked(False)
    assert not dialog.adjust_contrast


def test_apply_manual_translation(dialog, qtbot):
    dialog.dx_input.setText("10")
    dialog.dy_input.setText("20")
    qtbot.mouseClick(dialog.apply_trans_button, Qt.MouseButton.LeftButton)
    assert dialog.offset_x == 10
    assert dialog.offset_y == 20


def test_apply_rotation(dialog, qtbot):
    dialog.rotation_input.setText("45")
    qtbot.mouseClick(dialog.rotate_button, Qt.MouseButton.LeftButton)
    assert dialog.transformations[-1][0] == 45


def test_apply_scale(dialog, qtbot):
    dialog.scale_input.setText("1.5")
    qtbot.mouseClick(dialog.scale_button, Qt.MouseButton.LeftButton)
    assert dialog.transformations[-1][-1] == "x1.5"


def test_move_aligned_image(dialog):
    dialog.move_aligned_image(10, 20)
    assert dialog.offset_x == 10
    assert dialog.offset_y == 20


def test_reset_zoom(dialog, qtbot):
    qtbot.mouseClick(dialog.reset_button, Qt.MouseButton.LeftButton)
    assert dialog.image_view.moving_item.transform().isIdentity()


def test_accept_alignment(dialog, qtbot):
    qtbot.mouseClick(dialog.confirm_button, Qt.MouseButton.LeftButton)
    assert dialog.result_accepted


def test_key_press_event(dialog, qtbot):
    qtbot.keyClick(dialog, Qt.Key.Key_Left)
    assert dialog.offset_x == -1


def test_fitted_state_tracking(dialog):
    view = dialog.image_view
    assert view._is_fitted is True

    # Scrolling/panning shifts it to False
    view.scrollContentsBy(5, 5)
    assert view._is_fitted is False

    # Resetting zoom restores it to True
    view.reset_zoom()
    assert view._is_fitted is True
