
import image_processing.cell_intensity
import ui.app as app, controller, image_processing.canvas, image_processing.stardist
from PyQt6.QtWidgets import  QApplication

if __name__ == "__main__":


    import sys

    __app = QApplication(sys.argv)
    ui = app.Ui_MainWindow()
    model_canvas = image_processing.canvas.ImageGraphicsView()
    model_stardist = image_processing.stardist.StarDist()
    model_cellIntensity = image_processing.cell_intensity.CellIntensity()

    _controller = controller.Controller(model_canvas, 
                                        model_stardist, 
                                        model_cellIntensity, 
                                        ui)
    ui.show()
    sys.exit(__app.exec())