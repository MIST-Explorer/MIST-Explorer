import image_processing.cell_intensity
import ui.app as app, controller, image_processing.canvas, image_processing.stardist, image_processing.register

from PyQt6.QtWidgets import  QApplication

if __name__ == "__main__":

    import cProfile
    import pstats
    import sys

    import sys

    __app = QApplication(sys.argv)
    ui = app.Ui_MainWindow()
    model_canvas = image_processing.canvas.ImageGraphicsView()
    model_stardist = image_processing.stardist.StarDist()
    model_register = image_processing.register.Register()
    model_cellIntensity = image_processing.cell_intensity.CellIntensity()
    _controller = controller.Controller(model_canvas, 
                                        model_stardist, 
                                        model_cellIntensity, 
                                        model_register,
                                        ui)
    ui.show()

    profiler = cProfile.Profile()
    profiler.enable()

    __app.exec()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()

    # sys.exit()