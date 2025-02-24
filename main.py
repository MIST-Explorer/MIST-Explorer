import core.cell_intensity
import ui.app as app, controller, core.canvas, core.stardist, core.register

from PyQt6.QtWidgets import  QApplication

if __name__ == "__main__":

    import cProfile
    import pstats
    import sys

    __app = QApplication(sys.argv)
    ui = app.Ui_MainWindow()

    model_canvas = core.canvas.ImageGraphicsView()
    model_stardist = core.stardist.StarDist()
    model_register = core.register.Register()
    model_cellIntensity = core.cell_intensity.CellIntensity()

    _controller = controller.Controller(model_canvas, 
                                        model_stardist, 
                                        model_cellIntensity, 
                                        model_register,
                                        ui)
    ui.show()

    # profiler = cProfile.Profile()
    # profiler.enable()

    __app.exec()

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.print_stats()

    # sys.exit()