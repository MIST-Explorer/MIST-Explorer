import ui.app as app, controller

from PyQt6.QtWidgets import  QApplication

if __name__ == "__main__":

    import cProfile
    import pstats
    import sys

    __app = QApplication(sys.argv)
    window = app.Ui_MainWindow()

    _controller = controller.Controller(window)
    window.show()

    # profiler = cProfile.Profile()
    # profiler.enable()

    __app.exec()

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.print_stats()

    # sys.exit()