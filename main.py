
if __name__ == "__main__":

    import ui.app as app, controller, image_processing.canvas, image_processing.stardist
    from PyQt6.QtWidgets import  QApplication
    import sys

    __app = QApplication(sys.argv)
    ui = app.Ui_MainWindow()
    model = image_processing.canvas.ImageGraphicsView()
    model_stardist = image_processing.stardist.StarDist()
    
    _controller = controller.Controller(model, model_stardist, ui)
    ui.show()
    sys.exit(__app.exec())