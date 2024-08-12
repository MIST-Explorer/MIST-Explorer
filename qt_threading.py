from PyQt6.QtCore import QThread, pyqtSignal

class Worker(QThread):
    result = pyqtSignal(object)  
    error = pyqtSignal(str)      
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            print("thread is running")
            self.result.emit(result)
        except Exception as e:
            self.error.emit(str(e))