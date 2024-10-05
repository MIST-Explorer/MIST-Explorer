from PyQt6.QtCore import QThread, pyqtSignal

class Worker(QThread):
    signal = pyqtSignal(object)  
    error = pyqtSignal(str)      
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            print("thread is running")
            self.result = self.func(*self.args, **self.kwargs)
            self.signal.emit(self.result)
        except Exception as e:
            self.error.emit(str(e))