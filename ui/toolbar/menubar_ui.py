from PyQt6.QtWidgets import QMenuBar, QWidget, QMenu
from PyQt6.QtCore import QRect, QCoreApplication
from ui.toolbar.Action import Action

class MenuBarUI(QWidget):
    def __init__(self, parent=None, enc=None):
        self.enc = enc

        super().__init__()
        self.menubar = QMenuBar(parent)
        self.menubar.setGeometry(QRect(0, 0, 1061, 22))
        self.menuFile = QMenu(self.menubar)
        self.menuOpen =QMenu(self.menuFile)
        parent.setMenuBar(self.menubar)
        self.__createActions(parent)
        self.__addActions()
        self.__retranslateUI()

    def __createActions(self, parent):
        self.actionOpen = Action(parent, "actionOpen", "icons/folder.png")
        self.actionSaveAs = Action(parent, "actionSaveAs", "icons/save-as.png")
        self.actionOpenReference = Action(parent, "action_reference")
        # self.actionOpenFiles = Action(parent, "action_openFiles")

    def __addActions(self):
        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.menuOpen.menuAction())
        self.menuFile.addAction(self.actionSaveAs)
        self.menuOpen.addActions([self.actionOpenReference, self.actionOpen])

    def __retranslateUI(self):
        _translate = QCoreApplication.translate
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        
        self.actionOpenReference.setText(_translate("MainWindow", "Open Reference"))
        self.menuOpen.setTitle(_translate("MainWindow", "Open"))
        self.actionOpen.setText(_translate("MainWindow", "Open Image"))
        # self.actionOpenFiles.setText(_translate("MainWindow", "Open Files"))
        self.actionSaveAs.setText(_translate("MainWindow", "Save As..."))
        
        self.actionSaveAs.triggered.connect(self.print_hello)

    def print_hello(self):
        self.enc.save()


