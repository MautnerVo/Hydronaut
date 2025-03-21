import sys
from PyQt5 import QtWidgets, uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

class MatplotlibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.figure = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)


        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.canvas)


        self.layout.setStretch(0, 1)


        self.layout.setContentsMargins(0, 0, 0, 0)


    def plot(self,data):
        ax = self.figure.add_subplot(111)
        ax.plot(data[:1000])

        ax.set_xticks([])
        # ax.set_yticks([])

        self.figure.tight_layout(pad=0.0)
        self.canvas.draw()

    def resizeEvent(self, event):
        self.canvas.resize(self.size())
        super(MatplotlibWidget, self).resizeEvent(event)


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('Hydronaut_VisualisationTool.ui', self)

        self.widget_2 = self.findChild(QtWidgets.QWidget, 'widget_2')
        self.widget = self.findChild(QtWidgets.QWidget, 'widget')

        self.pltWidget = self.embed_matplotlib(self.widget_2)
        self.pltWidget1 = self.embed_matplotlib(self.widget)

        self.show()

        self.loadata()


    def embed_matplotlib(self, target_widget):
        matplotlib_widget = MatplotlibWidget()

        target_layout = QtWidgets.QVBoxLayout(target_widget)

        target_layout.addWidget(matplotlib_widget)

        target_layout.setStretch(0, 1)

        target_layout.setContentsMargins(0, 0, 0, 0)

        target_widget.setLayout(target_layout)
        return matplotlib_widget

    def loadata(self):
        self.df = pd.read_csv("1_EMG_521_0420_00133.csv")
        self.pltWidget.plot(self.df)

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
