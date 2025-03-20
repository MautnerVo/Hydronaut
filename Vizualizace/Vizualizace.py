import sys
from PyQt5 import QtWidgets, uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class MatplotlibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.figure = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)


        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.canvas)


        self.layout.setStretch(0, 1)


        self.layout.setContentsMargins(0, 0, 0, 0)

        self.plot()

    def plot(self):
        ax = self.figure.add_subplot(111)
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)

        ax.set_yticks([])

        self.figure.tight_layout(pad=0.0)  # Adjust padding to remove excess space
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

        self.embed_matplotlib(self.widget_2)
        self.embed_matplotlib(self.widget)

        self.show()

    def embed_matplotlib(self, target_widget):
        matplotlib_widget = MatplotlibWidget()

        target_layout = QtWidgets.QVBoxLayout(target_widget)

        target_layout.addWidget(matplotlib_widget)

        target_layout.setStretch(0, 1)

        target_layout.setContentsMargins(0, 0, 0, 0)

        target_widget.setLayout(target_layout)


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
