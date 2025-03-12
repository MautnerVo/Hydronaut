import sys
from PyQt5 import QtWidgets, uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


# Custom widget for embedding Matplotlib plot
class MatplotlibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        # Create a figure and canvas to display the plot
        self.figure = Figure(figsize=(5, 3), tight_layout=True)  # Set tight_layout to reduce white space
        self.canvas = FigureCanvas(self.figure)  # Create a canvas to display the figure

        # Create a layout and add the canvas to it
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.canvas)

        # Set the layout's stretch factor to 1 to fill all available space
        self.layout.setStretch(0, 1)

        # Remove the default margin for a more compact fit
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Example plot
        self.plot()

    def plot(self):
        ax = self.figure.add_subplot(111)
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)

        # Remove extra space from all sides of the plot
        # ax.margins(0, 0)  # Remove margins on both axes

        # Disable the axis ticks to further clean the plot appearance
        # ax.set_xticks([])
        ax.set_yticks([])

        # Automatically adjust the layout and remove extra padding
        self.figure.tight_layout(pad=0.0)  # Adjust padding to remove excess space
        self.canvas.draw()

    def resizeEvent(self, event):
        # Resize the canvas when the widget is resized
        self.canvas.resize(self.size())
        super(MatplotlibWidget, self).resizeEvent(event)


# Load UI and embed Matplotlib widget
class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('Hydronaut_VisualisationTool.ui', self)

        # Find the QWidget where you want to embed the plot
        self.widget_2 = self.findChild(QtWidgets.QWidget, 'widget_2')  # Replace with actual widget name
        self.widget = self.findChild(QtWidgets.QWidget, 'widget')  # Replace with actual widget name

        # Embed Matplotlib plot into the widgets (replace QWidgets with MatplotlibWidget)
        self.embed_matplotlib(self.widget_2)
        self.embed_matplotlib(self.widget)

        self.show()

    def embed_matplotlib(self, target_widget):
        # Create a MatplotlibWidget and set it into the target QWidget
        matplotlib_widget = MatplotlibWidget()

        # Create a layout for the target widget
        target_layout = QtWidgets.QVBoxLayout(target_widget)

        # Add Matplotlib widget to the layout
        target_layout.addWidget(matplotlib_widget)

        # Set stretch factor to 1 so it fills the available space
        target_layout.setStretch(0, 1)

        # Remove margins to make it take all the space
        target_layout.setContentsMargins(0, 0, 0, 0)

        # Ensure that the layout stretches in both directions
        target_widget.setLayout(target_layout)


# Run the application
app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
