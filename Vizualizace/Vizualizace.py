import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os
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

    def plot(self, data,data1=None,data2=None, max=-1, min=0):
        # print(min,max,len(data))
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x_values = np.arange(len(data))[min:max]
        ax.plot(x_values,data[min:max])
        try:
            ax.plot(x_values,data1[min:max],color="red")
            ax.plot(x_values,data2[min:max],color="green")
        except:
            pass

        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_ylim(data.min(),data.max())

        ax.tick_params(axis='x', rotation=45)
        self.figure.tight_layout(pad=0.0)
        self.canvas.draw()

    def resizeEvent(self, event):
        self.canvas.resize(self.size())
        super(MatplotlibWidget, self).resizeEvent(event)

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi(r'C:\Users\vojtu\PycharmProjects\Hydronaut\Vizualizace\Hydronaut_VisualisationTool.ui', self)

        self.widget_2 = self.findChild(QtWidgets.QWidget, 'widget_2')
        self.widget = self.findChild(QtWidgets.QWidget, 'widget')

        self.horizontalScrollBar = self.findChild(QtWidgets.QScrollBar, 'horizontalScrollBar_2')
        self.horizontalScrollBar_1 = self.findChild(QtWidgets.QScrollBar, 'horizontalScrollBar_3')

        self.pb_save_offset = self.findChild(QtWidgets.QPushButton, 'pb_safe_offset')
        self.pb_load_offset = self.findChild(QtWidgets.QPushButton, 'pb_load_offset')
        self.pb_file_load = self.findChild(QtWidgets.QPushButton, 'pb_file_load')

        self.pltWidget = self.embed_matplotlib(self.widget_2)
        self.pltWidget1 = self.embed_matplotlib(self.widget)

        self.horizontalScrollBar.valueChanged.connect(self.Update_Plot_1)

        self.pb_file_load.clicked.connect(self.load_csv_files)
        self.pb_save_offset.clicked.connect(self.Save_file)
        self.horizontalScrollBar.valueChanged.connect(self.Update_Plot_1)
        self.horizontalScrollBar_1.valueChanged.connect(self.Update_Plots)

        self.df = None
        self.files= []
        self.show()

    def load_csv_files(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select CSV or TXT File", "", "CSV and TXT Files (*.csv *.txt);;All Files (*)"
            )
            file_path = os.path.abspath(file_path)
            self.files.append(file_path)
            if len(self.files) == 2:
                if self.files[0].endswith(".csv") and self.files[1].endswith(".txt"):
                    self.df = pd.read_csv(self.files[0],delimiter="\t")
                    self.df1 = pd.read_csv(self.files[1],delimiter="\t",skiprows=4)
                    self.load_data()
                elif self.files[0].endswith(".txt") and self.files[1].endswith(".csv"):

                    self.df1 = pd.read_csv(self.files[0],delimiter="\t",skiprows=4)
                    self.df = pd.read_csv(self.files[1],delimiter="\t")
                    self.load_data()

            if len(self.files) > 2:
                self.files.pop(0)
                self.files.pop(0)

        except Exception as e:
            print(e)

    def embed_matplotlib(self, target_widget):
        matplotlib_widget = MatplotlibWidget()

        target_layout = QtWidgets.QVBoxLayout(target_widget)
        target_layout.addWidget(matplotlib_widget)

        target_layout.setStretch(0, 1)
        target_layout.setContentsMargins(0, 0, 0, 0)

        target_widget.setLayout(target_layout)
        return matplotlib_widget

    def load_data(self):
            self.horizontalScrollBar.setRange(0, int((len(self.df) - 1000)/2))
            self.horizontalScrollBar.setSingleStep(1)
            self.horizontalScrollBar_1.setRange(0, len(self.df1) - 1000)
            self.horizontalScrollBar_1.setSingleStep(1)
            self.Update_Plots()
            # print(self.horizontalScrollBar.minimum(), self.horizontalScrollBar_1.maximum())

    def Update_Plot_1(self):
        if self.df is None or self.df.empty:
            print("No data loaded!")
            return

        data = self.df.iloc[:, 0].values[::2]
        value = self.horizontalScrollBar.value()
        offset=self.horizontalScrollBar_1.value()
        print(value,offset)
        self.pltWidget.plot(data=data,min=value+offset,max=value+1000+offset)

    def Update_Plot_2(self):
        data = self.df1.iloc[:,0].values
        data1 = self.df1.iloc[:,1].values
        data2 = self.df1.iloc[:,2].values
        value = self.horizontalScrollBar_1.value()
        self.pltWidget1.plot(data=data,data1=data1,data2=data2,min=value,max=value+1000)

    def Update_Plots(self):
        self.Update_Plot_1()
        self.Update_Plot_2()

    def Save_file(self):
        value = self.horizontalScrollBar.value()
        try:
            df_copy = self.df.iloc[value:].copy()
            df_copy.columns = ["Value"]
            df_copy.to_csv("output.csv", index=True, index_label="Index", sep="\t")
        except Exception as e:
            print("Chyba při ukládání souboru:", e)

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
