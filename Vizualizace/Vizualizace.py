import sys
from fileinput import filename

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

        self.pb_export = self.findChild(QtWidgets.QPushButton, 'pb_export')
        self.pb_load_IMU = self.findChild(QtWidgets.QPushButton, 'pb_load_IMU')
        self.pb_load_EMG = self.findChild(QtWidgets.QPushButton, 'pb_load_EMG')

        self.sb_set_offset = self.findChild(QtWidgets.QSpinBox, 'sb_set_offset')
        self.sb_set_pos = self.findChild(QtWidgets.QSpinBox, 'sb_set_pos')

        self.pltWidget = self.embed_matplotlib(self.widget_2)
        self.pltWidget1 = self.embed_matplotlib(self.widget)

        self.horizontalScrollBar.valueChanged.connect(self.Update_Plot_1)
        self.pb_load_EMG.clicked.connect(self.load_emg)
        self.pb_load_IMU.clicked.connect(self.load_imu)

        self.pb_export.clicked.connect(self.Save_file)
        self.horizontalScrollBar.valueChanged.connect(self.Update_Plot_1)
        self.horizontalScrollBar_1.valueChanged.connect(self.Update_Plots)

        self.df_emg = None
        self.df_imu = None
        self.IMU = None
        self.EMG = None
        self.show()

    def load_imu(self):
            self.pb_load_IMU.setStyleSheet("color: rgb(0, 0, 255);")
            folder_path = QFileDialog.getExistingDirectory(
                self,
                "Select IMU Folder",
                "",
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )

            if folder_path != "":
                self.df_imu = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file).replace("\\", "/")
                    if os.path.isfile(file_path):
                        print("Processing file:", file)
                        try:
                            if self.df_imu[0].empty and file.startswith("b"):
                                self.df_imu[0] = pd.read_csv(file_path, delimiter="\t", skiprows=4)
                            elif self.df_imu[1].empty and file.startswith("t"):
                                self.df_imu[1] = pd.read_csv(file_path, delimiter="\t", skiprows=4)
                            elif self.df_imu[2].empty and file.startswith("r"):
                                self.df_imu[2] = pd.read_csv(file_path, delimiter="\t", skiprows=4)
                            elif self.df_imu[3].empty and file.startswith("g"):
                                self.df_imu[3] = pd.read_csv(file_path, delimiter="\t", skiprows=4)
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")

                if not self.df_imu[0].empty and not self.df_imu[1].empty and not self.df_imu[2].empty and not self.df_imu[3].empty:
                    self.pb_load_IMU.setStyleSheet("color: rgb(0, 255, 0);")
                    self.horizontalScrollBar_1.setRange(0, len(self.df_imu[0]) - 1000)
                    self.horizontalScrollBar_1.setSingleStep(1)
                    self.Update_Plot_2()
                else:
                    self.pb_load_IMU.setStyleSheet("color: rgb(255, 0, 0);")
            if self.df_imu is None:
                self.pb_load_IMU.setStyleSheet("color: rgb(255, 0, 0);")

    def load_emg(self):
            self.pb_load_EMG.setStyleSheet("color: rgb(0, 0, 255);")
            folder_path = QFileDialog.getExistingDirectory(
                self,
                "Select EMG Folder",
                "",
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            if folder_path != "":
                self.df_emg = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
                for path in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, path).replace("\\", "/")
                    if os.path.isfile(file_path):
                        print("Processing file:", path)
                        try:
                            if self.df_emg[0].empty and path.startswith("b"):
                                self.df_emg[0] = pd.read_csv(file_path, delimiter="\t")
                            elif self.df_emg[1].empty and path.startswith("t"):
                                self.df_emg[1] = pd.read_csv(file_path, delimiter="\t")
                            elif self.df_emg[2].empty and path.startswith("r"):
                                self.df_emg[2] = pd.read_csv(file_path, delimiter="\t")
                            elif self.df_emg[3].empty and path.startswith("g"):
                                self.df_emg[3] = pd.read_csv(file_path, delimiter="\t")
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")

                if not self.df_emg[0].empty and not self.df_emg[1].empty and not self.df_emg[2].empty and not self.df_emg[3].empty:
                    self.pb_load_EMG.setStyleSheet("color: rgb(0, 255, 0);")
                    self.horizontalScrollBar.setRange(0, int((len(self.df_emg[0]) - 1000) / 2))
                    self.horizontalScrollBar.setSingleStep(1)
                    self.Update_Plot_1()
                else:
                    self.pb_load_EMG.setStyleSheet("color: rgb(255, 0, 0);")
            if self.df_emg is None:
                self.pb_load_EMG.setStyleSheet("color: rgb(255, 0, 0);")

    @staticmethod
    def embed_matplotlib(target_widget):
        matplotlib_widget = MatplotlibWidget()

        target_layout = QtWidgets.QVBoxLayout(target_widget)
        target_layout.addWidget(matplotlib_widget)

        target_layout.setStretch(0, 1)
        target_layout.setContentsMargins(0, 0, 0, 0)

        target_widget.setLayout(target_layout)
        return matplotlib_widget


    def Update_Plot_1(self):
        data = self.df_emg[0].iloc[:, 1].values[::2]
        value = self.horizontalScrollBar.value()
        offset=self.horizontalScrollBar_1.value()
        self.pltWidget.plot(data=data,min=value+offset,max=value+1000+offset)

    def Update_Plot_2(self):
        data = self.df_imu[0].iloc[:,0].values
        data1 = self.df_imu[0].iloc[:,1].values
        data2 = self.df_imu[0].iloc[:,2].values
        value = self.horizontalScrollBar_1.value()
        self.pltWidget1.plot(data=data,data1=data1,data2=data2,min=value,max=value+1000)

    def Update_Plots(self):
        self.Update_Plot_1()
        self.Update_Plot_2()

    def Save_file(self):
        value = self.horizontalScrollBar.value()
        df_copy = self.df_emg.iloc[value:,1].copy()
        df_copy.columns = ["Value"]
        filepath, _ = QFileDialog.getSaveFileName(
                self, "Save file", "", "All Files (*);; Text Files (*.txt)"
            )
        df_copy.to_csv(filepath, index=True, index_label="Index", sep="\t")

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
