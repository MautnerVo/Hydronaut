import sys
import threading
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os
import pandas as pd
import faulthandler
faulthandler.enable()


class MatplotlibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.figure = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.canvas)

        self.layout.setStretch(0, 1)
        self.layout.setContentsMargins(0, 0, 0, 0)

    def plot(self, data,data1=None,data2=None, max=-1, min=0,local_max=False):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x_values = np.arange(len(data))[min:max]
        ax.plot(x_values,data[min:max])
        if data1 is not None:
            try:
                ax.plot(x_values,data1[min:max],color="red")
                ax.plot(x_values,data2[min:max],color="green")

                if not local_max:
                    y_min = np.nanmin([
                        np.min(data),
                        np.min(data1) if data1 is not None else np.inf,
                        np.min(data2) if data2 is not None else np.inf
                    ])
                    y_max = np.max(data)
                else:
                    y_min = np.nanmin([
                        np.min(data[min:max]),
                        np.min(data1[min:max]) if data1 is not None else np.inf,
                        np.min(data2[min:max]) if data2 is not None else np.inf
                    ])
                    y_max = np.max(data)
                ax.set_ylim(y_min, y_max)
            except:
                pass

        # ax.set_xticks([])
        # ax.set_yticks([])

        if data1 is None and data2 is None:
            if not local_max:
                ax.set_ylim(data.min(),data.max())
            else:
                ax.set_ylim(data[min:max].min(), data[min:max].max())

        ax.tick_params(axis='x', rotation=45)
        self.figure.tight_layout(pad=0.0)
        self.canvas.draw()

    def resizeEvent(self, event):
        self.canvas.resize(self.size())
        super(MatplotlibWidget, self).resizeEvent(event)

class Ui(QtWidgets.QMainWindow):
    emg_loading_finished = pyqtSignal(bool)
    imu_loading_finished = pyqtSignal(bool)
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi(os.path.dirname(os.path.abspath(__file__))+r'\\Hydronaut_VisualisationTool.ui', self)

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
        self.cb_local_maximum = self.findChild(QtWidgets.QCheckBox, 'cb_local_maximum')

        self.horizontalScrollBar.valueChanged.connect(self.Update_Plot_1)
        self.pb_load_EMG.clicked.connect(self.emg_loader)
        self.pb_load_IMU.clicked.connect(self.imu_loader)

        self.pb_export.clicked.connect(self.Save_file)
        self.horizontalScrollBar.valueChanged.connect(self.Update_Plot_1)
        self.horizontalScrollBar_1.valueChanged.connect(self.Update_Plots)

        self.sb_set_offset.setMaximum(1000000)
        self.sb_set_pos.setMaximum(1000000)

        self.sb_set_offset.editingFinished.connect(self.Update_Plot_1)
        self.sb_set_pos.editingFinished.connect(self.Update_Plots)
        self.cb_local_maximum.stateChanged.connect(self.Update_Plots)

        self.emg_loading_finished.connect(self.finish_emg_loading)
        self.imu_loading_finished.connect(self.finish_imu_loading)

        self.df_emg = []
        self.df_imu = []
        self.show()

    def imu_loader(self):

        self.pb_load_IMU.setStyleSheet("color: rgb(0, 0, 255);")
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select IMU Folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder_path != "":
            thread_imu = threading.Thread(target=self.load_imu, args=(folder_path,))
            thread_imu.start()
        else:
            self.pb_load_IMU.setStyleSheet("color: rgb(255, 0, 0);")

    def load_imu(self,folder_path):
            self.df_imu = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
            success = False
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file).replace("\\", "/")
                if os.path.isfile(file_path):
                    print("Processing file:", file)
                    try:
                        if self.df_imu[0].empty and file.startswith("b"):
                            self.df_imu[0] = pd.read_csv(file_path, delimiter="\t", skiprows=4)
                        elif self.df_imu[1].empty and file.startswith("t"):
                            self.df_imu[1] = pd.read_csv(file_path, delimiter="\t", skiprows=4)
                        elif self.df_imu[2].empty and file.startswith("g"):
                            self.df_imu[2] = pd.read_csv(file_path, delimiter="\t", skiprows=4)
                        elif self.df_imu[3].empty and file.startswith("r"):
                            self.df_imu[3] = pd.read_csv(file_path, delimiter="\t", skiprows=4)
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

            if not self.df_imu[0].empty and not self.df_imu[1].empty and not self.df_imu[2].empty and not self.df_imu[3].empty:

                success = True
            else:
                success = False
            self.imu_loading_finished.emit(success)

    def finish_imu_loading(self,success):
        if success:
            self.pb_load_IMU.setStyleSheet("color: rgb(0, 255, 0);")
            self.horizontalScrollBar_1.setRange(0, len(self.df_imu[0]) - 1000)
            self.horizontalScrollBar_1.setSingleStep(1)
            self.Update_Plot_1()
        else:
            self.pb_load_IMU.setStyleSheet("color: rgb(255, 0, 0);")

    def emg_loader(self):
        self.pb_load_EMG.setStyleSheet("color: rgb(0, 0, 255);")
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select EMG Folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder_path != "":
            thread_emg = threading.Thread(target=self.load_emg,args=(folder_path,))
            thread_emg.start()
        else:
            self.pb_load_EMG.setStyleSheet("color: rgb(255, 0, 0);")


    def load_emg(self,folder_path):
            self.df_emg = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
            success = False
            for path in os.listdir(folder_path):
                file_path = os.path.join(folder_path, path).replace("\\", "/")
                if os.path.isfile(file_path):
                    print("Processing file:", path)
                    try:
                        if self.df_emg[0].empty and path.startswith("b"):
                            self.df_emg[0] = pd.read_csv(file_path, delimiter="\t")
                        elif self.df_emg[1].empty and path.startswith("t"):
                            self.df_emg[1] = pd.read_csv(file_path, delimiter="\t")
                        elif self.df_emg[2].empty and path.startswith("g"):
                            self.df_emg[2] = pd.read_csv(file_path, delimiter="\t")
                        elif self.df_emg[3].empty and path.startswith("r"):
                            self.df_emg[3] = pd.read_csv(file_path, delimiter="\t")
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")


            if not self.df_emg[0].empty and not self.df_emg[1].empty and not self.df_emg[2].empty and not self.df_emg[3].empty:
                success = True
            else:
                success = False

            self.emg_loading_finished.emit(success)

    def finish_emg_loading(self,success):
        if success:
            self.pb_load_EMG.setStyleSheet("color: rgb(0, 255, 0);")
            self.horizontalScrollBar.setRange(0, int((len(self.df_emg[0]) - 1000) / 2))
            self.horizontalScrollBar.setSingleStep(1)
            self.Update_Plot_2()
        else:
            self.pb_load_EMG.setStyleSheet("color: rgb(255, 0, 0);")

    def arrange_size(self, size,df_emg):
            for i,df in enumerate(df_emg):
                values = df.to_numpy().tolist()
                result = []

                for value in values:
                    result.append(value)

                while size > len(result):
                    result.append([np.nan, np.nan])

                df_emg[i] = pd.DataFrame(result,columns=df.columns)
            return df_emg

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
        if len(self.df_emg) != 0:
            data = self.df_imu[0].iloc[:, 9].values
            data1 = self.df_imu[0].iloc[:, 10].values
            data2 = self.df_imu[0].iloc[:, 11].values
            value = self.horizontalScrollBar.value()
            value += self.sb_set_offset.value()
            offset= self.horizontalScrollBar_1.value()
            offset += self.sb_set_pos.value()
            checked = self.cb_local_maximum.isChecked()
            self.pltWidget.plot(data=data,data1=data1,data2=data2,min=value+offset,max=value+1000+offset,local_max=checked)

    def Update_Plot_2(self):
        if len(self.df_imu) != 0:
            data = self.df_emg[0].iloc[:, 1].values[::2]
            value = self.horizontalScrollBar_1.value()
            value +=  self.sb_set_pos.value()
            checked = self.cb_local_maximum.isChecked()
            self.pltWidget1.plot(data=data,min=value,max=value+1000,local_max=checked)

    def Update_Plots(self):
        self.Update_Plot_1()
        self.Update_Plot_2()

    def Save_file(self):
        value = self.horizontalScrollBar.value()
        value += self.sb_set_offset.value()
        df_emg_cut = [df[value:] for df in self.df_emg]

        df_emg_cut = self.arrange_size(max(df_emg.shape[0] for df_emg in df_emg_cut),df_emg_cut)

        print(df_emg_cut[0].shape)
        print(df_emg_cut[1].shape)
        print(df_emg_cut[2].shape)
        print(df_emg_cut[3].shape)

        print(len([x / 200 for x in range(len(df_emg_cut[0]))]))
        try:
            out_df = pd.DataFrame({
                "Sample": [x / 200 for x in range(len(df_emg_cut[0]))],
                "Biceps_EMG": df_emg_cut[0].iloc[:, 1].values,
                "Biceps_Mat[0][0]": self.Adjust_rate(self.df_imu[0]["Mat[0][0]"], len(df_emg_cut[0])).flatten(),
                "Biceps_Mat[1][0]": self.Adjust_rate(self.df_imu[0]["Mat[1][0]"], len(df_emg_cut[0])).flatten(),
                "Biceps_Mat[2][0]": self.Adjust_rate(self.df_imu[0]["Mat[2][0]"], len(df_emg_cut[0])).flatten(),
                "Biceps_Mat[2][1]": self.Adjust_rate(self.df_imu[0]["Mat[2][1]"], len(df_emg_cut[0])).flatten(),
                "Biceps_Mat[2][2]": self.Adjust_rate(self.df_imu[0]["Mat[2][2]"], len(df_emg_cut[0])).flatten(),
                "Triceps_EMG": df_emg_cut[1].iloc[:, 1].values,
                "Triceps_Mat[0][0]": self.Adjust_rate(self.df_imu[1]["Mat[0][0]"], len(df_emg_cut[0])).flatten(),
                "Triceps_Mat[1][0]": self.Adjust_rate(self.df_imu[1]["Mat[1][0]"], len(df_emg_cut[0])).flatten(),
                "Triceps_Mat[2][0]": self.Adjust_rate(self.df_imu[1]["Mat[2][0]"], len(df_emg_cut[0])).flatten(),
                "Triceps_Mat[2][1]": self.Adjust_rate(self.df_imu[1]["Mat[2][1]"], len(df_emg_cut[0])).flatten(),
                "Triceps_Mat[2][2]": self.Adjust_rate(self.df_imu[1]["Mat[2][2]"], len(df_emg_cut[0])).flatten(),
                "Gastrocnemious_EMG": df_emg_cut[3].iloc[:, 1].values,
                "Gastrocnemious_Mat[0][0]": self.Adjust_rate(self.df_imu[3]["Mat[0][0]"],
                                                             len(df_emg_cut[0])).flatten(),
                "Gastrocnemious_Mat[1][0]": self.Adjust_rate(self.df_imu[3]["Mat[1][0]"],
                                                             len(df_emg_cut[0])).flatten(),
                "Gastrocnemious_Mat[2][0]": self.Adjust_rate(self.df_imu[3]["Mat[2][0]"],
                                                             len(df_emg_cut[0])).flatten(),
                "Gastrocnemious_Mat[2][1]": self.Adjust_rate(self.df_imu[3]["Mat[2][1]"],
                                                             len(df_emg_cut[0])).flatten(),
                "Gastrocnemious_Mat[2][2]": self.Adjust_rate(self.df_imu[3]["Mat[2][2]"],
                                                             len(df_emg_cut[0])).flatten(),
                "Rectus_EMG": df_emg_cut[2].iloc[:, 1].values,
                "Rectus_Mat[0][0]": self.Adjust_rate(self.df_imu[2]["Mat[0][0]"],
                                                     len(df_emg_cut[0])).flatten(),
                "Rectus_Mat[1][0]": self.Adjust_rate(self.df_imu[2]["Mat[1][0]"],
                                                     len(df_emg_cut[0])).flatten(),
                "Rectus_Mat[2][0]": self.Adjust_rate(self.df_imu[2]["Mat[2][0]"],
                                                     len(df_emg_cut[0])).flatten(),
                "Rectus_Mat[2][1]": self.Adjust_rate(self.df_imu[2]["Mat[2][1]"],
                                                     len(df_emg_cut[0])).flatten(),
                "Rectus_Mat[2][2]": self.Adjust_rate(self.df_imu[2]["Mat[2][2]"],
                                                     len(df_emg_cut[0])).flatten(),
            })

            # print(out_df.shape)
            # print(self.df_emg[0].shape)

            # filepath, _ = QFileDialog.getSaveFileName(
            #         self, "Save file", "", "All Files (*);; Text Files (*.txt)"
            #     )

            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Save file",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )

            if filepath != "":
                out_df.to_csv(filepath, index=False)

        except Exception as e:
            print(e)

    def Adjust_rate(self, dataframe,data_len):
        values = dataframe.tolist()
        result = []

        for i, value in enumerate(values):
            result.append(value)
            result.append(value)

        while data_len > len(result):
            result.append("")

        print(np.array(result).shape)
        return np.array(result)

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
