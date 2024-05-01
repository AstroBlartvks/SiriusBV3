from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtGui import QColor
from UI.Form import Ui_MainWindow

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import sys
import os
import math
import json
import threading
import datetime

from resources.database_handler import WorkBook
from resources.database_handler import generate_black_image_2, generate_color_spectrum_image_2
from resources.pipeline_widget2D import PipelineWidget, MplCanvas_ver2, MplHeatmap, PieGraph, LearningGraph
from resources.pipeline_widget2D import save_image_texture
from resources.pipeline_widget3D import CylinderWidget

from model_creator import ModelCreate, Training, temp_handler, is_cuda, save_model_path

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.lstm1 = nn.LSTM(input_size, hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2)
        
        self.fc1 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc2 = nn.Linear(hidden_size // 4, output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        out = self.relu(self.fc1(lstm_out2))
        out = self.softmax(self.fc2(out))
        return out


class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        """
        Главный класс окна программы
        """
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.workbook = WorkBook()                  #WorkBook для работы с excel
        
        self.pipeline_map_3D_created = False        #Костыль для отображения 3Д карты
        self.pipe_3D_created = False                #Костыль для отображения 3D pipe
        self.choosen_pipeline = None                #Переменная класса - выбранный газопровод

        self.pytorch_model = MyModel(input_size=7, hidden_size=32, output_size=5)
        self.pytorch_loaded = False
        self.nn_model = None                        #Модель нейронной сети
        self.optimizer = None                       #Оптимизатор
        self.dataset_path = None                    #Путь до базы данных
        self.model_sequence = []                    #Настриваемая модель

        if not is_cuda():                           #Проверка наличия cuda
            self.ui.comboBox_7.setEnabled(False)
            self.ui.comboBox_6.setEnabled(False)
            self.ui.label_23.setText("False")

        self.navigate_buttons()                     #Подключение навигационных кнопок


    def load_nn_model(self):
        self.dir_model = QtWidgets.QFileDialog.getExistingDirectory(self,"Выбрать папку",".") + "\\"

        with open(self.dir_model+"model_temp_save.json", "r") as read_file:
            data = json.load(read_file)
        
        self.model_sequence = data[:-1] 
        self.optimizer = data[-1]["optim"]
        self.nn_model = ModelCreate(self.model_sequence)
        self.nn_model.load_state_dict(torch.load(self.dir_model+"model.pt"))
        self.ui.label_45.setText(f"Загружена | {str(datetime.datetime.now())[11:19]}")
        self.ui.plainTextEdit_6.setPlainText(self.dir_model+"\n"+str(self.nn_model))
        self.pytorch_loaded = True


    def start_training(self):
        if self.dataset_path is None:
            self.show_critical_msg("Датасет", "Датасет не был загружен")
            return
        
        device = "cuda:0" if "CUDA" in [self.ui.comboBox_7.currentText(), self.ui.comboBox_6.currentText()] else "cpu"
        self.Training_class = Training(self.nn_model, self.model_sequence[0][1], self.model_sequence[-1][2], self.dataset_path, device)
        self.Training_class.prepare_dataset()
        
        verbs = self.Training_class.train(self.optimizer, float(self.ui.lineEdit_19.text()), int(self.ui.lineEdit_20.text()))

        if isinstance(self.ui.widget_6, QtWidgets.QVBoxLayout):
            self.ui.widget_6.removeWidget(self.graph_learning_hist)

        self.graph_learning_hist = LearningGraph([list(range(len(self.Training_class.global_history[0]))), list(range(len(self.Training_class.global_history[0]))), list(range(len(self.Training_class.global_history[0])))]
                                , self.Training_class.global_history, ["Test", "Valid", "Train"], orientation="h", labelXY = ["Номер сегмента", "Значение"])

        if not isinstance(self.ui.widget_6, QtWidgets.QVBoxLayout):
            self.ui.widget_6 = QtWidgets.QVBoxLayout(self.ui.widget_6)
            self.ui.widget_6.setObjectName("widget_6")

        self.ui.widget_6.addWidget(self.graph_learning_hist)
    
        self.ui.listWidget_3.clear()
        self.ui.listWidget_3.addItems(verbs)

        loss_test = 1 - sum(self.Training_class.global_history[0]) / len(self.Training_class.global_history[0])
        loss_valid = 1 - sum(self.Training_class.global_history[1]) / len(self.Training_class.global_history[1])
        loss_train = 1 - sum(self.Training_class.global_history[2]) / len(self.Training_class.global_history[2])
        loss = (loss_test + loss_valid + loss_train) / 3
        self.ui.label_8.setText(f"Среднеквадратичная: {round(loss, 5)}\nТочность: {round(100*(loss**1.3))}%")

        self.nn_model = self.Training_class.model
     

    def load_dataset(self):
        try:
            self.dataset_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выбрать файл для обучения",".", "Excel (*.xlsx);CSV (*.csv);All Files (*)")
            self.ui.plainTextEdit_5.setPlainText(str(self.dataset_path))
        except Exception as exp:
            self.show_critical_msg("Ошибка при загрузке датасета", str(exp))
            self.dataset_path = None


    def save_model(self, path=False, restart=False):
        if [0, 0, 0, 0] in self.model_sequence:
            self.show_critical_msg("Нейросеть ошибка", "Неполная модель, не все слои заполнены")
            self.ui.label_45.setText(f"Нет")
            return

        self.ui.label_45.setText(f"Да | {str(datetime.datetime.now())[11:19]}")
        if self.nn_model is None or restart:
            self.nn_model = ModelCreate(self.model_sequence)

        if path:
            self.dir_model = QtWidgets.QFileDialog.getExistingDirectory(self,"Выбрать папку",".") + "\\"
        else:
            self.dir_model = os.getcwd() + "\\"

        with open(self.dir_model+"model_temp_save.json", "w") as write_file:
            json.dump(self.model_sequence + [{"optim": self.ui.comboBox_15.currentText()}], write_file)
        
        save_model_path(self.nn_model, self.dir_model)
        self.optimizer = self.ui.comboBox_15.currentText()

        self.ui.plainTextEdit_6.setPlainText("Сохранено в папке:\n"+str(self.dir_model)+"\n"+str(self.nn_model))
        self.pytorch_loaded = True


    def load_layer(self):
        index = self.ui.comboBox_12.currentIndex()
        layer = self.model_sequence[index]
        self.ui.lineEdit_17.clear()
        self.ui.lineEdit_18.clear()
        self.ui.comboBox_13.setCurrentIndex(0)
        self.ui.comboBox_14.setCurrentIndex(0)

        if index > 0 and self.model_sequence[index - 1][1] != 0:
            self.ui.lineEdit_17.setText(str(self.model_sequence[index - 1][2]))
        elif index == 0:
            self.ui.lineEdit_17.setText(str(self.ui.lineEdit_21.text()))

        if layer == [0, 0, 0, 0]: return
        self.ui.comboBox_13.setCurrentIndex({"FC": 0, "RNN": 1, "LSTM": 2, "GRU": 3}[self.model_sequence[index][0]])
        self.ui.comboBox_14.setCurrentIndex({"Sigmoid": 0, "ReLU": 1, "Tanh": 2, "Softmax": 3}[self.model_sequence[index][3]])
        self.ui.lineEdit_17.setText(str(self.model_sequence[index][1]))
        self.ui.lineEdit_18.setText(str(self.model_sequence[index][2]))


    def save_layer(self):
        index = self.ui.comboBox_12.currentIndex()
        type_layer = self.ui.comboBox_13.currentText()
        input_ = int(self.ui.lineEdit_17.text())
        output_ = int(self.ui.lineEdit_18.text())
        type_activ = self.ui.comboBox_14.currentText()
        self.model_sequence[index][0] = type_layer
        self.model_sequence[index][1] = input_
        self.model_sequence[index][2] = output_
        self.model_sequence[index][3] = type_activ


    def create_model(self):
        self.model_sequence = list([[0, 0, 0, 0] for _ in range(int(self.ui.comboBox_16.currentText()))])
        self.ui.comboBox_12.clear()
        self.ui.comboBox_12.addItems(list([str(i+1) for i in range(len(self.model_sequence))]))
        self.ui.comboBox_12.setCurrentIndex(0)


    def use_nn_model(self):
        try:
            if not self.pytorch_loaded:
                self.show_critical_msg("Ошибка", "Модель не загружена")
                return

            values = torch.Tensor(list(map(float, [self.ui.tableWidget_3.item(0, x).text() for x in range(self.ui.tableWidget_3.columnCount())]))).unsqueeze(0).to("cuda:0" if self.ui.comboBox_6.currentText() == "CUDA" else "cpu")
            output = list([round(x.item(), 5) for x in self.nn_model(values)[0]])
            num_1 = "1. Cвяз. с производством\n(Потеря металла)\n"+str(output[0])
            num_2 = "2. Отложения\n(Геометрия)\n"+str(output[1])
            num_3 = "3. Внешние воздействия\n(Потеря металла)\n"+str(output[2])
            num_4 = "4. Действия человека\n(Изменение толщ. ст.)\n"+str(output[3])
            num_5 = "5. Нет особенностей\n(Дефектов не обнаружено)\n"+str(output[4])
            self.ui.listWidget_2.clear()
            self.ui.listWidget_2.addItems([num_1, num_2, num_3, num_4, num_5])
            self.ui.listWidget_2.item(output.index(max(output))).setBackground(QColor("red"))

            chances = list([round(x, 5) for x in output])
            labels = ["Cвяз. с производством", "Отложения", "Внешние воздействия", "Действия человека", "Нет особенностей"]

            if isinstance(self.ui.widget_5, QtWidgets.QVBoxLayout):
                self.ui.widget_5.removeWidget(self.pie_graph_1)

            self.pie_graph_1 = PieGraph(labels, chances)

            if not isinstance(self.ui.widget_5, QtWidgets.QVBoxLayout):
                self.ui.widget_5 = QtWidgets.QVBoxLayout(self.ui.widget_5)
                self.ui.widget_5.setObjectName("widget_pie_graph_1")

            self.ui.widget_5.addWidget(self.pie_graph_1)


        except Exception as exp:
            if "nonetype" in str(exp).lower() and "text" in str(exp).lower():
                self.show_critical_msg("Ошибка", "Укажите все входные значения")
            if "with base 10" in str(exp).lower():
                self.show_critical_msg("Ошибка", "Укажите все числовые значения в формате (число.послезапятой)")
            else:
                self.show_critical_msg("Ошибка", str(exp))

        
    def load_excel_database(self):
        try:
            """
            Загрузка excel файла и его обработка
            """
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", ".", "Excel Files (*.xlsx);;All Files (*)")
            self.workbook.open(filename)            # Открытие файла
            self.workbook.get_info()                # Обработка файла, получение информации
            info_base = self.workbook.info_base

            self.ui.listWidget.clear()
            for info in info_base:                  # Выводл информации о трубопроводе
                if info == "Условия во время нормального режима":
                    self.ui.listWidget.addItem(self.workbook.get_info_string_1())
                elif info == "Условия во время инспекции:":
                    self.ui.listWidget.addItem(self.workbook.get_info_string_2())
                else:
                    self.ui.listWidget.addItem(info+": "+info_base[info])

            self.workbook.process_the_db()          # Обработка файла, обработка базы данных
            self.set_info_table()
            max_number = len(self.workbook.db[0])
            self.ui.spinBox.setMinimum(0)
            self.ui.spinBox.setMaximum(max_number - 1)
            self.draw_graph_1()
            self.draw_graph_2() 
            self.draw_graph_3()
            self.draw_graph_4() 
        except Exception as exp:
            self.show_critical_msg("Ошибка", str(exp))


    def set_info_table(self):                   # Загрузка таблицы
        try:
            self.ui.tableWidget_2.setRowCount(len(self.workbook.db[0]))
            for x in range(self.ui.tableWidget_2.rowCount()):
                for y in range(self.ui.tableWidget_2.columnCount()):
                    self.ui.tableWidget_2.setItem(x, y, QTableWidgetItem(self.workbook.db[y][x]))
                    if self.workbook.db[7][x] != "":
                        self.ui.tableWidget_2.item(x, y).setBackground(QColor(128, 0, 0))
        except Exception as exp:
            self.show_critical_msg("Ошибка", str(exp))


    def draw_pipe(self, LENGTH, RADIUS, WIDTH, texture_is=None):
        colors = vtk.vtkNamedColors()
        points = vtk.vtkPoints()
        idx = 0
        angle = 0
        while angle / 60 <= 2.0 * vtk.vtkMath.Pi() + (vtk.vtkMath.Pi() / 60.0):
            points.InsertNextPoint((RADIUS+WIDTH) * math.cos(angle),
                                (RADIUS+WIDTH) * math.sin(angle),
                                0.0)
            angle = angle + (vtk.vtkMath.Pi() / 60.0)
            idx += 1

        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(idx)
        for i in range(0, idx):
            line.GetPointIds().SetId(i, i)

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line)

        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(lines)

        extrude = vtk.vtkLinearExtrusionFilter()
        extrude.SetInputData(polyData)
        extrude.SetExtrusionTypeToNormalExtrusion()
        extrude.SetVector(0, 0, LENGTH)
        extrude.Update()

        texture_map = vtk.vtkTextureMapToCylinder()
        texture_map.SetInputConnection(extrude.GetOutputPort())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(texture_map.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        texture = vtk.vtkTexture()
        reader = vtk.vtkPNGReader()
        reader.SetFileName(texture_is)
        texture.SetInputConnection(reader.GetOutputPort())
        actor.SetTexture(texture)
        actor.GetProperty().SetLighting(False)  # Отключить освещение для актора

        ren = vtk.vtkRenderer()
        ren.SetBackground(colors.GetColor3d("SlateGray"))
        ren.AddActor(actor)

        return ren, actor


    def show_3d_pipe(self):                     # 3Д представление трубопровода по частям
        try:
            if self.pipe_3D_created:
                pipe_3D2 = QVTKRenderWindowInteractor(self.pipe_3D)
                self.ui.verticalLayout_9.removeWidget(self.pipe_3D)
                self.pipe_3D = pipe_3D2
            else:
                self.pipe_3D_created = True
                self.pipe_3D = QVTKRenderWindowInteractor(self.ui.VTKFrame)

            pipe_1, actor_1 = self.draw_pipe(5000, 76, 0, ".\\resources\\image\\inner_textures.png")
            pipe_2, actor_2 = self.draw_pipe(5000, 76, 12.6, ".\\resources\\image\\outer_textures.png")
            pipe_1.AddActor(actor_2)

            self.pipe_3D.GetRenderWindow().AddRenderer(pipe_1)
            self.iren = self.pipe_3D.GetRenderWindow().GetInteractor()
            style = vtk.vtkInteractorStyleTrackballCamera()
            self.iren.SetInteractorStyle(style)

            self.show()
            self.iren.Initialize()
            self.iren.Start()
            self.ui.verticalLayout_9.addWidget(self.pipe_3D)
        except Exception as exp:
            self.show_critical_msg("Ошибка", str(exp))


    def draw_graph_4(self):                     # 3Д представление трубопровода по частям
        try:
            pipelines = self.create_pipeline()

            if self.pipeline_map_3D_created:
                self.ui.verticalLayout_2.removeWidget(self.pipeline_map_3D)
            else:
                self.ui.verticalLayout_2.removeWidget(self.ui.openGLWidget)
                self.pipeline_map_3D_created = True

            self.pipeline_map_3D = CylinderWidget(pipelines, self.ui.tableWidget, self.choosen_pipeline)
            self.ui.verticalLayout_2.addWidget(self.pipeline_map_3D)
        except Exception as exp:
            self.show_critical_msg("Ошибка", str(exp))


    def draw_graph_3(self):                     # График 2Д с трубопроводами
        try:
            pipelines = self.create_pipeline()
            
            if not isinstance(self.ui.widget_3, QtWidgets.QVBoxLayout):
                self.pipeline_widget = PipelineWidget(pipelines, self.ui.tableWidget, self.choosen_pipeline, self)
                self.ui.widget_3 = QtWidgets.QVBoxLayout(self.ui.widget_3)
                self.ui.widget_3.setObjectName("widget_3")
            else:
                self.ui.widget_3.removeWidget(self.pipeline_widget)
                self.pipeline_widget = PipelineWidget(pipelines, self.ui.tableWidget, self.choosen_pipeline, self)
        
            self.ui.widget_3.addWidget(self.pipeline_widget)
        except Exception as exp:
            self.show_critical_msg("Ошибка", str(exp))


    def draw_graph_2(self):                     # График потерь металла
        try:
            if isinstance(self.ui.widget_2, QtWidgets.QVBoxLayout):
                self.ui.widget_2.removeWidget(self.chart_loss_walls)

            ys = list([float(self.workbook.db[8][x]) if self.workbook.db[8][x] != "" else 0.0 for x in range(len(self.workbook.db[8]))])
            xs = list(range(len(ys)))

            ys2 = list([float(self.workbook.db[7][x]) if self.workbook.db[7][x] != "" else 0.0 for x in range(len(self.workbook.db[7]))])
            xs2 = list(range(len(ys)))

            self.chart_loss_walls = MplCanvas_ver2([xs, xs2], [ys, ys2], ["Процент потерь", "Потеря металла"], orientation="h",
                                                labelXY = ["Номер сегмента", "Значение"])

            if not isinstance(self.ui.widget_2, QtWidgets.QVBoxLayout):
                self.ui.widget_2 = QtWidgets.QVBoxLayout(self.ui.widget_2)
                self.ui.widget_2.setObjectName("widget_1")

            self.ui.widget_2.addWidget(self.chart_loss_walls)
        except Exception as exp:
            self.show_critical_msg("Ошибка", str(exp))


    def draw_graph_1(self):                     # График толщины стенки
        try:
            if isinstance(self.ui.widget, QtWidgets.QVBoxLayout):
                self.ui.widget.removeWidget(self.chart_wide_walls)

            ys = list([float(self.workbook.db[6][x]) if self.workbook.db[6][x] != "" else (float(self.workbook.db[5][x]) if self.workbook.db[5][x] != "" else float(self.workbook.db[5][x-1])) for x in range(len(self.workbook.db[6]))])
            xs = list(range(len(ys)))

            ys2 = list([float(self.workbook.db[5][x]) if self.workbook.db[5][x] != "" else float(self.workbook.db[5][x-1]) for x in range(len(self.workbook.db[5]))])
            xs2 = list(range(len(ys)))

            self.chart_wide_walls = MplCanvas_ver2([xs, xs2], [ys, ys2], ["Ост. толщина", "Контр. толщина"], orientation="h",
                                                labelXY = ["Номер сегмента", "Значение"])

            if not isinstance(self.ui.widget, QtWidgets.QVBoxLayout):
                self.ui.widget = QtWidgets.QVBoxLayout(self.ui.widget)
                self.ui.widget.setObjectName("widget_1")

            self.ui.widget.addWidget(self.chart_wide_walls)
        except Exception as exp:
            self.show_critical_msg("Ошибка", str(exp))
        

    def create_pipeline(self):
        try:
            """
            Создание батча pipeline (трубопровод)
            """
            sector_1 = self.workbook.db[0]
            sector_2 = self.workbook.db[2]
            sector_3 = list([self.workbook.db[5][x] if self.workbook.db[5][x] != "" else self.workbook.db[5][x - 1] for x in range(len(self.workbook.db[5]))])
            sector_4 = list([self.workbook.db[6][x] if self.workbook.db[6][x] != "" else self.workbook.db[5][x - 1] for x in range(len(self.workbook.db[6]))])
            sector_5 = list([float(self.workbook.db[7][x]) if self.workbook.db[7][x] != "" else 0.0 for x in range(len(self.workbook.db[7]))])
            sector_6 = self.workbook.db[12]
            return [sector_1, sector_2, sector_3, sector_4, sector_5, sector_6]
        except Exception as exp:
                self.show_critical_msg("Ошибка", str(exp))


    def changed_pipeline(self):
        try:
            number_id = self.ui.spinBox.text()
            if number_id == "": return 
            number_id = int(number_id)
            if number_id == 0: return
            self.ui.tableWidget.setItem(0, 2, QtWidgets.QTableWidgetItem(self.workbook.db[5][number_id]))
            loss_wall = self.workbook.db[6][number_id]
            loss_wall = float(loss_wall) if loss_wall != "" else float(self.workbook.db[5][number_id])
            self.ui.tableWidget.setItem(0, 3, QtWidgets.QTableWidgetItem(str(loss_wall)))
            loss = round(abs(float(self.workbook.db[5][number_id]) - loss_wall), 2)
            self.ui.tableWidget.setItem(0, 4, QtWidgets.QTableWidgetItem(str(loss)))
            self.ui.tableWidget.setItem(0, 5, QtWidgets.QTableWidgetItem(self.workbook.db[12][number_id]))
            self.ui.tableWidget.setItem(0, 1, QtWidgets.QTableWidgetItem(self.workbook.db[2][number_id]))
            self.ui.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem(self.workbook.db[0][number_id] + ":" + self.workbook.db[1][number_id]))
            self.choosen_pipeline = number_id

            diametr = float(self.workbook.info_base["Диаметр трубопровода"]) * 1000
            image_height = round(1000 * (float(self.workbook.db[2][self.choosen_pipeline+1]) - float(self.workbook.db[2][self.choosen_pipeline-1])))
            image_width = round(3.1413 * diametr)
            tube_depth_percent = self.workbook.db[8][self.choosen_pipeline]
            if not (tube_depth_percent == ''): 
                tube_depth_percent = float(tube_depth_percent) / 100
                tube_length = round(float(self.workbook.db[10][self.choosen_pipeline]))
                tube_width = round(float(self.workbook.db[11][self.choosen_pipeline]))
                spot_angle = float(self.workbook.db[4][self.choosen_pipeline])
                spot_start_x = round(1000 * (float(self.workbook.db[2][self.choosen_pipeline]) - float(self.workbook.db[2][self.choosen_pipeline-1])))
                wall = self.workbook.db[5][self.choosen_pipeline]
                wall = float(self.workbook.db[5][self.choosen_pipeline - 1]) if wall == "" else float(wall)
                image = generate_color_spectrum_image_2(image_width, image_height, tube_length, tube_width, wall, tube_depth_percent, spot_angle, spot_start_x, diametr)
            else:
                image = generate_black_image_2(image_width, image_height)

            thread_temp = threading.Thread(target=save_image_texture(image,))
            thread_temp.start()

            if not isinstance(self.ui.widget_4, QtWidgets.QVBoxLayout):
                self.mplimgcanvas = MplHeatmap(image)
                self.ui.widget_4 = QtWidgets.QVBoxLayout(self.ui.widget_4)
                self.ui.widget_4.setObjectName("widget_4")
            else:
                self.ui.widget_4.removeWidget(self.mplimgcanvas)
                self.mplimgcanvas = MplHeatmap(image)

                self.ui.widget_4.addWidget(self.mplimgcanvas)
            
            thread_temp.join()
            self.show_3d_pipe()
        except Exception as exp:
            self.show_critical_msg("Ошибка", str(exp))


    def find_offset_Follias(self):
        try:
            Length = float(self.workbook.db[10][self.choosen_pipeline])
            wall = float(self.workbook.db[5][self.choosen_pipeline] if self.workbook.db[6][self.choosen_pipeline] == "" else self.workbook.db[6][self.choosen_pipeline])
            radius = float(self.workbook.info_base["Диаметр трубопровода"]) / 2
            Offset = (1 + float(self.ui.lineEdit_6.text().replace(",", ".")) * ((Length * 0.001 / 2) * (Length * 0.001 / 2)) / (radius*wall))
            self.ui.lineEdit_7.setText(f"Коэффицент Фолиаса: {Offset:.2f}")
        except Exception as exp:
            self.show_critical_msg("Ошибка", str(exp))


    def find_total_length_pipe_defects(self):
        try:
            diametr = float(self.workbook.info_base["Диаметр трубопровода"])
            wall_wide = list([float(self.workbook.db[5][i]) for i in range(len(self.workbook.db[5])) if self.workbook.db[5][i] != ""])
            wall_wide = sum(wall_wide) / len(wall_wide)
            K_delta_const = 10 #default = 14
            K_delta = K_delta_const / (wall_wide * diametr)
            indexs = list([x for x in range(len(self.workbook.db[7])) if self.workbook.db[7][x] != ""])
            all_sum = 0

            for i in indexs:
                length = float(self.workbook.db[10][i])
                wide = float(self.workbook.db[11][i])
                deep = float(self.workbook.db[7][i])
                wide_state = ((math.ceil(wide / 100) - 1) if (math.ceil(wide / 100) - 1) < 5 else 4) + 1
                if deep < 0.5: deep_state = 0.5
                elif 0.5 < deep < 1: deep_state = 1
                elif 1 < deep < 2: deep_state = 2
                elif 2 < deep < 3: deep_state = 3
                elif 3 < deep < 4: deep_state = 4
                elif deep > 4: deep_state = 5

                K_pn = 1
                K_rn = wide_state * deep_state
                all_sum += K_pn * K_rn * length
            
            Total_length_pipe_defects_mm = K_delta * all_sum
            self.ui.lineEdit_12.setText(f"Суммарная приведенная длина дефектов: {Total_length_pipe_defects_mm:.5f} mm; {(Total_length_pipe_defects_mm/1000):.2f} m")
        except Exception as exp:
            self.show_critical_msg("Ошибка", str(exp))


    def navigate_buttons(self):
        self.ui.pushButton.clicked.connect(self.load_excel_database)
        self.ui.pushButton_2.clicked.connect(lambda: self.nav_button(1))
        self.ui.pushButton_3.clicked.connect(lambda: self.nav_button(2))
        self.ui.pushButton_4.clicked.connect(lambda: self.nav_button(0))

        self.ui.pushButton_6.clicked.connect(self.find_offset_Follias)
        self.ui.pushButton_8.clicked.connect(self.find_total_length_pipe_defects)

        self.ui.pushButton_9.clicked.connect(self.use_nn_model)
        self.ui.pushButton_10.clicked.connect(self.load_nn_model)
        self.ui.pushButton_53.clicked.connect(self.create_model)
        self.ui.pushButton_50.clicked.connect(self.save_layer)
        self.ui.pushButton_51.clicked.connect(lambda: self.save_model(False, True))
        self.ui.pushButton_56.clicked.connect(lambda: self.save_model(True, False))
        self.ui.pushButton_57.clicked.connect(self.load_nn_model)

        self.ui.pushButton_55.clicked.connect(self.load_dataset)
        self.ui.pushButton_52.clicked.connect(self.start_training)

        self.ui.spinBox.editingFinished.connect(lambda: self.changed_pipeline())
        self.ui.comboBox_12.currentIndexChanged.connect(self.load_layer)
    

    def show_critical_msg(self, Title, Text):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(Text) 
        msg.setWindowTitle(Title) 
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        retval = msg.exec_()  


    def show_information_msg(self, Title, Text):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information) 
        msg.setText(Text) 
        msg.setWindowTitle(Title) 
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        retval = msg.exec_()  


    def nav_button(self, indx):
        self.ui.stackedWidget.setCurrentIndex(indx)
        


def main():
    app = QtWidgets.QApplication([])
    application = mywindow()
    application.show()
    
    sys.exit(app.exec())

main()
