from PyQt5 import QtWidgets, QtWebEngineWidgets, QtCore
from PyQt5.QtWidgets import QTableWidgetItem, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QFileDialog
from PyQt5.QtGui import QColor, QPen, QPainter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os,inspect 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import plotly.graph_objs as graph_objs
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class Registry:
    def __init__(self):
        self.C = None #Временных данных
        self.R = None #Данных
        self.D = None #Много данных

class TempVariables:
    def __init__(self):
        self.RAX = Registry()
        self.RBX = Registry()
        self.RCX = Registry()
        self.REX = Registry() #Для возврата

VARIABLEBASE = TempVariables()

def _on_downloadFunc(item):
    print('downloading to', item.path())
    item.accept()

def save_image_texture(image):
    image = np.rot90(image)
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=image.min(), vmax=image.max())
    image_data = (cmap(norm(image)) * 255).astype(np.uint8)
    image = Image.fromarray(image_data)
    image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
    image.save("resources/image/inner_textures.png")

class MplCanvas_ver2(QtWidgets.QWidget):
    def __init__(self, xs, ys, names, activation_func = None, labelXY = None, orientation="v", mode='lines+markers', parent=None):
        super().__init__(parent)
        self.browser = QtWebEngineWidgets.QWebEngineView(self)
        self.browser.page().profile().downloadRequested.connect(_on_downloadFunc)

        vlayout = QtWidgets.QVBoxLayout(self)
        vlayout.addWidget(self.browser)

        self.xs = xs
        self.ys = ys
        self.names = names
        self.orientation = orientation
        self.activation_func = activation_func
        self.mode = mode 
        self.labelXY = labelXY
        self.show_graph()

    def show_graph(self):
        fig = graph_objs.Figure()
        for i in range(len(self.xs)):
            fig.add_trace(graph_objs.Scatter(x=self.xs[i], y=self.ys[i], mode=self.mode, name=self.names[i]))
        fig.update_layout(legend_orientation=self.orientation,
                          legend=dict(x=.5, xanchor="center"),
                          margin=dict(l=0, r=0, t=0, b=0))
        if not (self.labelXY is None):
            fig.update_traces(hoverinfo="all", hovertemplate=self.labelXY[0] + ": %{x}<br>"+self.labelXY[1]+": %{y}")
        
        self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))


class LearningGraph(QtWidgets.QWidget):
    def __init__(self, xs, ys, names, activation_func = None, labelXY = None, orientation="v", mode='lines+markers', parent=None):
        super().__init__(parent)
        self.browser = QtWebEngineWidgets.QWebEngineView(self)
        self.browser.page().profile().downloadRequested.connect(_on_downloadFunc)

        vlayout = QtWidgets.QVBoxLayout(self)
        vlayout.addWidget(self.browser)

        self.xs = xs
        self.ys = ys
        self.names = names
        self.orientation = orientation
        self.activation_func = activation_func
        self.mode = mode 
        self.labelXY = labelXY
        self.show_graph()

    def show_graph(self):
        fig = graph_objs.Figure()
        for i in range(len(self.xs)):
            fig.add_trace(graph_objs.Scatter(x=self.xs[i], y=self.ys[i], mode=self.mode, name=self.names[i]))
        fig.update_layout(legend_orientation=self.orientation,
                          legend=dict(x=.5, xanchor="center"),
                          margin=dict(l=0, r=0, t=0, b=0))
        if not (self.labelXY is None):
            fig.update_traces(hoverinfo="all", hovertemplate=self.labelXY[0] + ": %{x}<br>"+self.labelXY[1]+": %{y}")
        
        self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))


class MplHeatmap(QtWidgets.QWidget):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.browser = QtWebEngineWidgets.QWebEngineView(self)
        self.browser.page().profile().downloadRequested.connect(_on_downloadFunc)

        vlayout = QtWidgets.QVBoxLayout(self)
        vlayout.addWidget(self.browser)

        fig = graph_objs.Figure(graph_objs.Heatmap(z = image, colorscale='Viridis'))
        HTML = fig.to_html()
        
        main_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        with open(os.path.join(main_directory,"heatmap.html"), "w", encoding="utf-8") as File: File.write(HTML)
        url = QtCore.QUrl.fromLocalFile(os.path.join(main_directory,"heatmap.html"))
        self.browser.load(url)
        #self.browser.setHtml(HTML)


class PieGraph(QtWidgets.QWidget):
    def __init__(self, labels, values, parent=None):
        super().__init__(parent)
        self.browser = QtWebEngineWidgets.QWebEngineView(self)
        self.browser.page().profile().downloadRequested.connect(_on_downloadFunc)

        vlayout = QtWidgets.QVBoxLayout(self)
        vlayout.addWidget(self.browser)

        self.labels = labels
        self.values = values
        self.show_graph()

    def show_graph(self):
        fig = go.Figure(data=[go.Pie(labels=self.labels, values=self.values, hole=.2)])
        self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))


class MplImgCanvas_ver2(QtWidgets.QWidget):
    def __init__(self, image, image_length, image_width, parent=None):
        super().__init__(parent)
        self.image = image
        self.browser = QtWebEngineWidgets.QWebEngineView(self)
        self.browser.page().profile().downloadRequested.connect(_on_downloadFunc)

        vlayout = QtWidgets.QVBoxLayout(self)
        vlayout.addWidget(self.browser)
        fig = graph_objs.Figure()

        scale_factor = 0.5
        fig.add_layout_image(
                x=0,
                sizex=image_width,
                y=image_length,
                sizey=image_length,
                xref="x",
                yref="y",
                opacity=1.0,
                source=image
        )
        fig.update_xaxes(showgrid=False)#, range=(0, image_width))
        fig.update_yaxes(showgrid=False)#, range=(0, image_length))

        self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))


class PipelineSection(QGraphicsLineItem):
    def __init__(self, x1, y1, x2, y2, scene, tablewidget, choosen_pipeline, info=None):
        super().__init__(x1, y1, x2, y2)
        self.number = x1//50
        self.is_faulty = True if info[4] != 0.0 else False
        self.scene = scene
        self.info = info
        self.tablewidget = tablewidget

        pen = QPen()
        pen.setWidth(5)
        if self.is_faulty:
            pen.setColor(QColor(255, 0, 0))
        else:
            pen.setColor(QColor(0, 255, 0))

        self.setPen(pen)

        self.start_marker = QGraphicsLineItem(x1-3, y1-4, x1-3, y1+4)
        self.end_marker = QGraphicsLineItem(x2-3, y2-4, x2-3, y2+4)

        self.scene.addItem(self.start_marker)
        self.scene.addItem(self.end_marker)

    def mousePressEvent(self, event):
        for i in range(6):
            self.tablewidget.setItem(0, i, QTableWidgetItem(str(self.info[i])))
        self.choosen_pipeline = self.number

class PipelineWidget(QGraphicsView):
    def __init__(self, pipelines, tablewidget, choosen_pipeline, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.tablewidget = tablewidget
        self.choosen_pipeline = choosen_pipeline

        for pipeid in range(len(pipelines[0])):
            info = list([pipelines[x][pipeid] for x in range(6)])
            section = PipelineSection(pipeid * 50, 0, (pipeid + 1) * 50, 0, self.scene, self.tablewidget, self.choosen_pipeline, info=info)
            self.scene.addItem(section)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.HighQualityAntialiasing, True)

    def wheelEvent(self, event):
        factor = 1.2
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor
        self.scale(factor, factor)