import sys
from PyQt5.QtWidgets import QOpenGLWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *

class CylinderWidget(QOpenGLWidget):
    def __init__(self, pipelines, tablewidget, pipeline_choosen):
        super().__init__()
        self.spacing = 0.3 # Расстояние между трубами
        self.camera_x = 0.5  # Положение камеры по оси X
        self.speed = 0.1
        self.camera_y = 2
        self.pipelines = pipelines
        self.tablewidget = tablewidget
        self.pipeline_choosen = pipeline_choosen

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.9, 0.9, 0.9, 1.0)
        self.grabKeyboard()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w/h, 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(self.camera_x, self.camera_y, 0, self.camera_x, 0, 0, 0, 0, -1)

        for i in range(len(self.pipelines[0])):
            self.drawCylinder(i * (1 + self.spacing), self.pipelines[4][i])

    def drawCylinder(self, x, loss):
        quadric = gluNewQuadric()

        if loss != 0.0:
            glColor3f(1.0, 0.0, 0.0)
        else:
            glColor3f(0.0, 1.0, 0.0)

        glPushMatrix()
        glTranslatef(x, 0, 0)
        glRotatef(90, 0, 1, 0)

        gluCylinder(quadric, 0.2, 0.2, 1, 20, 20)

        glTranslatef(0, 0, 1)
        gluDisk(quadric, 0, 0.2, 20, 20)

        glPopMatrix()

        gluDeleteQuadric(quadric)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A and self.camera_x > 0.5:
            self.camera_x -= self.speed
        elif event.key() == Qt.Key_D and self.camera_x < len(self.pipelines[0]) * (1 + self.spacing) - 0.5:
            self.camera_x += self.speed
        if event.key() == Qt.Key_W and self.camera_y > 1.8:
            self.camera_y -= self.speed
        elif event.key() == Qt.Key_S and self.camera_y < 4.0:
            self.camera_y += self.speed

        self.speed = 0.1 + 0.5 * self.camera_y
        self.update()

    def mousePressEvent(self, event):
        for i in range(len(self.pipelines[0])):
            x_cylinder = i * (1 + self.spacing)
            if x_cylinder - self.spacing <= self.camera_x <= x_cylinder + 1 + self.spacing:
                for j in range(6):
                    self.tablewidget.setItem(0, j, QTableWidgetItem(str(self.pipelines[j][i])))
                self.pipeline_choosen = i
                break
