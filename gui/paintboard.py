import sys

from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import QSize, Qt, QPoint
from PyQt5.QtGui import QPixmap, QColor, QPainter, QPen


class PaintBoard(QWidget):
    def __init__(self, Parent = None, size = QSize(320, 240), fill_bg_color = QColor(255, 255, 255, 255)):
        super().__init__(Parent)
        self.__size = size
        self.__fill = fill_bg_color

        # 新建画布
        self.__board = QPixmap(self.__size)
        self.__board.fill(self.__fill)

        # 新建绘图工具
        self.__painter = QPainter()
        # 设置画笔起始点
        self.__begin_point = QPoint()
        self.__end_point = QPoint()

        #其他一些绘图相关的设置
        self.__thickness = 18  # 默认画笔粗细
        self.__penColor = QColor(0, 0, 0, 255)  # 默认画笔颜色

        # 设置QWidget参数
        self.setFixedSize(self.__size)

    def getContentAsQImage(self):
        image = self.__board.toImage()
        return image

    def setBoardFill(self, fill):
        self.__fill = fill
        self.__board.fill(fill)
        self.update()

    # 设置画笔颜色
    def setPenColor(self, color):
        self.__penColor = color

    # 设置画笔粗细
    def setPenThickness(self, thickness=10):
        self.__thickness = thickness

    # 下面这些方法都是overwrite

    # 如果没有它的话，所有画的内容都是自动连在一起的，也就是说start point永远都是上一次画的结束位置，而不是自动感知到目前press的位置
    def mousePressEvent(self,event):
        if event.button() == Qt.LeftButton:
            self.__begin_point = event.pos()
            self.__end_point = event.pos()

    def paintEvent(self, e):
        # QPainter的begin方法使得绘画开始，它的参数表示painting在那里进行，这里的话就是在目前这个QWidget上进行
        # 一旦 .begin()，所有跟QPainter有关的设置都会重置
        self.__painter.begin(self)
        # 重载函数.drawPixmap(x, y, pm)，把pm画在(x,y)位置
        self.__painter.drawPixmap(0, 0, self.__board)
        self.__painter.end()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton:
            self.__end_point = e.pos()

            self.__painter.begin(self.__board)
            self.__painter.setPen(QPen(self.__penColor, self.__thickness))
            self.__painter.drawLine(self.__begin_point, self.__end_point)
            self.__painter.end()

            self.__begin_point = self.__end_point
            self.update()


    def Clear(self):
        self.__board.fill(self.__fill)
        self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PaintBoard()
    window.show()
    sys.exit(app.exec_())
