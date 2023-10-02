import os.path
import sys
import hashlib
import time
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
from tensorflow import keras
from keras_preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont


class MainWindows(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.EMOTIONS = ["生气", "厌恶", "害怕", "开心", "悲伤", "惊讶", "正常"]
        self.detection_model_path = 'module/haarcascade_frontalface_default.xml'  # 人脸检测模型
        self.emotion_model_path = 'module/mini_XCEPTION.102-0.66.hdf5'
        self.face_detection = cv2.CascadeClassifier('module/haarcascade_frontalface_default.xml')
        self.emotion_classifier = keras.models.load_model('module/mini_XCEPTION.102-0.66.hdf5', compile=False)
        self.shows = None
        self.name_hash = None
        self.showImage = None
        self.pix = None
        self.id = None
        self.txt_asin = None
        self.__main_layout = None  # 总布局
        self.__video_layout = None  # 视频展示布局
        self.__button_layout = None  # 按键总布局
        self.video_qlabel = None  # 视频框
        self.button_open_camera = None  # 打开摄像头
        self.button_photograph = None  # 拍照
        self.timer_camera = QtCore.QTimer()  # 定时器
        self.cap = cv2.VideoCapture()  # 视频流
        # self.cap.set(5, 60)
        self.CAM_NUM = 0  # 定义摄像头

        self.setWindowTitle("笑脸检测系统")
        self.setWindowIcon(QtGui.QIcon('./image/ico.png'))
        self.resize(670, 600)

        # self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self.setFixedSize(680, 610)
        self.set_layout()  # 设置布局函数
        self.init_txt()  # 输入框
        self.set_video()  # 设置视频布局函数
        self.set_button()  # 设置按键布局函数
        self.set_add()  # 将元素添加到总布局
        self.setLayout(self.__main_layout)  # 应用布局
        self.slot_init()

    def set_layout(self):
        self.__main_layout = QtWidgets.QVBoxLayout()  # 总布局->垂直布局
        self.__video_layout = QtWidgets.QVBoxLayout()  # 视频布局->垂直布局
        self.__button_layout = QtWidgets.QHBoxLayout()  # 按键布局->水平布局

    def init_txt(self):
        self.txt_asin = QtWidgets.QLineEdit()
        self.txt_asin.setPlaceholderText('请输入编号')
        self.txt_asin.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[^\\\\/:*?\"<>|`·]*")))

    def set_video(self):
        self.video_qlabel = QtWidgets.QLabel()
        self.video_qlabel.setFixedSize(640, 480)
        self.pix = QtGui.QPixmap('image/bj.jpg')

        # self.video_qlabel.setStyleSheet("background-color:red;border:0px")
        self.video_qlabel.setPixmap(self.pix)
        self.__video_layout.addWidget(self.video_qlabel)
        self.__video_layout.addWidget(self.txt_asin)

    def set_button(self):
        self.button_open_camera = QtWidgets.QPushButton("打开摄像头")
        self.button_photograph = QtWidgets.QPushButton("拍照")
        self.__button_layout.addWidget(self.button_open_camera)
        self.__button_layout.addWidget(self.button_photograph)

    def set_add(self):
        self.__main_layout.addLayout(self.__video_layout)
        self.__main_layout.addStretch()
        self.__main_layout.addLayout(self.__button_layout)

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_cameras)
        self.button_photograph.clicked.connect(self.button_photographs)
        self.timer_camera.timeout.connect(self.show_camera)

    def button_open_cameras(self):
        if not self.timer_camera.isActive():  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if not flag:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(10)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('关闭摄像头')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.video_qlabel.clear()  # 清空视频显示区域
            self.video_qlabel.setPixmap(self.pix)
            self.button_open_camera.setText('打开摄像头')

    def button_photographs(self):
        if self.cap.isOpened():
            if self.txt_asin.text() != '':
                code = self.opens()
                if code[0] == 0:
                    self.init_error_tips(code[1])
                else:
                    self.init_success_tips()
                # thread = NewThread(self)
                # thread.success.connect(self.init_success_tips)
                # thread.error.connect(self.init_error_tips)
                # thread.start()
            else:
                QtWidgets.QMessageBox.warning(self, '警告', '未输入编号', QtWidgets.QMessageBox.Yes,
                                              QtWidgets.QMessageBox.Yes)

        else:
            QtWidgets.QMessageBox.warning(self, '警告', '摄像头未打开', QtWidgets.QMessageBox.Yes,
                                          QtWidgets.QMessageBox.Yes)

    def init_success_tips(self):
        infoBox = QtWidgets.QMessageBox()
        # infoBox.setIcon(QtWidgets.QMessageBox.Information)
        infoBox.setWindowIcon(QtGui.QIcon('./image/ico.png'))
        infoBox.setText("保存完成！")
        infoBox.setWindowTitle("提示")
        infoBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        infoBox.button(QtWidgets.QMessageBox.Ok).animateClick(500)  # 3秒自动关闭
        infoBox.exec_()

    def init_error_tips(self, state):
        infoBox = QtWidgets.QMessageBox.warning(self, '警告', state, QtWidgets.QMessageBox.Yes,
                                                QtWidgets.QMessageBox.Yes)

    def show_camera(self):
        flag, image = self.cap.read()  # 从视频流中读取
        self.shows = cv2.resize(image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        self.shows = cv2.cvtColor(self.shows, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        self.shows = cv2.flip(self.shows, 1)
        self.showImage = QtGui.QImage(self.shows.data, self.shows.shape[1], self.shows.shape[0],
                                      QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.video_qlabel.setPixmap(QtGui.QPixmap.fromImage(self.showImage))  # 往显示视频的Label里 显示QImage

    def opens(self):
        flag, image = self.cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images = image.copy()
        images = cv2.flip(images, 1)
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
        
        canvas = np.zeros((480, 300, 3), dtype="uint8")
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
                           key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = self.emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = self.EMOTIONS[preds.argmax()]
            text = '情绪:{}:{:.2f}%'.format(label, emotion_probability * 100)
            images = Image.fromarray(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(images)
            fontText = ImageFont.truetype('./module/SIMYOU.TTF', 20, encoding="utf-8")
            draw.text(xy=(10, 450), text=text, font=fontText, fill=(255, 0, 0))
            images = cv2.cvtColor(np.asarray(images), cv2.COLOR_RGB2BGR)
            self.name_hash = hashlib.md5()
            self.name_hash.update(str(time.time()).encode("utf-8"))
            if not os.path.exists('data_img'):
                os.mkdir("data_img")
            if not os.path.exists('./data_img/' + self.txt_asin.text()):
                os.makedirs('./data_img/' + self.txt_asin.text())
            name = './data_img/' + self.txt_asin.text() + '/' + str(self.name_hash.hexdigest()) + '.jpg'

            # print(preds[preds.argmax()])
            cv2.imwrite(name, images)
            self.txt_asin.setText('')

        else:
            return [0, '未检测到人脸']

        return [1, '成功']
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    windows = MainWindows()
    windows.show()
    sys.exit(app.exec_())