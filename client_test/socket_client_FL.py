import struct
import socket
from _thread import *
from PIL import Image
import os
import sys
import configparser

import torch
import torchvision.transforms as T
import numpy as np
import densenet_1ch

from threading import Thread
from queue import Queue

import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
# from server_module.request_data import RequestData

form_class = uic.loadUiType("client_FL.ui")[0]

qt_path= os.path.dirname(PyQt5.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(qt_path, "Qt/plugins") # qtpixmap 환경변수 설정

_request_format: str = "3s1s3siiiiii"
""" 요청 구조체 포맷
data[0]: 시작 문자열(3s)
data[1]: 작업 요청 문자(1s)
data[2]: 데이터 타입(3s)
data[3]: 테스트 인덱스(i)
data[3]: 패킷 크기(i)
data[4]: 텐서의 배치(i)
data[4]: 텐서의 채널(i)
data[5]: 텐서의 너비(i)
data[6]: 텐서의 높이(i)
"""

_response_format: str = "3s1siif"
""" 응답 구조체 포맷
data[0]: 시작 문자열(3c)
data[1]: 작업 요청 문자(1s)
data[2]: 요청 상태 코드(i)
data[3]: 작업 결과(i)
data[4]: 작업 결과 점수(f)
"""


label_tags = {
            0: 'Normal',
            1: 'Pneumonia',
}
qu = Queue(1)

class Communicate(QObject):
    updateState = pyqtSignal()

class Disconnect(QObject):
    errorState = pyqtSignal()


class Worker(Thread):
    def __init__(self, update=None, discn=None):
        self.update = update
        self.discn = discn

        super(Worker, self).__init__()
        self.running = True
        self.selected_file =''
        self.target_model = ''
        self.client_socket = ''
    
    def setting(self, file_path, model, socket):
        self.selected_file = file_path
        self.target_model = model
        self.client_socket = socket
    
    def run(self):
        try:
            pred, score = self.make_tensor()
            qu.put([pred, score])
            self.update.updateState.emit()
        except Exception as ex:
            print(ex, "발생. 스레드 종료")
            self.discn.errorState.emit()
    
    def make_tensor(self) :
        transforms = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])

        img = custom_pil_loader(self.selected_file) # image_path
        transformed_img = (transforms(img))
        transformed_img = transformed_img.unsqueeze(0).to("cpu")

        with torch.no_grad():
            output = self.target_model.features[0](transformed_img.to("cpu"))
            output = np.array(output, dtype='float16')
        try:
            pred, score = self.transmit_data(output, self.client_socket)
            return pred, score
        except Exception as ex:
            pass
    
    def transmit_data(self, tensor, cli_socket) :
        tensor = tensor.tobytes()
        buffer_size = 1024

        try:
            # data = (send_packet).to_bytes(4, byteorder="little")
            request_packet = struct.pack(_request_format, b"HNS", b'D', b'F16', 0, len(tensor), 1, 64, 128, 128)
            cli_socket.send(request_packet)

            server_res = cli_socket.recv(8)
            server_res = struct.unpack('3s1si', server_res)

        except Exception as ex:
                print(ex, "발생")

        try:
            num = 0
            while len(tensor)> num * buffer_size:
                packet = tensor[num * buffer_size:(num+1) * buffer_size] + np.array([num], dtype='float16').tobytes()
                try:
                    cli_socket.send(packet)
                    resp = cli_socket.recv(12)
                    resp = struct.unpack('3s1sii', resp)
                    if resp[2] != 0 or num != resp[3]:
                        print(num, ' , ', resp[3])
                        raise Exception
                    num += 1
                except Exception as ex:
                    continue
            # 결과 대기
            result = cli_socket.recv(16)
            result = struct.unpack(_response_format, result) # torch.Tensor([np.frombuffer(result, dtype=np.float32)])

            # _score, predicted = result.max(1)
            _score = result[4]
            pred = label_tags[result[3]] # label_tags[predicted.item()]
            return pred, _score

        except Exception as ex:
            num = 0
            print(ex, "발생")


def custom_pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img.convert('L')

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read('client_set.ini', encoding='utf=8')
        self.host_port = int(self.config['host']['port'])
        self.host_adr = self.config['host']['address']

        self.target_model = densenet_1ch.densenet201(num_classes=int(self.config['model']['class_number']))
        self.target_model.eval()
        self.selected_file = ''
        self.client_socket = 0
        self.upst = Communicate() # 결과 업데이트 시그널
        self.discn = Disconnect() # 예외 발생시 소켓 종료 시그널
        
        self.checkTimer = QTimer() # live check
        self.checkTimer.setInterval(5000)
        self.checkTimer.timeout.connect(self.response_toCheck)
        
        self.thr = Worker(update=self.upst, discn=self.discn)
        self.upst.updateState.connect(self.refresh)
        self.discn.errorState.connect(self.disconnect_server)
        # target_model = target_model.to("cuda")
        

        self.setupUi(self)
        self.select_btn.clicked.connect(self.loadImageFromFile)
        self.transmit_btn.clicked.connect(self.make_thread) # make_tensor -> connectServer
        self.connect_btn.clicked.connect(self.connect_server)
        self.disconnect_btn.clicked.connect(self.disconnect_server)

        wgt = torch.load(self.config['model']['wgt_path'], map_location="cpu")
        self.target_model.features[0].load_state_dict(wgt, strict=False)

        self.ip_num.setText(self.host_adr) # self.ipv4
        self.port_num.setText(str(self.host_port)) # self.port
        self.transmit_btn.setDisabled(True)
        self.disconnect_btn.setEnabled(False)

    def response_toCheck(self):
        try:
            livecheck_response = struct.pack(_request_format, b"HNS", b'C', b'RES', 0, 0, 0, 0, 0, 0)
            self.client_socket.send(livecheck_response)
        except Exception as ex:
            self.disconnect_server()
            print("오류")


    def loadImageFromFile(self) :
        #QPixmap 객체 생성 후 이미지 파일을 이용하여 QPixmap에 사진 데이터 Load하고, Label을 이용하여 화면에 표시
        fname = self.callFile()
        if fname[0] != '':
            self.qPixmapFileVar = QPixmap()
            self.selected_file = fname[0]

            img = Image.open(self.selected_file)
            width, height = img.size
            
            if width > height:
                ratio = float(self.image_zone.width() / width)
            else:
                ratio = float(self.image_zone.height() / height)

            self.qPixmapFileVar.load(self.selected_file)
            self.qPixmapFileVar = self.qPixmapFileVar.scaled(int(width * ratio), int(height * ratio))
            self.image_zone.setPixmap(self.qPixmapFileVar)

            if self.client_socket:
                self.transmit_btn.setEnabled(True)

    def callFile(self) :
        fname = QFileDialog.getOpenFileName(self)
        return fname

    def make_thread(self):
        self.thr.setting(self.selected_file, self.target_model, self.client_socket)
        self.thr.daemon = True
        try:
            self.checkTimer.stop()
            self.thr.start()
            self.transmit_btn.setEnabled(False)
        except Exception as ex:
            self.disconnect_server()


    def connect_server(self) :
        self.client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        
        self.config['host']['address'] = self.ip_num.toPlainText()
        self.config['host']['port'] = self.port_num.toPlainText()

        try:
            self.client_socket.connect((self.ip_num.toPlainText(), int(self.port_num.toPlainText())))
            print('연결에 성공했습니다.')
            with open('client_set.ini', 'w', encoding='utf-8') as configfile:
                self.config.write(configfile)
            
            self.connect_btn.setDisabled(True)
            self.ip_num.setDisabled(True)
            self.port_num.setDisabled(True)

            if self.selected_file != '':
                self.transmit_btn.setEnabled(True)
            
            self.checkTimer.start()
            self.disconnect_btn.setEnabled(True)
        
        except Exception as ex:
            print(ex)
    

    def disconnect_server(self):
        print("disconnect")
        try:
            if self.client_socket:
                self.client_socket.close()
                self.connect_btn.setEnabled(True)
                self.transmit_btn.setDisabled(True)

                self.ip_num.setEnabled(True)
                self.port_num.setEnabled(True)
                self.thr = Worker(update=self.upst, discn=self.discn)

                self.checkTimer.stop()
                self.disconnect_btn.setEnabled(False)
            else:
                print("any socket exist")
        except Exception as ex:
                print(ex, "발생")

    def refresh(self):
        self.scan_txt.clear()
        self.scan_txt.append(str(list(qu.queue)[0][0])) # list(qu.queue)[0]
        self.score_txt.clear()
        self.score_txt.append(str(list(qu.queue)[0][1]))
        self.thr.join()

        qu.queue.clear()
        self.thr = Worker(update=self.upst, discn=self.discn)
        self.transmit_btn.setEnabled(True)
        self.checkTimer.start()


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()