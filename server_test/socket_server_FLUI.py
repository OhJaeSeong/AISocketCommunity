from socket import *
from os.path import exists
from _thread import *
import configparser
import threading

import torch
import numpy as np
import torch.nn as nn
import densenet_1ch as densenet
import torch.nn.functional as F

from server_processor import Processor

import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic

import sys
import os
from server_processor import terminate

form_class = uic.loadUiType("server_FL.ui")[0]
qt_path= os.path.dirname(PyQt5.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(qt_path, "Qt/plugins")

class TimeOutException(Exception):
    pass

# class terminate(QObject):
#     stopSignal = pyqtSignal()

def init_model(class_number, wgt_path):
    model = densenet.densenet201(num_classes=class_number)
    model.eval()
    wgt = torch.load(wgt_path, map_location="cpu")
    model.load_state_dict(wgt['model_state_dict'], strict=False)
    return model


class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        HOST = gethostbyname(gethostname())
        self.config = configparser.ConfigParser()
        self.config.read('server_set.ini', encoding='utf=8')
        self.serverSock = socket(AF_INET, SOCK_STREAM) # 1.socket 생성(create)
        self.stopSignal = terminate()
        self.model = init_model(int(self.config['model']['class_number']), self.config['model']['wgt_path'])
                
        self.thr = Processor(self.serverSock, self.model)

        self.setupUi(self)
        self.bind_btn.clicked.connect(self.bind_adr)
        self.start_btn.clicked.connect(self.start_server)
        self.stop_btn.clicked.connect(self.stop_server)

        self.ip_num.setText(HOST)
        self.port_num.setText(self.config['address']['port'])
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        
    
    def bind_adr(self):
        HOST = gethostbyname(gethostname()) # 현재 컴퓨터의 이더넷 ip
        print("현재 ip : ", HOST, ' port : ', int(self.config['address']['port']), '에서 접속 시도')

        try:
            self.serverSock.bind((self.ip_num.toPlainText(), int(self.port_num.toPlainText()))) # 2.주소(IP/Port) 할당(bind)
            self.serverSock.listen(1) # 3.연결 대기(listen)
            self.bind_btn.setEnabled(False)
            self.ip_num.setEnabled(False)
            self.port_num.setEnabled(False)

            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.bind_status.setText("bind")
            self.bind_status.setStyleSheet("background-color: green")

            self.config['address']['ipv4'] = self.ip_num.toPlainText()
            self.config['address']['port'] = self.port_num.toPlainText()
            with open('server_set.ini', 'w', encoding='utf-8') as configfile:
                self.config.write(configfile)
        
        except Exception as ex:        
            print(ex, " 발생. 바인드 해제")
    
    def release_bind(self):
        if self.serverSock:
            self.serverSock.close()
            self.serverSock = socket(AF_INET, SOCK_STREAM)
        print("bind 해제")

        self.bind_btn.setEnabled(True)
        self.ip_num.setEnabled(True)
        self.port_num.setEnabled(True)
    
    def start_server(self):
        if self.serverSock:
            try:
                # self.thr = threading.Thread(target=make_clientThread, args=(self.serverSock, self.model), daemon=True)
                print("server thread start")
                self.thr.start()
                self.start_btn.setEnabled(False)
                self.server_status.setText("server on")
                self.server_status.setStyleSheet("background-color: green")
            except Exception as ex:
                print(ex, "발생. 서버 시작 불가능")
    
    def stop_server(self):
        self.serverSock.close()
        self.stopSignal.stopSignal.emit()
        
        self.thr.finish_thread()
        self.release_bind()
        self.thr = Processor(self.serverSock, self.model)

        self.bind_status.setText("not bind")
        self.server_status.setText("server off")
        self.bind_status.setStyleSheet("background-color: gray")
        self.server_status.setStyleSheet("background-color: gray")

    def closeEvent(self, event):
        quit_msg = "Want to exit?"
        reply = QMessageBox.question(self, 'Message', quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.thr.finish_thread()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
    