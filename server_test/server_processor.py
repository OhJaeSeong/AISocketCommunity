from socket import *
from os.path import exists
from _thread import *

import torch
import numpy as np
import torch.nn as nn
import densenet_1ch as densenet
import torch.nn.functional as F

import select
import struct
import threading
from functools import wraps
from threading import Thread

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

_request_format: str = "3s1s3siiiiii"
_response_format: str = "3s1siif"
""" 응답 구조체 포맷
data[0]: 시작 문자열(3c)
data[1]: 작업 요청 문자(1s)
data[2]: 요청 상태 코드(i)
data[3]: 작업 결과(i)
data[4]: 작업 결과 점수(f)
"""

class terminate(QObject):
    stopSignal = pyqtSignal()

class Processor(Thread):
    def __init__(self, server_socket : socket, target_model : densenet):
        super(Processor, self).__init__()
        self.terminator = terminate()
        self.terminator.stopSignal.connect(self.finish_thread)
        self.server_socket = server_socket
        self.target_model = target_model
        self.connectionSock = None

    def finish_thread(self):
        if self. server_socket:
            self.server_socket.close()
        if self.connectionSock:
            self.connectionSock.close()
        print("close server...")
    
    def scan_image(self, input):
        out = self.target_model.features[1:12](input)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.target_model.classifier(out)
        return out

    def run(self):
        count = 0
        while self.server_socket: # 동일한 ip, port번호(즉 동일한 사용자)가 접속요청이 된경우 예외처리, 거부를 할수있는 로직
            # try:
            self.connectionSock, addr = self.server_socket.accept() # 4.연결 승인(accept)
            print(str(addr),'에서 접속했습니다')
            # self.process_thread(self.connectionSock, addr)

            thr = threading.Thread(target=self.process_thread, args=(self.connectionSock, addr))
            thr.start()
            count += 1
            print("현재까지 요청 숫자 : ", count)
            
            # except Exception as ex:
            #     print("연결 중단 ")
            #     break

    def process_thread(self, connectionSock : socket, addr):
        request_len = 32
        time_limit = 5
        buffer_size = 1024

        try:
            retry_count = 3
            while connectionSock and self.server_socket:  
                try:
                    r,w,x = select.select([connectionSock,],[],[], 10)
                    if len(r) <= 0: #timeout
                        print("transmit delayed")
                        raise TimeoutError
                    
                    request_data = connectionSock.recv(request_len)
                    req_struct = struct.unpack(_request_format, request_data)

                    if req_struct[1] == b'C':
                        # print("live come ", req_struct[0], req_struct[1], req_struct[2])
                        if retry_count != 5:
                            retry_count = 5

                    else:
                        if req_struct[4] != 0:
                            response = struct.pack('3s1si', b"hns", req_struct[1], 0)
                            connectionSock.send(response)  
                    
                        resp = struct.pack('3s1sii', b"HNS",  req_struct[1], 0, 0) # 패킷 상태(0: 정상 1:에러), 순서
                        err_resp = struct.pack('3s1sii', b"HNS",  req_struct[1], 1, -1)

                        if req_struct[4] % buffer_size == 0:
                            transmit_scale = int(req_struct[4] / buffer_size)
                        else:
                            transmit_scale = int(req_struct[4] / buffer_size) + 1

                        tensor = None
                        r,w,x = select.select([connectionSock,],[],[],time_limit)
                        if len(r) <= 0: #timeout
                            print("transmit delayed")
                            raise TimeoutError
                        
                        tensor_buffer = connectionSock.recv(buffer_size + 2)
                        tensor = tensor_buffer[0:buffer_size]
                        packet_num = tensor_buffer[buffer_size:buffer_size+2]
                        if int(np.frombuffer(packet_num, dtype=np.float16)) == 0: # 0 => 
                            connectionSock.send(resp)
                        # tensor = exchange_packet(time_limit, buffer_size, resp)

                        try:
                            loop_count = 0
                            
                            while loop_count < (transmit_scale - 1): # 
                                try:
                                    r,w,x = select.select([connectionSock,],[],[], time_limit) # 너무 오래기다린 경우 제외 몇번의 응답기회를 더 줄수 있도록 하기
                                    if len(r) <= 0: #timeout
                                        print("transmit delayed")
                                        raise TimeoutError
                                    
                                    tensor_buffer = connectionSock.recv(buffer_size + 2) # 무한대기 방지를 위한 타임아웃 기능 생성
                                    tensor += tensor_buffer[0:buffer_size]
                                    loop_count += 1
                                    
                                    packet_num = tensor_buffer[buffer_size:buffer_size+2]
                                    packet_num = np.frombuffer(packet_num, dtype=np.float16)

                                    if int(packet_num) == loop_count: # 예외 발생시 클라이언트에 알려줌
                                        resp = struct.pack('3s1sii', b"HNS",  req_struct[1], 0, int(packet_num))
                                        connectionSock.send(resp)
                                    else:
                                        raise Exception
                                except Exception as ex:        
                                    retry_count -= 1
                                    if retry_count > 0:
                                        connectionSock.send(err_resp) 
                                        continue
                                
                            # tensor = torch.frombuffer(tensor[0:1024], dtype=torch.float32)
                            tensor = torch.Tensor(np.frombuffer(tensor, dtype=np.float16))
                            tensor = torch.reshape(tensor, (1, 64, 128, 128))
                                
                            with torch.no_grad():
                                out = self.scan_image(tensor)
                                out = np.array(out)

                            if out[0][0] > out[0][1]:
                                result = struct.pack(_response_format, b"HNS",  req_struct[1], 0, 0, np.float16(out[0][1]))
                            else:
                                result = struct.pack(_response_format, b"HNS",  req_struct[1], 0, 1, np.float16(out[0][1]))
                            # out = out.tobytes()
                            try:
                                connectionSock.send(result)
                            except Exception as ex:
                                pass
                        
                        except Exception as ex:
                            print(ex)
                        
                except TimeoutError:
                    if retry_count > 1:
                        retry_count -= 1
                        print("남은 재시도 : ", retry_count)
                        continue
                    else:
                        print("client dead. socket finish")
                        connectionSock.close()
                        raise
                    
                except Exception as ex:
                    print(ex, " 발생")
                    connectionSock.close()
                    raise
                    # qtime 주기적으로 status체크
                    # erorr 종류에 따라 처리 방식 다름. timeout 일 경우 위로 올려줌
        
        except Exception as ex:
            print('연결 종료')