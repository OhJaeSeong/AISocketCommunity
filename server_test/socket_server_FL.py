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

import select
import struct
from functools import wraps

_request_format: str = "3s1s3siiiiii"
_response_format: str = "3s1siif"
""" 응답 구조체 포맷
data[0]: 시작 문자열(3c)
data[1]: 작업 요청 문자(1s)
data[2]: 요청 상태 코드(i)
data[3]: 작업 결과(i)
data[4]: 작업 결과 점수(f)
"""

class TimeOutException(Exception):
    pass


def activate_model(req_struct, time_limit):
    buffer_size = 1024
    resp = struct.pack('3s1sii', b"HNS",  req_struct[1], 0, 0) # 패킷 상태(0: 정상 1:에러), 순서
    err_resp = struct.pack('3s1sii', b"HNS",  req_struct[1], 1, -1)

    if req_struct[4] % buffer_size == 0:
        transmit_scale = int(req_struct[4] / buffer_size)
    else:
        transmit_scale = int(req_struct[4] / buffer_size) + 1

    r,w,x = select.select([connectionSock,],[],[],time_limit)
    if len(r) <= 0: #timeout
        print("transmit delayed")
        raise TimeoutError
    
    tensor_buffer = connectionSock.recv(buffer_size + 2)
    tensor = tensor_buffer[0:buffer_size]
    packet_num = tensor_buffer[buffer_size:buffer_size+2]
    if int(np.frombuffer(packet_num, dtype=np.float16)) == 0: # 0 => 
        connectionSock.send(resp)

    try:
        loop_count = 0
        
        while loop_count < (transmit_scale - 1): # 
            r,w,x = select.select([connectionSock,],[],[], time_limit) # 너무 오래기다린 경우 제외 몇번의 응답기회를 더 줄수 있도록 하기
            try:
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
    
    except Exception as ex:
        print(ex)
    
    return tensor


def process_thread(connectionSock : socket, addr, target_model : densenet):
    request_len = 32
    time_limit = 5
    # clock 시작점 -> 시작 시간 받기

    try:
        retry_count = 5
        while connectionSock:      
            try:
                r,w,x = select.select([connectionSock,],[],[], 10)
                if len(r) <= 0: #timeout
                    print("transmit delayed")
                    raise TimeoutError
                
                request_data = connectionSock.recv(request_len)
                req_struct = struct.unpack(_request_format, request_data)

                if req_struct[1] == b'C':
                    print("live come ", req_struct[0], req_struct[1], req_struct[2])
                    if retry_count != 5:
                        retry_count = 5

                else:
                    if req_struct[4] != 0:
                        response = struct.pack('3s1si', b"hns", req_struct[1], 0)
                        connectionSock.send(response)
                    
                    tensor = activate_model(req_struct, time_limit)
                    print(len(tensor))
                    # tensor = torch.frombuffer(tensor[0:1024], dtype=torch.float32)
                    tensor = torch.Tensor(np.frombuffer(tensor, dtype=np.float16))
                    tensor = torch.reshape(tensor, (1, 64, 128, 128))
                        
                    with torch.no_grad():
                        out = scan_image(target_model, tensor)
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

def scan_image(model, input):
    out = model.features[1:12](input)
    out = F.relu(out, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = model.classifier(out)
    return out

def init_model(class_number, wgt_path):
    model = densenet.densenet201(num_classes=class_number)
    model.eval()
    wgt = torch.load(wgt_path, map_location="cpu")
    model.load_state_dict(wgt['model_state_dict'], strict=False)
    return model


if __name__ == "__main__" :
    config = configparser.ConfigParser()
    config.read('server_set.ini', encoding='utf=8')

    serverSock = socket(AF_INET, SOCK_STREAM) # 1.socket 생성(create)
    HOST = gethostbyname(gethostname()) # 현재 컴퓨터의 이더넷 ip
    print("현재 ip : ", HOST, ' port : ', int(config['address']['port']), '에서 실행')

    serverSock.bind((HOST, int(config['address']['port']))) # 2.주소(IP/Port) 할당(bind)
    serverSock.listen(1) # 3.연결 대기(listen)
    model = init_model(int(config['model']['class_number']), config['model']['wgt_path'])
    # target_model = target_model.to("cuda")

    count = 0
    while True: # 동일한 ip, port번호(즉 동일한 사용자)가 접속요청이 된경우 예외처리, 거부를 할수있는 로직
        if count > 3:
            break
        connectionSock, addr = serverSock.accept() # 4.연결 승인(accept)
        print(str(addr),'에서 접속했습니다')
        thr = threading.Thread(target=process_thread, args=(connectionSock, addr, model))
        thr.start()
        count += 1
        print("현재까지 요청 숫자 : ", count)

    serverSock.close() # 6.socket 종료(close)