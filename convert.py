# 필수 패키지 import
import sys
import torch
import torch.nn as nn
import numpy as np
import json

# SNN 변환 툴 import
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import ann2snn

# 로컬 모듈 import
from train import load_json, setup_parser
import utils



# 메인함수 시작
if __name__ =="__main__":
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) 
    args.update(param)
    print(args)



# 클래스 정의부분을 systemPath에 추가
sys.path.insert(0, "/home/hongwonseok/ECG-SNN/src/model/model")

# 데이터로더 정의(기존 경로에서 그대로 가져오기)
#temp_train_loader = utils.get_data_loaders(config.EcgConfig.dataset)
#temp_valid_loader = utils.get_data_loaders(config.EcgConfig.dataset)

# 모델 정의 2 : 완성된 MLP 모델을 불러와서 변환하도록 하자.
modelPath = "/data/ghdrnjs/ECG-SNN/best_model/CNN/CNN_best_model.pth"

temp_model= torch.load(modelPath)
print(temp_model.state_dict())
