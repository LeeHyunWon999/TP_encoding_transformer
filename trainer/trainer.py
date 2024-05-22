import einops
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import get_data_loaders , get_loss, get_model, get_metric, Mode
from model.metrics import Metrics
import model.models as model
from model.loss import Weighted_CE_Loss
import os 


class ECGClassifierTrainer:

    def __init__(self, args) -> None:
        
        
        self.args = args

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args['device']['gpu']
        
        self.model_args = self.args['model']
        self.trainer_args = self.args['trainer']
        self.data_loader_args = self.args['data_loader']
        self.device = args['device']['cuda']

        self.model = get_model(self.model_args).to(self.device)
        print('selected :',self.model_args['type'])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.trainer_args['lr'], weight_decay=self.trainer_args['weight_decay'])
        
        self.loss = get_loss(self.args['loss']).to(self.device)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.trainer_args['T_max'], self.trainer_args['eta_min'])
        self.early_stopping_epochs = self.trainer_args['early_stopping_epochs']
        
        self.data_loader = get_data_loaders(self.data_loader_args, self.model_args)
        
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        self.best_model_path = self.trainer_args['best_model_path']

        
        self.metrics = {
            Mode.train: Metrics(),
            Mode.eval: Metrics()
        }
        
       
        # train metric
        self.train_auroc = get_metric(self.args['metric'], self.model_args['args'], self.metrics[Mode.train])
            
        # valid metric
        self.valid_auroc = get_metric(self.args['metric'], self.model_args['args'], self.metrics[Mode.eval])

        
        #tensorboard
        self.writer = SummaryWriter(log_dir='result')



    def train(self):
        # epoch만큼 train돌려서 한 epoch당 나온 결과 confusion matrix image를 confusion_matrices_image_train에 저장
        # validation또한 동일. (성동일)
        # 근데 validation은 짝수 epoch마다 진행.
        # 최종적으로 return으로 confusion_matrices_image_train, confusion_matrices_image_eval를 보냄.

        for epoch in range(self.trainer_args['num_epochs']):

            self.train_epoch(epoch)
            early_stop = self.validate_epoch(epoch)  

            if early_stop == "break":
                break
            
        self.writer.close()


    def train_epoch(self, epoch):
        self.model.train()
        
        
        # train data loader를 tqdm으로 넣어줌.
        loader = tqdm(self.data_loader[Mode.train])
        accuracy = 0
        total_loss = 0
        # metric을 초기화
        # 매 epoch마다 초기화 해줘야함. 아니면 이전 epoch의 측정 값이 현재 epoch의 측정값에 영향을 줄 수 있음.
        self.metrics[Mode.train].reset()
        
        for index, data in enumerate(loader):
            self.optimizer.zero_grad()
            signal, label = [d.to(self.device) for d in data]
            
            
            # einops.rearrange()는 텐서의 형태를 재배열하는데 사용됨. signal은 원래 (batch, channel, elemnet)로 되어있던 것이 
            # (batch, element, channel)형태로 변경하여 model의 input으로 들어가게됨.
            prediction = self.model(signal)
            loss = self.loss(prediction, label)
            loss.backward()
            self.optimizer.step()
          
            # metric
            accuracy += torch.sum(prediction.argmax(1) == label)
            self.train_auroc.update(prediction,label)
    

            # train의 acc를 update함.
            self.metrics[Mode.train].update(prediction.argmax(1), label)
            total_loss += loss.item()
            
            # 출력
            loader.set_description(f"TRAINING: {epoch}, loss: {loss.item()}. Target: {label[:8].tolist()}, Prediction: {prediction.argmax(1)[:8].tolist()}")
        
        # metric and loss 계산
        
        train_acc = accuracy.item() / len(loader) / self.data_loader_args['args']['batch_size']
        auroc_result = self.train_auroc.compute()
        train_loss = total_loss / len(loader)
        
        # epoch 한번이 끝난후 출력
        print(f"Train Accuracy: {train_acc},  auroc_result :  {auroc_result},  Loss :  {train_loss}")
        print(f"confusion metrix : \n {self.metrics[Mode.train].confusion_matrix()}")
        
        # tensorboard
        self.writer.add_scalar("acc/train",epoch, train_acc)
        self.writer.add_scalar("Loss/train",epoch, train_loss)
        self.writer.add_scalar("AUROC/train", epoch, auroc_result)

        self.scheduler.step()
        
        

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()

        accuracy = 0
        total_loss = 0
        loader = tqdm(self.data_loader[Mode.eval])
        self.metrics[Mode.eval].reset()
        for index, data in enumerate(loader):
            signal, label = [d.to(self.device) for d in data]
            prediction = self.model(signal)
            loss = self.loss(prediction, label)
            
            # metric
            accuracy += torch.sum(prediction.argmax(1) == label)
            self.valid_auroc.update(prediction,label)

            total_loss += loss.item()
            self.metrics[Mode.eval].update(prediction.argmax(1), label)
            loader.set_description(f"VALIDATION: {epoch}, loss: {loss.item()}. Target: {label[:8].tolist()}, Prediction: {prediction.argmax(1)[:8].tolist()}")
        
        valid_acc = accuracy.item() / len(loader) / self.data_loader_args['args']['batch_size']
        auroc_result = self.valid_auroc.compute()
 
        cm = self.metrics[Mode.eval].confusion_matrix()
        valid_loss = total_loss / len(loader)
        
        # epoch 한번이 끝난후 출력
        print(f"valid Accuracy: {valid_acc}, auroc_result :  {auroc_result},  Loss :  {valid_loss}")
        print(f"confusion metrix : \n {cm}")
        
        # tensorboard
        self.writer.add_scalar("acc/valid", valid_acc, epoch)
        self.writer.add_scalar("Loss/valid", valid_loss , epoch)
        self.writer.add_scalar("AUROC/valid",  auroc_result, epoch)

        
        # early stopping
        if valid_loss > self.best_loss:
            self.early_stop_counter += 1
        else:
            self.best_loss = valid_loss

        
            torch.save(self.model, self.best_model_path)
            self.early_stop_counter = 0
            
        if self.early_stop_counter >= self.early_stopping_epochs:
            print("Early Stopping")
            return "break" 
        
        return "continue" 


