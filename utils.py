

from src.model.models import SIMPLEMLP, ECG1DCNN, ECGformer
from src.model.metrics import Metrics
from src.model.loss import Weighted_CE_Loss
from src.data.data_loader import MITLoader_MLP , MITLoader_CNN_Transformer
import torch.utils.data as data
from src.data.augmentations import Compose, RandomNoise, RandomShift
import dataclasses
from enum import Enum
import torch


class Mode(Enum):
    train = "train"
    eval = "eval"

def get_data_loaders(data_args, model_args):
    
    
    path = get_path(data_args['args'])
    transforms = get_trans()
    if data_args['type'] == "MIT-BIH":
        
        if model_args['type'] == 'MLP':
            return {
                Mode.train: data.DataLoader(
                    MITLoader_MLP(path[Mode.train], transforms[Mode.train]),
                    batch_size=data_args['args']['batch_size'],
                    num_workers=data_args['args']['num_workers'],
                    shuffle=True,
                    drop_last=True
                ),
                Mode.eval: data.DataLoader(
                    MITLoader_MLP(path[Mode.eval], transforms[Mode.eval]),
                    batch_size=data_args['args']['batch_size'],
                    num_workers=data_args['args']['num_workers'],
                    shuffle=False,
                    drop_last=True
                )
            }

        else :
            return {
                Mode.train: data.DataLoader(
                    MITLoader_CNN_Transformer(path[Mode.train], transforms[Mode.train]),
                    batch_size=data_args['args']['batch_size'],
                    num_workers=data_args['args']['num_workers'],
                    shuffle=True,
                    drop_last=True
                ),
                Mode.eval: data.DataLoader(
                    MITLoader_CNN_Transformer(path[Mode.eval], transforms[Mode.eval]),
                    batch_size=data_args['args']['batch_size'],
                    num_workers=data_args['args']['num_workers'],
                    shuffle=False,
                    drop_last=True
                )
            }
            

def get_path(args):
    path = {
        Mode.train: args['path']["train"],
        Mode.eval: args['path']["eval"]
    }
    return path
    
def get_trans():
    transforms = {
        Mode.train: Compose([RandomNoise(0.05, 0.5), RandomShift(10, 0.5)]),
        Mode.eval: lambda x: x
    }
    return transforms


def get_model(args):
    name = args['type'].upper()
    
    if name == "MLP":
        return SIMPLEMLP(args['args']['num_classes'])
    elif name == "CNN":
        return ECG1DCNN(args['args']['num_classes'])
    elif name == "TRANSFORMER":
        return ECGformer(args['args']['num_layers'],
                         args['args']['signal_length'],
                         args['args']['num_classes'],
                         args['args']['input_channels'],
                         args['args']['embed_size'],
                         args['args']['num_heads'],
                         args['args']['expansion'])
                         
    
    
def get_metric(args, model_args, metrics):

   
    if args['type'] == "f1":
        return metrics.f1_score(args, model_args)

    elif args['type'] == "AUROC":
        return metrics.AUROC(args, model_args)

    elif args['type'] == "AUPRC":
        return metrics.AUPRC(args, model_args)
    

def get_loss(args):
    
    if args['type'] == "Weighted_CE_loss":
        return Weighted_CE_Loss(torch.tensor(args['weight']))