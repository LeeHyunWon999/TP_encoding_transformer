import torch
import torch.nn

# def Weighted_CE_Loss():
    
#     return torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.4, 0.2, 0.5, 0.2]))


def Weighted_CE_Loss(args_weight):
    
    return torch.nn.CrossEntropyLoss(weight=args_weight)