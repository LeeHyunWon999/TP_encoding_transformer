import io

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchmetrics import F1Score, AUROC, AveragePrecision
from configs.config import EcgConfig, Mode
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



class Metrics:

    def __init__(self) -> None:
        self.predictions = []
        self.labels = []
        

    def reset(self):
        self.predictions = []
        self.labels = []

    def update(self, prediction, label):
        prediction = prediction.tolist()
        label = label.tolist()
        self.predictions += prediction
        self.labels += label
 
    
    def AUROC(self, args, model_args):

        auroc = AUROC(task=args['task'], num_classes=model_args['num_classes'], average=args['average'])
        return auroc
    
    
    def f1_score(self, args, model_args):

        f1_weighted = F1Score(task=args['task'], num_classes=model_args['num_classes'], average=args['average'])
        return f1_weighted
    

    
    def AUPRC(self, args, model_args):

        auprc = AveragePrecision(task=args['task'], num_classes=model_args['num_classes'], average=args['average'])
        return auprc


    
    def sensitivity(self):
        pass

    @property
    def specificity(self):
        pass

    @property
    def recall(self):
        pass

    @property
    def accuracy(self):
        pass

    def confusion_matrix(self):
        cm = confusion_matrix(self.labels, self.predictions)
        return cm

    def confusion_matrix_image(self):
        figure = ConfusionMatrixDisplay(self.confusion_matrix()).plot().figure_

        def get_img_from_fig(fig, dpi=180):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            img = cv2.imdecode(img_arr, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

        numpy_figure = get_img_from_fig(figure)
        plt.close()
        plt.clf()
        plt.cla()
        return numpy_figure
