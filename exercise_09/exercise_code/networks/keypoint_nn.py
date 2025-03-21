"""Models for facial keypoint detection"""

from os import access
import torch
import torch.nn as nn
import pytorch_lightning as pl


import torch.optim as optim 
import numpy as np


# TODO: Choose from either model and uncomment that line
class KeypointModel(nn.Module):
# class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        NOTE: You could either choose between pytorch or pytorch lightning, 
            by switching the class name line.
        """
        super().__init__()
        # self.save_hyperparameters(hparams) # uncomment when using pl.LightningModule
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################

        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


        self.model = nn.Sequential(

          nn.Conv2d(1, 16, 4),  # 96 -> 93
          nn.BatchNorm2d(16),
          nn.ReLU(),
          nn.MaxPool2d(2, 2), # 93 -> 46
          nn.Dropout2d(0.1),
          
          nn.Conv2d(16, 32, 3),  # 46 -> 44
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.MaxPool2d(2, 2), # 44 -> 22
          nn.Dropout2d(0.1),

          nn.Conv2d(32, 64, 2), # 22 -> 21
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.MaxPool2d(2, 2), # 21 -> 10
          nn.Dropout2d(0.1), 

          nn.Flatten(1, -1),
          
          nn.Linear(64 * 10 * 10, 500),
          nn.BatchNorm1d(500),
          nn.ReLU(),
          nn.Dropout(0.1),

          nn.Linear(500, 500),
          nn.BatchNorm1d(500),
          nn.ReLU(),
          nn.Dropout(0.1),

          nn.Linear(500, 250),
          nn.BatchNorm1d(250),
          nn.ReLU(),
          nn.Dropout(0.1),

          nn.Linear(250, 30)
        )

        self.model.apply(self._init_weight)


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        x = self.model(x)     

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


    def _init_weight(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias) 
                   



class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
