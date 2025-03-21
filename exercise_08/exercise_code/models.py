import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

import torch.optim as optim

import matplotlib.pyplot as plt


class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        ########################################################################
        # TODO: Initialize your encoder!                                       #                                       
        # Hint: You can use nn.Sequential() to define your encoder.            #
        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             # 
        # Look online for the APIs.                                            #
        # Hint: wrap them up in nn.Sequential().                               #
        # Example: nn.Sequential(nn.Linear(10, 20), nn.ReLU())                 #
        ########################################################################
        
        hidden_size = hparams["hidden_size"]


        self.encoder = nn.Sequential(
          nn.Linear(28 * 28, 512),
          nn.BatchNorm1d(512),
          nn.Dropout(p=0.1),
          nn.ReLU(),

          nn.Linear(512, 128),
          nn.BatchNorm1d(128),
          nn.Dropout(p=0.1),
          nn.ReLU(),
          
          nn.Linear(128, 20),
          nn.BatchNorm1d(20),
          nn.Dropout(p=0.1),
          nn.Tanh()
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)


class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None

        ########################################################################
        # TODO: Initialize your decoder!                                       #
        ########################################################################
        hidden_size = hparams["hidden_size"]


        self.decoder = nn.Sequential(
          nn.Linear(20, 512),
          nn.ReLU(),
          # nn.Tanh(),

          nn.Linear(512, 256),
          nn.ReLU(),
          # nn.Tanh(),
          nn.Linear(256, 28*28)
        )
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        reconstruction = None
        ########################################################################
        # TODO: Feed the input image to your encoder to generate the latent    #
        #  vector. Then decode the latent vector and get your reconstruction   #
        #  of the input.                                                       #
        ########################################################################
        
        reconstruction = self.decoder(self.encoder(x))

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return reconstruction

    def set_optimizer(self):

        self.optimizer = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################

        # self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr = self.hparams["learning_rate"], weight_decay = self.hparams["weight_decay"])
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.hparams["learning_rate"], weight_decay = self.hparams["weight_decay"])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################


    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the training step, similraly to the way it is shown in      #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Don't forget to reset the gradients before each training step!       #
        #                                                                      #
        # Hint 2:                                                              #
        # Don't forget to set the model to training mode before training!      #
        #                                                                      #
        # Hint 3:                                                              #
        # Don't forget to reshape the input, so it fits fully connected layers.#
        #                                                                      #
        # Hint 4:                                                              #
        # Don't forget to move the data to the correct device!                 #                                     
        ########################################################################

        self.train()

        images = batch     # labels are not used
        images = images.to(self.device)

        # flatten the images to a vector         
        images = images.view(images.shape[0], -1)

        self.optimizer.zero_grad() # reset the gradients 

        pred = self.forward(images)
        loss = loss_func(pred, images)

        loss.backward()        # Stage 2: Backward()
        self.optimizer.step() # Stage 3: Update the parameters
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss


    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the validation step, similraly to the way it is shown in    #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Here we don't supply as many tips. Make sure you follow the pipeline #
        # from the notebook.                                                   #
        ########################################################################

        self.eval()

        # no update in the parameters
        images = batch 
        images = images.to(self.device)

        images = images.view(images.shape[0], -1) 
        pred = self.forward(images)
        loss = loss_func(pred, images)
        loss += loss.item()

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        ########################################################################
        # TODO:                                                                #
        # Given an Encoder, finalize your classifier, by adding a classifier   #   
        # block of fully connected layers.                                     #                                                             
        ########################################################################
        
        latent_dim = self.hparams["latent_dim"]
        hidden_size = self.hparams["hidden_size"]
        num_classes = self.hparams["num_classes"]

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            

            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),

            nn.Linear(256, num_classes)
          )        

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.set_optimizer()

    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):
        
        self.optimizer = None
        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.hparams["learning_rate"], weight_decay = self.hparams["weight_decay"])
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.hparams["learning_rate"], weight_decay = self.hparams["weight_decay"])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
