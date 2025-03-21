"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.types import Device
import torchvision.models as models
import torch.nn.functional as F

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hparams=None):
        super(SegmentationNN, self).__init__()

        self.alexnet = models.alexnet(pretrained=True).features
        for param in self.alexnet.parameters():
            param.requiresGrad = False
        
        # convolution -> normalization -> relu       
        self.encoder1 = nn.Sequential(
          *list(self.alexnet.children())[:2]
          )       # 3  -> 64,  240 -> 59 (64, 59, 59)
        self.down1 = self.alexnet[2]                                            # 64 -> 64,  59 -> 29 (64, 29, 29)
        self.encoder2 = nn.Sequential(
          *list(self.alexnet.children())[3:5]
          )      # 64 -> 192, 29 -> 29 (192, 29, 29)
        self.down2 = self.alexnet[5]                                            # (192, 14, 14)
        self.encoder3 = nn.Sequential(
          *list(self.alexnet.children())[6:10]
          )     # (256, 14, 14)
        self.encoder4 = nn.Sequential(
          *list(self.alexnet.children())[10:12]
          )    # ( 256, 14, 14)
        self.pool4 = self.alexnet[12]  # (192, 6 , 6)


        # 14 -> 29, enc2
        self.up2 = nn.ConvTranspose2d(256, 192, kernel_size = 5, stride = 2, padding = 1)                  # ( 192, 29, 29 ) // 14 -> 29
        self.decoder2 = nn.Sequential(
          nn.Conv2d(in_channels = 192 * 2, out_channels = 192, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(192),
          # nn.ReLU(),
          nn.Tanh(),
          nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(128),
          # nn.ReLU(),
          nn.Tanh(),
          nn.Dropout(0.3)
        )

        # 29 -> 59 enc1 
        self.up1 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 5, stride = 2, padding = 1)                  # ( 64, 59, 59 )

        self.decoder1 = nn.Sequential(
          nn.Conv2d(in_channels = 64 * 2, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),  # ( 64, 59, 59)
          nn.BatchNorm2d(64),
          # nn.ReLU(),
          nn.Tanh(),
          nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),  # ( 64, 59, 59)
          nn.BatchNorm2d(32),
          # nn.ReLU(),
          nn.Tanh(),
          nn.Dropout(0.3)
        )

        self.up0 = nn.ConvTranspose2d(in_channels = 32, out_channels = 23, kernel_size = 8, stride = 4, padding = 0)  # ( 32, 240, 240 )
        self.out_conv = nn.Sequential(
          nn.Conv2d(in_channels = 23, out_channels = 23, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(23),
          nn.Tanh()
        ) 
        

        # # ############# Initialization Part, Dont erase ################
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         nn.init.xavier_normal_(m.weight.data)
        # # ############# Initialization Part, Dont erase ################  
        
        
    def forward(self, x):
      enc1 = self.encoder1(x)
      down1 = self.down1(enc1)                 # 64 -> 64,  59 -> 29 (64, 29, 29)
      enc2 = self.encoder2(down1)              # 64 -> 192, 29 -> 29 (192, 29, 29)
      down2 = self.down2(enc2)                 # (192, 14, 14)
      enc3 = self.encoder3(down2)              # (384, 14, 14)

      up2 = self.up2(enc3)
      cat2 = torch.cat((up2, enc2), dim = 1)
      dec2 = self.decoder2(cat2)

      up1 = self.up1(dec2)
      cat1 = torch.cat((up1, enc1), dim = 1)
      dec1 = self.decoder1(cat1)

      up0 = self.up0(dec1)
      out_conv = self.out_conv(up0)

      return out_conv


    
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()




