import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models, transforms
from sklearn.metrics import r2_score
import numpy as np

class CNNmodel(pl.LightningModule):
    def __init__(self, batch_size=6):
        super().__init__()

        self.batch_size = batch_size

        self.predictions, self.ground_truths = [],[]

        self.encoder = torch.nn.Sequential(
            *list(models.vgg16(pretrained=True).children())[:-2],
        )

        self.valueDecoder = torch.nn.Sequential(
            nn.Flatten(),
            # Fully connected layer
            nn.Linear(512*4*4, 4608),
            nn.Linear(4608, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            # output: 1
        )

        self.faceDecoder = torch.nn.Sequential(
            nn.ConvTranspose2d(512, 3, kernel_size=3, stride=2),
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2),
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2),
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2,),
            # output = 3 x 95 x 95
        )

        # self.weights_init(self.encoder)
        # self.weights_init(self.hm3d_xy_Decoder)
        # self.weights_init(self.hm3d_yz_Decoder)
        # self.weights_init(self.hm3d_xz_Decoder)

    def init_layer(self, layer, classname):
        if classname.find('Conv') != -1:
            nn.init.xavier_uniform_(layer.weight)
        elif classname.find('Linear') != -1:
            nn.init.xavier_uniform_(layer.weight)
        elif classname.find('BatchNorm') != -1:
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif classname.find('Bottle') != -1:
            nn.init.xavier_uniform_(layer.conv1.weight)
            nn.init.xavier_uniform_(layer.conv2.weight)
            nn.init.xavier_uniform_(layer.conv3.weight)

    def weights_init(self, model, is_resnet=False):
        if is_resnet:
            for layer in model:
                classname = layer.__class__.__name__
                if 'Seq' in classname:
                    for subLayer in layer:
                        subclassname = subLayer.__class__.__name__
                        self.init_layer(subLayer, subclassname)
                else:
                    self.init_layer(layer, classname)
        else:
            for layer in model:
                classname = layer.__class__.__name__
                self.init_layer(layer, classname)

    def forward(self, x):
        # print('img size:', x.size())
        x_enout = self.encoder(x)
        x_valout = self.valueDecoder(x_enout)
        x_faceout = self.faceDecoder(x_enout)
        # print('x: ', x.size(), ', x_valout: ', x_valout.size(), ', x_faceout: ', x_faceout.size()) 

        return x, x_valout, x_faceout

    def training_step(self, batch, batch_idx):
        image, value = batch
        tensor_value = torch.zeros(self.batch_size, 1).cuda()
        for i in range(self.batch_size):
            tensor_value[i][0] = value[i]
        x, x_val, x_face = self.forward(image)

        lambda_value = 10
        lambda_face = 0.001

        # print('x_val:', x_val.size(), 'tensor_value:', tensor_value.size())
        loss_value = self.cust_loss(x_val, tensor_value) * lambda_value

        new_gt = self.resize_gt(image.cuda(), x_face.size())
        loss_face = F.mse_loss(x_face, new_gt, reduction='sum') * lambda_face

        loss = loss_value + loss_face

        self.log_dict({
            'L': loss,
            'L_v': loss_value,
            'L_face': loss_face,
        }, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, value = batch
        tensor_value = torch.zeros(self.batch_size, 1).cuda()
        for i in range(self.batch_size):
            tensor_value[i] = value[i]
        x, x_val, x_face = self.forward(image)

        val_loss = F.mse_loss(x_val, tensor_value, reduction='sum') * 10

        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        image, value = batch
        tensor_value = torch.zeros(self.batch_size, 1).cuda()
        for i in range(self.batch_size):
            tensor_value[i][0] = value[i]
        x, x_val, x_face = self.forward(image)

        x_val = x_val.cpu().detach().numpy()
        tensor_value = tensor_value.cpu().detach().numpy()

        for i in range(self.batch_size):
            self.ground_truths.append(tensor_value[i])
            self.predictions.append(x_val[i])
        
        loss_r2 = r2_score(x_val, tensor_value)

        self.log_dict({
            'R2': loss_r2,
        }, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        return optimizer

    def cust_loss(self, pred, gt):
        loss = 0
        for i in range(self.batch_size):
            tmp = gt[i][0] - pred[i][0]
            if tmp > 0:
                loss += tmp**2
            else:
                loss += (tmp * 2) ** 2

        return loss
    
    def r2_loss(self, pred, gt, gt_mean):
        # m = torch.ones(gt.size()) * gt_mean
        ss_tot = torch.sum((gt - gt_mean) ** 2)
        ss_res = torch.sum((gt - pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2
    
    def resize_gt(self, gt, pred_size):
        new_gt = torch.zeros(pred_size).cuda()
        trans = transforms.Compose([
            transforms.Resize(pred_size[-1]),
        ])

        for i in range(self.batch_size):
            new_gt[i] = trans(gt[i])

        return new_gt