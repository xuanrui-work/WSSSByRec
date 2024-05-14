import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconLoss(nn.Module):
    def __init__(self, L=1):
        super().__init__()
        self.L = L
    
    def forward(self, x, y_img, y_mask):
        num_classes = y_mask.shape[1]
        y_mask = y_mask.unsqueeze(2)
        loss = 0
        for _ in range(self.L):
            # reparameterization of categorical distribution
            sm = F.gumbel_softmax(y_mask.view(-1, num_classes), hard=True)
            sm = sm.view(y_mask.shape)
            # compose the image from pixelets and masks
            x_hat = torch.sum(sm * y_img, dim=1)
            loss += F.mse_loss(x_hat, x)
        loss /= self.L
        return loss

class MaskRegLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, label, y_mask):
        y_mask = F.softmax(y_mask, dim=1)
        h, w = y_mask.shape[-2:]
        y_pred = 1/(h*w) * torch.sum(y_mask, dim=(-2, -1))
        loss = torch.sum(-label*y_pred + (1-label)*y_pred)
        return loss

class ClsGuideLoss(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        self.classifier = classifier
    
    def forward(self, label, y_img):
        num_classes = y_img.shape[1]
        loss = 0
        for i in num_classes:
            # label 0 is the background class
            if i == num_classes-1:
                x_hat = y_img[:, i]
                y_pred = self.classifier(x_hat)
                y_true = torch.zeros_like(y_pred)
                loss += F.binary_cross_entropy(y_pred, y_true)
            else:
                x_hat = y_img[:, i]
                y_pred = self.classifier(x_hat)
                y_true = torch.zeros_like(y_pred)
                y_true[:, i] = label[:, i]
                loss += F.binary_cross_entropy(y_pred, y_true)
        loss /= num_classes
        return loss
