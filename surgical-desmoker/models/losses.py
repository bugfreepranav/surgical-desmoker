import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_norm), self.vgg(target_norm))

class TotalDesmokingLoss(nn.Module):
    def __init__(self, lambda_smoke=0.1, lambda_perc=0.04):
        super().__init__()
        self.lambda_smoke = lambda_smoke
        self.lambda_perc = lambda_perc
        self.perceptual = PerceptualLoss()

    def forward(self, J_pred, J_gt, S_pred, S_gt):
        l1_img = F.l1_loss(J_pred, J_gt)
        l1_smoke = F.l1_loss(S_pred, S_gt)
        perc = self.perceptual(J_pred, J_gt)
        return l1_img + (self.lambda_smoke * l1_smoke) + (self.lambda_perc * perc)
