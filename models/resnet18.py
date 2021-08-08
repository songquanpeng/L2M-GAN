import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


# Reference: https://github.com/genforce/mganprior/blob/master/inversion/losses.py#L43
class ResNet18(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.pre_processing_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.pre_processing_std = torch.Tensor([0.229, 0.224, 0.225])
        self.img_size = args.img_size
        self.cuda(args.device)

    def cuda(self, device=None):
        self.model.cuda(device=device)
        self.pre_processing_mean = self.pre_processing_mean.cuda(device=device)
        self.pre_processing_std = self.pre_processing_std.cuda(device=device)

    def forward(self, x):
        x = (x * 0.5 + 0.5).sub_(self.pre_processing_mean[:, None, None]).div_(self.pre_processing_std[:, None, None])
        x_features = self.model(F.interpolate(x, size=self.img_size, mode='nearest'))
        return x_features
