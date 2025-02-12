import torch
import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        for param in self.model.parameters():
            param.requires_grad = True
            

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    # // def unfreeze_layers(self, num_layers=0):
    # //    """解冻后几层进行微调"""
    # //    if num_layers > 0:
    # //        for param in list(self.model.parameters())[-num_layers:]:
    # //            param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)

class WideResNet50(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet50, self).__init__()
        self.model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        
        for param in self.model.parameters():
            param.requires_grad = True
            

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    

    def forward(self, x):
        return self.model(x)
    
    
    
class WideResNet101(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet101, self).__init__()
        
        self.model = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V2)
        
        for param in self.model.parameters():
            param.requires_grad = True
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
    
class VIT_b_16(nn.Module):
    def __init__(self, num_classes):
        super(VIT_b_16, self).__init__()
        
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        
        for param in self.model.parameters():
            param.requires_grad = True
        
        in_features = self.model.heads[0].in_features
        self.model.heads = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
    
class EfficientNet(nn.Module):    
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        
        self.model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        
        for param in self.model.parameters():
            param.requires_grad = True
            
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
    
    
class EfficientNet_M(nn.Module):    
    def __init__(self, num_classes):
        super(EfficientNet_M, self).__init__()
        
        self.model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        
        for param in self.model.parameters():
            param.requires_grad = True
            
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)