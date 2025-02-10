import torch
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        # 加载预训练模型
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = True
            
  
          # 修改最后的全连接层
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def unfreeze_layers(self, num_layers=0):
        """解冻后几层进行微调"""
        if num_layers > 0:
            for param in list(self.model.parameters())[-num_layers:]:
                param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)

class WideResNet50(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet50, self).__init__()
        # 加载预训练模型
        self.model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = True
            
  
          # 修改最后的全连接层
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def unfreeze_layers(self, num_layers=0):
        """解冻后几层进行微调"""
        if num_layers > 0:
            for param in list(self.model.parameters())[-num_layers:]:
                param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)
    
    
    
class WideResNet101(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet101, self).__init__()
        # 加载预训练模型
        self.model = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V2)
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = True
            

          # 修改最后的全连接层
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def unfreeze_layers(self, num_layers=0):
        """解冻后几层进行微调"""
        if num_layers > 0:
            for param in list(self.model.parameters())[-num_layers:]:
                param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)