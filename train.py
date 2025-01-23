import os

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold

from tqdm import tqdm
from datetime import datetime

def get_labels(train_df):
    """获取所有唯一标签并创建映射"""
    # 获取所有唯一标签
    unique_labels = train_df['label'].unique()
    num_classes = len(unique_labels)
    
    # 创建标签到索引的映射
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    return label_to_idx, idx_to_label, num_classes

class ButterflyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = row['label_idx']
        # 转换为张量
        label = torch.tensor(label)
        
        return image, label
    
class ButterflyTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # label = row['label_idx']
        # # 转换为张量
        # label = torch.tensor(label)
        
        return image
    
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        # 加载预训练模型
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
            
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

def main():
    
    train_path = './data/train.csv'
    test_path = './data/test.csv'
    
    train_df = pd.read_csv(train_path)
    
    label_to_idx, idx_to_label, num_classes = get_labels(train_df)
    print(f"Total number of classes: {num_classes}")
    
    print(label_to_idx)
    
    transform_ops = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])
    
    train_df['label_idx'] = train_df['label'].map(label_to_idx)
    
    train_dataset = ButterflyDataset(
        csv_file=train_df,
        root_dir='./data/train_images/',
        transform=transform_ops
    )
    
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet50(num_classes=num_classes)
    model = model.to(device)
    
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    dataset_size = len(train_dataset)
    indices = torch.randperm(dataset_size)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
        print(f'FOLD {fold + 1}/{k_folds}')
        
        # 创建训练集和验证集的数据加载器
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_subsampler)
        val_loader = DataLoader(train_dataset, batch_size=64, sampler=val_subsampler)
        
        # 重置模型
        model = ResNet50(num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        
        train_loop = tqdm(train_loader, 
                    desc=f'Fold {fold + 1}/{k_folds} Training',
                    leave=True)
        val_loop = tqdm(val_loader,
                       desc=f'Fold {fold + 1}/{k_folds} Validation',
                       leave=True)
        model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            acc = 100.*correct/total
            
            train_loop.set_postfix({'loss': loss.item(), 'acc': f'{acc:.2f}%'})
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_acc = 100.*val_correct/val_total
                
                val_loop.set_postfix({'loss': loss.item(), 'acc': f'{val_acc:.2f}%'})

    test_df = pd.read_csv(test_path)
    
    test_dataset = ButterflyTestDataset(
        csv_file=test_df,
        root_dir='./data/test_images/',
        transform=transform_ops
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,  # 可以根据需要调整
        shuffle=False,   # 测试集不需要打乱
    )
    model.eval()
    probabilities = []
    ids = []
    
    with torch.no_grad():
        for images in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            probabilities.extend(probs.cpu().numpy())
    
    # 创建提交文件
    predictions_df = pd.DataFrame({
        'ID': test_df['filename'],
        'TARGET': probabilities
    })
    
    # 保存预测结果
    predictions_df.to_csv('submission.csv', index=False)
    print("预测结果已保存到 submission.csv")
    
if __name__ == '__main__':
    main()