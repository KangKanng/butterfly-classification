import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

def get_labels(train_df):
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
    
transform_ops_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomChoice([
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((180, 180)),
        transforms.RandomRotation((270, 270)),
        transforms.Lambda(lambda x: x)  # 不旋转
    ]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

transform_ops_vit = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomChoice([
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((180, 180)),
        transforms.RandomRotation((270, 270)),
        transforms.Lambda(lambda x: x)  # 不旋转
    ]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

transform_ops_efficientnet_v2_l = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.RandomChoice([
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((180, 180)),
        transforms.RandomRotation((270, 270)),
        transforms.Lambda(lambda x: x)  # 不旋转
    ]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])