import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import pandas as pd
from sklearn.model_selection import KFold

from tqdm import tqdm
from datetime import datetime


from models import ResNet50, WideResNet50, WideResNet101
from dataset import ButterflyDataset, ButterflyTestDataset, get_labels


def main():
    train_path = './data/train.csv'
    test_path = './data/test.csv'
    
    train_df = pd.read_csv(train_path)
    
    label_to_idx, idx_to_label, num_classes = get_labels(train_df)
    
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WideResNet101(num_classes=num_classes)
    model = model.to(device)
    
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    dataset_size = len(train_dataset)
    indices = torch.randperm(dataset_size)
    
    epoches = 20
    batch_size = 64
    
    best_val_acc = 0
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
        print(f'FOLD {fold + 1}/{k_folds}')
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epoches):
            print(f'Epoch {epoch + 1}/{epoches}')
            
            train_loop = tqdm(train_loader, 
                        desc=f'Training Epoch {epoch + 1}/{epoches}',
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
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                acc = 100.*correct/total
                
                train_loop.set_postfix({'loss': loss.item(), 'acc': f'{acc:.2f}%'})
            
            print(f"Total Loss: {total_loss:.4f}")
            
            val_loop = tqdm(val_loader,
                           desc=f'Validation Epoch {epoch + 1}/{epoches}',
                           leave=True)
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
                    
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model.pth')
                print(f'Best model saved for fold {fold+1} with accuracy {val_acc:.2f}%')

    test_df = pd.read_csv(test_path)
    
    test_dataset = ButterflyTestDataset(
        csv_file=test_df,
        root_dir='./data/test_images/',
        transform=transform_ops
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    probabilities = []
    
    with torch.no_grad():
        for images in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probs, dim=1)
            probabilities.extend(predicted_class.cpu().numpy())
    
    
    predicted_names = [idx_to_label[idx] for idx in probabilities]
    
    predictions_df = pd.DataFrame({
        'filename': test_df['filename'],
        'label': predicted_names
    })

    predictions_df.to_csv('submission.csv', index=False)
    print("saved to submission.csv")
    
if __name__ == '__main__':
    main()