import os

import torch
import torch.nn.functional as F
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loss functions
def cosine_similarity_loss(x, y):
    return 1 - F.cosine_similarity(x, y).mean()

def mse_loss(x, y):
    return F.mse_loss(x, y)

def l1_loss(x, y):
    return F.l1_loss(x, y)

def cross_entropy_loss(x, y):
    return F.cross_entropy(x, y)

def batch_pearsonr(x, y):
    """
    计算两个tensor矩阵（x和y）每行之间的皮尔森相关系数。
    参数:
    - x: Tensor of shape (batch_size, features), 模型的输出
    - y: Tensor of shape (batch_size, features), 真实的标签

    返回值:
    - pearson_corr_for_each_row: 每一行的皮尔森相关系数
    """
    mean_x = torch.mean(x, dim=1, keepdim=True)
    mean_y = torch.mean(y, dim=1, keepdim=True)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.mul(ym).sum(1)
    r_den = torch.sqrt(xm.pow(2).sum(1) * ym.pow(2).sum(1))
    r_den[r_den == 0] = 1e-12 # avoid division by zero
    pearson_corr_for_each_row = r_num / r_den
    return pearson_corr_for_each_row

# Trainer class
class MyTrainer:
    def __init__(self, 
                 model, 
                 optimizer, scheduler=None, 
                 loss_function=mse_loss, 
                 device=device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = loss_function
        self.best_val_loss = np.inf

    def train_each_epoch(self, train_loader):
        self.model.train()
        train_loss = 0
        train_corr = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
    
            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_corr += batch_pearsonr(outputs, labels).sum().item()
        # Stop the scheduler after each epoch
        if self.scheduler is not None:
            self.scheduler.step()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_corr = train_corr / len(train_loader.dataset)
        return train_loss, train_corr
    
    def valEvaluate_each_epoch(self, val_loader):
        self.model.eval()
        val_loss = 0
        val_corr = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_corr += batch_pearsonr(outputs, labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_corr = val_corr / len(val_loader.dataset)
        return val_loss, val_corr

    def train(self, train_loader, val_loader, num_epoch=100):
        train_loss_list = []
        train_corr_list = []
        val_loss_list = []
        val_corr_list = []

        print("Start training...")
        for epoch in range(num_epoch):
            train_loss, train_corr = self.train_each_epoch(train_loader)
            val_loss, val_corr = self.valEvaluate_each_epoch(val_loader)

            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Corr: {train_corr:.4f}, Val Loss: {val_loss:.4f}, Val Corr: {val_corr:.4f}')

            # Save the model if the validation loss is the best we've seen so far
            if val_loss < self.best_val_loss:
                print(f'Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}). Saving model...')
                if not os.path.exists('model'):
                    os.makedirs('model')
                torch.save(self.model.state_dict(), f'model/best_model_{epoch}.pth')
                self.best_val_loss = val_loss

            train_loss_list.append(train_loss)
            train_corr_list.append(train_corr)
            val_loss_list.append(val_loss)
            val_corr_list.append(val_corr)
        
        return train_loss_list, train_corr_list, val_loss_list, val_corr_list
    
