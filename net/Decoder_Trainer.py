import os

import torch
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Decoder_trainer:
    def __init__(self, 
                 Decoder, 
                 optimizer, 
                 scheduler=None, 
                 loss_function=torch.nn.MSELoss, 
                 device=device):
        self.Decoder = Decoder.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = loss_function
        self.best_val_loss = np.inf

    def train_each_epoch(self, train_loader):
        self.Decoder.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
    
            self.optimizer.zero_grad()

            outputs = self.Decoder(labels)
            loss = self.criterion(outputs, images)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * images.size(0)

        # Stop the scheduler after each epoch
        if self.scheduler is not None:
            self.scheduler.step()
        
        train_loss = train_loss / len(train_loader.dataset)
        #train_corr = train_corr / len(train_loader.dataset)
        return train_loss
    
    def valEvaluate_each_epoch(self, val_loader):
        self.Decoder.eval()
        val_loss = 0
        val_corr = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.Decoder(labels)

                loss = self.criterion(outputs, images)
                val_loss += loss.item() * images.size(0)
                #val_corr += batch_pearsonr(outputs, labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        #val_corr = val_corr / len(val_loader.dataset)
        return val_loss#, val_corr

    def train(self, train_loader, val_loader, num_epoch=100):
        train_loss_list = []
        #train_corr_list = []
        val_loss_list = []
        #val_corr_list = []

        print("Start training...")
        for epoch in range(num_epoch):
            train_loss = self.train_each_epoch(train_loader)
            val_loss = self.valEvaluate_each_epoch(val_loader)

            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save the model if the validation loss is the best we've seen so far
            if val_loss < self.best_val_loss:
                print(f'Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}). Saving model...')
                if not os.path.exists('model'):
                    os.makedirs('model')
                torch.save(self.Decoder.state_dict(), f'model/best_Decoder_model.pth')
                self.best_val_loss = val_loss

            train_loss_list.append(train_loss)
            #train_corr_list.append(train_corr)
            val_loss_list.append(val_loss)
            #val_corr_list.append(val_corr)
        
        return train_loss_list, val_loss_list