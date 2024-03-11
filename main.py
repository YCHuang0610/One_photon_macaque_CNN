from config import config
from net.MyCNN import MyCNN
from net.MyCNN_Trainer import MyTrainer
from utilis.data_loader import My_twoPhoton_Dataset, train_val_split
from utilis.plot import plot_Training_process

import os

import torch
from torch.utils.data import DataLoader
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Get_data_ready(config):
    # Load the data
    # get image list
    tr_image_list = os.listdir(config['train_pic_dir'])
    tr_image_path_list = [os.path.join(config['train_pic_dir'], x) for x in tr_image_list]
    tr_label = np.load(config['train_label'])

    te_image_list = os.listdir(config['test_pic_dir'])
    te_image_path_list = [os.path.join(config['test_pic_dir'], x) for x in te_image_list]
    te_label = np.load(config['test_label'])

    return tr_image_path_list, tr_label, te_image_path_list, te_label

def Create_the_dataSet(tr_image_path_list, tr_label, 
                       te_image_path_list, te_label, 
                       val_split=None, cv=False, seed=0):
    # Create the dataset
    train_dataset = My_twoPhoton_Dataset(tr_image_path_list, tr_label)
    test_dataset = My_twoPhoton_Dataset(te_image_path_list, te_label)

    if val_split is not None:
        if cv == True:
            tr_val_split_indices = train_val_split(train_dataset, val_split, seed, cv=True)
            return train_dataset, tr_val_split_indices, test_dataset, 
        else:
            train_dataset, val_dataset = train_val_split(train_dataset, val_split, seed)
            return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Get data ready
    tr_image_path_list, tr_label, te_image_path_list, te_label = Get_data_ready(config)

    # Create the dataset
    train_dataset, val_dataset, test_dataset = Create_the_dataSet(tr_image_path_list, tr_label, 
                                                                  te_image_path_list, te_label, 
                                                                  val_split=config['val_split'], 
                                                                  cv=False, seed=0)
    
    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create the model
    model = MyCNN(tr_label.shape[1],
                  k=config['k'],
                  kernel_size=config['kernel_size'],
                  stride=config['stride'],
                  padding=config['padding'],
                  bias=config['bias'],
                  fc_hidden_units=config['fc_hidden_units'])
    
    # Set up the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config['step_size'],
                                                gamma=config['gamma'])
    
    # Create the trainer
    trainer = MyTrainer(model, 
                        optimizer, 
                        scheduler=scheduler, 
                        loss_function=torch.nn.MSELoss(), 
                        device=device)
    
    # Train the model
    train_loss_list, train_corr_list, val_loss_list, val_corr_list = trainer.train(train_loader=train_loader, 
                                                                                   val_loader=val_loader, 
                                                                                   num_epoch=config['num_epoch'])
    
    # Plot the training process
    plot_Training_process(train_loss_list, train_corr_list, val_loss_list, val_corr_list, 
                          save_path='plots/training_process.png')
    
    # Read the best model
    if config['test'] == True:
        print("===========================================test====================================")
        model.load_state_dict(torch.load(f"model/best_model_{config['test_epoch']}.pth"))
        trainer = MyTrainer(model, optimizer=optimizer, loss_function=torch.nn.MSELoss(), device=device)
        test_loss, test_corr = trainer.valEvaluate_each_epoch(test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Corr: {test_corr:.4f}')
