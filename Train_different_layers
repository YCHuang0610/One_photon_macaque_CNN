from config import config
from net.MyCNN import MyCNN, MyCNN_two_layers, MyCNN_six_layers
from net.MyCNN_Trainer import MyTrainer
from utilis.data_loader import My_twoPhoton_Dataset, train_val_split
from utilis.plot import plot_Training_process

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr
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
        
def Train_MyCNN_model(model, train_loader, val_loader, save_model_name='best_model.pth', config=config, device=device):
    # Set up the optimizer and scheduler
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=config['learning_rate'],
                                     weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config['learning_rate'],
                                    momentum=config['momentum'],
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
                                                                                   num_epoch=config['num_epoch'],
                                                                                   save_model_name=save_model_name)
    return train_loss_list, train_corr_list, val_loss_list, val_corr_list

def Test_MyCNN_model(model, test_loader, save_model_name='best_model.pth', device=device):
    model.load_state_dict(torch.load(os.path.join('model', save_model_name)))
    model.eval()
    model.to(device=device)
    # test model
    for images, labels in test_loader:
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(images)
    
    outputs = outputs.cpu().numpy()
    labels = labels.cpu().numpy()

    correlation_rows = []
    for i in range(outputs.shape[0]):
        correlation_rows.append(pearsonr(outputs[i], labels[i])[0])
    print("平均行相关系数：", np.mean(correlation_rows))

    correlation_cols = []
    for i in range(outputs.shape[1]):
        correlation_cols.append(pearsonr(outputs[:, i], labels[:, i])[0])
    print("平均列相关系数：", np.mean(correlation_cols))

    return correlation_rows, correlation_cols



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
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
    
    # Create the model
    for My_model in [MyCNN_two_layers, MyCNN, MyCNN_six_layers]:
        model = My_model(tr_label.shape[1],
                        k=config['k'],
                        kernel_size=config['kernel_size'],
                        stride=config['stride'],
                        padding=config['padding'],
                        bias=config['bias'],
                        fc_hidden_units=config['fc_hidden_units'])
        # Train the model
        train_loss_list, train_corr_list, val_loss_list, val_corr_list = Train_MyCNN_model(model, 
                                                                                           train_loader, 
                                                                                           val_loader,
                                                                                           save_model_name=f'best_model_{My_model.__name__}.pth', 
                                                                                           config=config, 
                                                                                           device=device)
        # Plot the training process
        plot_Training_process(train_loss_list, train_corr_list, val_loss_list, val_corr_list, 
                            save_path=os.path.join('plots/different_layers', f'Training_process_{My_model.__name__}.png'))
        
        if config['test'] == True:
            correlation_rows, correlation_cols = Test_MyCNN_model(model, 
                                                                test_loader, 
                                                                save_model_name=f'best_model_{My_model.__name__}.pth', 
                                                                device=device)
            
            # Save the correlation results
            np.save(os.path.join('output', f'correlation_rows_{My_model.__name__}.npy'), correlation_rows)
            np.save(os.path.join('output', f'correlation_cols_{My_model.__name__}.npy'), correlation_cols)

