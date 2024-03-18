from config import config
from net.MyCNN import MyCNN, DecoderCNN
from net.Decoder_Trainer import Decoder_trainer
from utilis.plot import plot_Training_process
from main import Get_data_ready, Create_the_dataSet

import os

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
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
    decoder = DecoderCNN(tr_label.shape[1],
                  k=config['k'],
                  kernel_size=config['kernel_size'],
                  stride=config['stride'],
                  padding=config['padding'],
                  bias=config['bias'],
                  fc_hidden_units=config['fc_hidden_units'])

    # train decoder model
    decoder.to(device)
    # Set up the optimizer and scheduler
    optimizer = torch.optim.Adam(decoder.parameters(), 
                                    lr=0.0001,
                                    weight_decay=config['weight_decay'])
    # optimizer = torch.optim.SGD(decoder.parameters(),
    #                             lr=config['learning_rate'],
    #                             momentum=config['momentum'],
    #                             weight_decay=config['weight_decay'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config['step_size'],
                                                gamma=config['gamma'])
    
    # Train the model\
    trainer = Decoder_trainer(decoder, 
                 optimizer, 
                 scheduler, 
                 loss_function=torch.nn.MSELoss(), device=device)
    
    # Train the decoder model
    train_loss_list, val_loss_list = trainer.train(train_loader, val_loader, config['num_epoch'])


    # Plot the training process
    plt.style.use('default')
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.title('Train/Val Loss')
    plt.legend()
    plt.savefig("plots/decoder_training_process.png")

    # Read the best model
    if config['test'] == True:
        print("=======================================test====================================")
        decoder.load_state_dict(torch.load(f"model/best_decoder_model.pth"))
        trainer = Decoder_trainer(decoder, optimizer=optimizer, loss_function=torch.nn.MSELoss(), device=device)
        test_loss = trainer.valEvaluate_each_epoch(test_loader)
        print(f'Test Loss: {test_loss:.4f}')




    
    