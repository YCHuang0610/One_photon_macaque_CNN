from config import config
from net.MyCNN import MyCNN, DecoderCNN
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
    encoder = MyCNN(tr_label.shape[1],
                  k=config['k'],
                  kernel_size=config['kernel_size'],
                  stride=config['stride'],
                  padding=config['padding'],
                  bias=config['bias'],
                  fc_hidden_units=config['fc_hidden_units'])
    decoder = DecoderCNN(tr_label.shape[1],
                  k=config['k'],
                  kernel_size=config['kernel_size'],
                  stride=config['stride'],
                  padding=config['padding'],
                  bias=config['bias'],
                  fc_hidden_units=config['fc_hidden_units'])
    
    # load encoder model
    encoder.load_state_dict(torch.load(f"model/best_model.pth"))
    encoder.eval()
    encoder.to(device)

    # train decoder model
    decoder.to(device)
    criterion = torch.nn.MSELoss()
    # Set up the optimizer and scheduler
    optimizer = torch.optim.Adam(decoder.parameters(), 
                                    lr=config['learning_rate'],
                                    weight_decay=config['weight_decay'])
    # optimizer = torch.optim.SGD(decoder.parameters(),
    #                             lr=config['learning_rate'],
    #                             momentum=config['momentum'],
    #                             weight_decay=config['weight_decay'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config['step_size'],
                                                gamma=config['gamma'])
    
    # Train the decoder model
    num_epochs = config['num_epoch']
    tr_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        decoder.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            encoder_output = encoder(images)
            decoder_output = decoder(encoder_output)
            
            # Compute loss
            loss = criterion(decoder_output, images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # Stop the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss / len(train_loader.dataset)

        # validate the model
        decoder.eval()
        val_loss = 0.0
        min_val_loss = np.inf
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                encoder_output = encoder(images)
                decoder_output = decoder(encoder_output)
                
                # Compute loss
                loss = criterion(decoder_output, images)
                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        tr_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # Save the best model
        if val_loss < min_val_loss:
            torch.save(decoder.state_dict(), f"model/best_decoder_model.pth")
            print(f"Validation loss decreased ({min_val_loss:.6f} --> {val_loss:.6f}). Saving model...")
            min_val_loss = val_loss

    # Plot the training process
    plt.style.use('default')
    plt.plot(tr_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.title('Train/Val Loss')
    plt.legend()
    plt.savefig("plots/decoder_training_process.png")




    
    