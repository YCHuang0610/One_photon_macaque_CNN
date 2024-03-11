from matplotlib import pyplot as plt
import os
import numpy as np

def plot_Training_process(train_loss_list, train_acc_list, test_loss_list, test_acc_list, save_path=None):
    """
    画图函数，
    输入为：
    train loss;
    train acc;
    test loss;
    test acc
    """
    plt.style.use('default')
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    axs[0].plot(train_loss_list, label='Train Loss')
    axs[0].plot(test_loss_list, label='Test Loss')
    axs[0].set_title('Train/Test Loss')
    axs[0].legend()
    # 将train_acc_list和test_acc_list画在一张图上
    axs[1].plot(train_acc_list, label='Train Acc')
    axs[1].plot(test_acc_list, label='Test Acc')
    axs[1].set_title('Train/Test Accuracy')
    axs[1].legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), 
                    exist_ok=True)
        plt.savefig(save_path)