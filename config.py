import os

config = {
    # 数据集相关
    'train_pic_dir': os.path.join('data', 'Pics'),
    'train_label': os.path.join('data', 'Rsp_train.npy'),
    'test_pic_dir': os.path.join('data', 'valPics'),
    'test_label': os.path.join('data', 'Rsp_val.npy'),
    # 数据读取与转换
    'transform': True,
    'transform_resize': (224, 224),
    'transform_mean': (0.5, 0.5, 0.5),
    'transform_std': (0.5, 0.5, 0.5),
    # 训练验证分割
    'val_split': 5,
    # 模型超参数相关
    'k': (16, 32, 64, 128),
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'bias': True,
    'fc_hidden_units': 8192,
    # 训练相关
    'batch_size': 256,
    'num_epoch': 70,
    ## optimizer
    'optimizer': 'SGD',
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    ## scheduler
    'step_size': 20,
    'gamma': 0.8,
    # 测试
    'test': True
}