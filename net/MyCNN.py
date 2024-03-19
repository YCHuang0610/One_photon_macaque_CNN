import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCNN_old(nn.Module):
    def __init__(self, label_length, 
                 x_shape=(3, 224, 224),
                 k=(16, 32, 64, 128), 
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias=True,
                 fc_hidden_units=512):
        super(MyCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, k[0], 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding, 
                               bias=bias)
        self.conv2 = nn.Conv2d(k[0], k[1], 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding, 
                               bias=bias)
        self.conv3 = nn.Conv2d(k[1], k[2], 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding, 
                               bias=bias)
        self.conv4 = nn.Conv2d(k[2], k[3], 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding, 
                               bias=bias)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算卷积层输出的特征图大小
        x = torch.randn(1, *x_shape)
        x = self.forward_conv(x)
        num_flatten = x.view(1, -1).size(1)

        # 线性层
        self.fc1 = nn.Linear(num_flatten, fc_hidden_units) 
        self.fc2 = nn.Linear(fc_hidden_units, label_length)
    
    def forward_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        return x
    
    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1) # 展平特征图
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MyCNN(nn.Module):
    def __init__(self, label_length, 
                 x_shape=(3, 224, 224),
                 k=(16, 32, 64, 128), 
                 kernel_size=3, stride=1,
                 padding=0, 
                 bias=True, 
                 fc_hidden_units=512):
        super(MyCNN, self).__init__()

        # 利用nn.Sequential简化卷积层定义
        self.features = nn.Sequential(
            nn.Conv2d(3, k[0], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(k[0], k[1], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(k[1], k[2], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(k[2], k[3], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 动态计算全连接层输入尺寸
        with torch.no_grad():
            self._to_linear = None
            self.forward_conv(torch.randn(1, *x_shape))

        # 线性层
        self.linearMapping = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._to_linear, fc_hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, label_length)
        )
    
    def forward_conv(self, x):
        if self._to_linear is None:
            x = self.features(x)
            self._to_linear = x.view(1, -1).size(1)
        else:
            x = self.features(x)
        return x
    
    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.linearMapping(x)
        return x
    
# 两层CNN
class MyCNN_two_layers(nn.Module):
    def __init__(self, label_length, 
                 x_shape=(3, 224, 224),
                 k=(16, 32, 64, 128), 
                 kernel_size=3, stride=1,
                 padding=0, 
                 bias=True, 
                 fc_hidden_units=512):
        super(MyCNN_two_layers, self).__init__()

        # 利用nn.Sequential简化卷积层定义
        self.features = nn.Sequential(
            nn.Conv2d(3, k[0], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(k[0], k[1], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 动态计算全连接层输入尺寸
        with torch.no_grad():
            self._to_linear = None
            self.forward_conv(torch.randn(1, *x_shape))

        # 线性层
        self.linearMapping = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._to_linear, fc_hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, label_length)
        )
    
    def forward_conv(self, x):
        if self._to_linear is None:
            x = self.features(x)
            self._to_linear = x.view(1, -1).size(1)
        else:
            x = self.features(x)
        return x
    
    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.linearMapping(x)
        return x
    

# 六层    
class MyCNN_six_layers(nn.Module):
    def __init__(self, label_length, 
                 x_shape=(3, 224, 224),
                 k=(16, 32, 64, 128), 
                 kernel_size=3, stride=1,
                 padding=0, 
                 bias=True, 
                 fc_hidden_units=512):
        super(MyCNN_six_layers, self).__init__()

        # 利用nn.Sequential简化卷积层定义
        self.features = nn.Sequential(
            nn.Conv2d(3, k[0], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(k[0], k[1], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(k[1], k[2], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(k[2], k[3], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(k[2], k[3], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(k[2], k[3], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 动态计算全连接层输入尺寸
        with torch.no_grad():
            self._to_linear = None
            self.forward_conv(torch.randn(1, *x_shape))

        # 线性层
        self.linearMapping = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._to_linear, fc_hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, label_length)
        )
    
    def forward_conv(self, x):
        if self._to_linear is None:
            x = self.features(x)
            self._to_linear = x.view(1, -1).size(1)
        else:
            x = self.features(x)
        return x
    
    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.linearMapping(x)
        return x


# 写一个解码器
class DecoderCNN(nn.Module):
    def __init__(self, label_length,
                 k=(16, 32, 64, 128), 
                 kernel_size=3, stride=1,
                 padding=0, bias=True,
                 fc_hidden_units=512,
                 output_shape=(3, 224, 224)):
        super(DecoderCNN, self).__init__()

        # 线性层
        self.linearMapping = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(label_length, fc_hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, 14*14*128),
            nn.ReLU(inplace=True)
        )

        # 反卷积层
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(k[3], k[2], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(k[2], k[1], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(k[1], k[0], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(k[0], 3, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        
    def forward(self, x):
        x = self.linearMapping(x)
        x = x.view(x.size(0), 128, 14, 14)
        x = self.deconv(x)
        return x