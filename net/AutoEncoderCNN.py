import torch 
from torch import nn
import torch.nn.functional as F

class AutoEncoderCNN(nn.Module):
    def __init__(self, label_length, 
                 x_shape=(3, 224, 224),
                 k=(16, 32, 64, 128), 
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias=True,
                 fc_hidden_units=512):
        super(AutoEncoderCNN, self).__init__()
        self.k = k
        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(3, self.k[0], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.k[0], self.k[1], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.k[1], self.k[2], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.k[2], self.k[3], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 反卷积层
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.k[3], self.k[2], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(self.k[2], self.k[1], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(self.k[1], self.k[0], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(self.k[0], 3, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # 动态计算全连接层输入尺寸
        with torch.no_grad():
            self._to_linear = None
            self.forward_conv(torch.randn(1, *x_shape))
        
        # TODO: 这里用变分自编码器取代线性层可能更好
        # 线性层
        self.linearMapping = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._to_linear, fc_hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, label_length),
            nn.ReLU(inplace=True)
        )
        # 反线性层
        self.deLinearMapping = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(label_length, fc_hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, self._to_linear),
            nn.ReLU(inplace=True)
        )
    
    def forward_conv(self, x):
        if self._to_linear is None:
            x = self.conv(x)
            self._to_linear = x.view(1, -1).size(1)
        else:
            x = self.conv(x)
        self.feature_map_size = x.shape[2:]
        return x
    
    def forward_deconv(self, x):
        x = self.deconv(x)
        return x
    
    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)
        mu = self.linearMapping(x)
        x = self.deLinearMapping(mu)
        x = x.view(-1, self.k[3], *self.feature_map_size)
        x = self.forward_deconv(x)
        return x, mu