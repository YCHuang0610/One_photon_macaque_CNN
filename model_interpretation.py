from config import config
from net.MyCNN import MyCNN
from utilis.data_loader import My_twoPhoton_Dataset, transform

import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def PearsonCorr(output, label):
    '''
    output: [1, n] torch
    label: [1, n] torch
    '''
    # 计算相关系数
    output = output.squeeze()
    label = label.squeeze()
    output_mean = torch.mean(output)
    label_mean = torch.mean(label)
    output_std = torch.std(output)
    label_std = torch.std(label)
    n = output.shape[0]
    corr = torch.sum((output - output_mean) * (label - label_mean)) / (n * output_std * label_std)
    return corr

class GradCAM:
    def __init__(self, model, last_conv_layer, before_last_conv_layer):
        self.model = model
        self.last_conv_layer = last_conv_layer
        self.before_last_conv_layer = before_last_conv_layer

    def generate_heatmap(self, image, label):
        '''
                
        如果您的目标不是单个预测类别，而是一个有固定长度的向量（例如，在回归任务、多标签分类任务或任何其他输出是向量的情况下），Grad-CAM 的基本原理仍然适用，
        但其具体实现方式需要根据具体任务稍作调整。在这些情况下，您关注的是模型输出向量中特定元素或多个元素对输入图像特征的影响。以下是几种不同情景下如何应用 Grad-CAM 的指导：

        多标签分类
        在多标签分类问题中，模型的输出是一个向量，其中每个元素代表一个类别的预测概率。如果您想要可视化特定类别（或几个特定类别）对应的激活图：

        单个类别：选择输出向量中对应于感兴趣类别的元素，将这个单一得分相对于最后一个卷积层输出的梯度进行反向传播。这与单类别分类任务相似，只是您关注的是向量中特定的元素。
        多个类别：可以通过将选定类别的得分相加（或其他合适的聚合操作）来创建一个合成得分，然后对这个得分相对于最后一个卷积层输出的梯度进行反向传播。

        回归任务
        在回归任务中，模型输出一个或多个连续值。如果您的目标是一个具有固定长度的向量，那么您可能关心模型对这个向量中每个维度的预测是如何与输入图像的特定区域相关联的。

        单个维度：您可以选择向量中的一个元素（即一个特定的回归目标），并计算此元素相对于最后一个卷积层输出的梯度。
        整个向量：如果您希望得到一个整体的视图，表明输入图像如何整体影响向量输出，可以考虑将输出向量的元素进行合适的聚合（例如，求和或平均），
        然后计算这个聚合得分相对于最后一个卷积层输出的梯度。

        实现提示
        对于非分类任务，您可能需要定义一个自定义的损失函数或得分函数，以选择或聚合模型输出向量的哪一部分来进行反向传播。
        例如，如果您对输出向量的特定元素感兴趣，您的损失函数可以是这个特定元素的值（或与真实值的差异，如果是在训练或验证过程中评估）。
        然后，您可以使用这个"损失"来进行反向传播，获取相应的梯度，并应用 Grad-CAM 算法的其余步骤。

        总之，即使您的目标是一个有固定长度的向量，Grad-CAM 依旧可以提供有用的视觉解释，帮助您理解模型是如何从输入图像中提取信息以做出预测的。
        只是实现方式需要根据您的具体任务和目标稍作调整。
        '''
        # 前向传播
        self.model.zero_grad()
        output = self.model(image)

        # 反向传播
        # target 设置为output和label之间的相关系数
        target = PearsonCorr(output, label)
        print(target)
        target.backward()

        # 获取梯度
        grads = self.last_conv_layer.weight.grad #[128, 64, 3, 3]
        # 对梯度进行全局平均池化
        pooled_gradients = torch.mean(grads, dim=[1, 2, 3]) # [128]
        # 获取最后一个卷积层的输出
        intermediate_output = self.before_last_conv_layer(image.detach())
        last_conv_layer_output = self.last_conv_layer(intermediate_output) # [1, 128, 28, 28]
        # 对每个通道的输出乘以对应的梯度 [1, 128, 28, 28]
        k = config['k']
        for i in range(k[3]):
            last_conv_layer_output[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(last_conv_layer_output, dim=1).squeeze().cpu().detach().numpy()

        return heatmap

    def visualize_gradcam(self, image_path, label, save_path='plots/grad_cam.png'):
        label = torch.tensor(label).to(device=device, dtype=torch.float32).unsqueeze(0)
        image = Image.open(image_path).convert('RGB')
        image_trans = transform(image).to(device=device, dtype=torch.float32).unsqueeze(0)

        cam = self.generate_heatmap(image_trans, label)

        cam = cv2.resize(cam, (image.width, image.height))

        cam = np.maximum(cam, 0)

        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        # 转换为灰度图
        heatmap = np.uint8(255 * cam)

        # 上采样到原始图像大小
        heatmap = cv2.resize(heatmap, (image.width, image.height))

        # 将热力图转换为PIL图像
        CAM_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 合并cam和原始图像并保存
        CAM_colormap_with_image = cv2.addWeighted(np.array(image), 0.6, CAM_colormap, 0.4, 0)
        cv2.imwrite(save_path, CAM_colormap_with_image)



te_image_list = os.listdir(config['test_pic_dir'])
te_image_path_list = [os.path.join(config['test_pic_dir'], x) for x in te_image_list]
te_label = np.load(config['test_label'])
test_dataset = My_twoPhoton_Dataset(te_image_path_list, te_label)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

model = MyCNN(te_label.shape[1],
                k=config['k'],
                kernel_size=config['kernel_size'],
                stride=config['stride'],
                padding=config['padding'],
                bias=config['bias'],
                fc_hidden_units=config['fc_hidden_units'])

model.load_state_dict(torch.load(f"model/best_model.pth"))
model.eval()
model.to(device=device)

# 测试模型
for images, labels in test_loader:
    images = images.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(images)

# 转换为numpy
outputs = outputs.cpu().numpy()
labels = labels.cpu().numpy()

# 计算outputs和labels每行之间的相关系数
############# 1. 相关系数分布
correlation = []
for i in range(outputs.shape[0]):
    correlation.append(pearsonr(outputs[i], labels[i])[0])

# 计算平均相关系数
mean_correlation = np.mean(correlation)
print(mean_correlation)

plt.hist(correlation, bins=50, edgecolor='black')
plt.title('Correlation Distribution')
plt.xlabel('Correlation')
plt.ylabel('Frequency')
plt.savefig('plots/test_correlation_distribution.png')

############## 2. 最高和最低的图片
## 2.1 相关系数最小的图片
# 找到第几张图片的相关系数最小
plt.figure(figsize=(10, 10))
for i in range(9):
    image = Image.open(te_image_path_list[np.argsort(correlation)[i]]).convert('RGB')
    plt.subplot(3, 3, i+1)
    plt.imshow(image)
    plt.title(f'correlation: {np.sort(correlation)[i]}')
plt.savefig('plots/worst_correlation_images.png')

## 2.2 相关系数最大的图片
# 把相关系数最高的前九张图片显示出来，每行显示3张图片
plt.figure(figsize=(10, 10))
for i in range(9):
    image = Image.open(te_image_path_list[np.argsort(correlation)[-i-1]]).convert('RGB')
    plt.subplot(3, 3, i+1)
    plt.imshow(image)
    plt.title(f'correlation: {np.sort(correlation)[-i-1]}')
plt.savefig('plots/best_correlation_images.png')

############## 3. 利用Grad-CAM可视化模型
# 选择一个图片，计算其Grad-CAM
# 选择相关最差的图片
grad_cam = GradCAM(model, model.features[-3], model.features[:9])
# # 选择相关最好的图片
# grad_cam.visualize_gradcam(te_image_path_list[np.argsort(correlation)[-1]], 
#                            labels[np.argsort(correlation)[-1]],
#                            save_path='plots/best_correlation_grad_cam.png')
# grad_cam.visualize_gradcam(te_image_path_list[np.argsort(correlation)[0]], 
#                            labels[np.argsort(correlation)[0]],
#                            save_path='plots/worst_correlation_grad_cam.png')

# 循环前九张相关系数最低的图片，计算其Grad-CAM
for i in range(9):
    grad_cam.visualize_gradcam(te_image_path_list[np.argsort(correlation)[i]], 
                               labels[np.argsort(correlation)[i]],
                               save_path=f'plots/worst_correlation_grad_cam/{i}.png')
    
# 循环前九张相关系数最高的图片，计算其Grad-CAM
for i in range(9):
    grad_cam.visualize_gradcam(te_image_path_list[np.argsort(correlation)[-i-1]], 
                               labels[np.argsort(correlation)[-i-1]],
                               save_path=f'plots/best_correlation_grad_cam/{i}.png')





