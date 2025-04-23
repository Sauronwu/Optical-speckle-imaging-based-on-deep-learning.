import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from net import * # 假设你的神经网络模型类命名为 Net
from skimage.metrics import structural_similarity as ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
class DataTest(Dataset):
    def __init__(self, filepath):
        data = np.load(filepath)
        self.len = data['padded_test'].shape[0]
        self.object = torch.from_numpy(data['padded_test'])
        self.speckle = torch.from_numpy(data['scattered_test'])

    def __getitem__(self, index):
        return self.object[index], self.speckle[index]

    def __len__(self):
        return self.len

test_data = DataTest('testing_data.npz')

# 加载模型
net = torch.load('trained_model.pth')

net = net.to(device)
net.eval()  # 将网络设置为评估模式，这会禁用一些特定于训练的操作

# 设置主文件夹和指标记录文件
main_folder_path = 'results_folder'
os.makedirs(main_folder_path, exist_ok=True)
metrics_file = open(os.path.join(main_folder_path, 'metrics.txt'), 'w')

# 对每张图片进行处理
for i in range(100):
    a, b = test_data[i]
    target, inputs = a.to(device), b.to(device)
    target = target.view(1, 1, 512, 512)
    inputs = inputs.view(1, 1, 512, 512)

    weight_type = next(net.parameters()).dtype
    # 将 inputs 张量的数据类型更改为与权重相同的类型
    inputs = inputs.to(weight_type)
    target = target.to(weight_type)


    with torch.no_grad():
        outputs = net(inputs)
        outputs_np = outputs.cpu().numpy()
        target_np = target.cpu().numpy()

        # 计算MSE和PSNR
        mse_result = mse(target_np[0, 0, :, :], outputs_np[0, 0, :, :])
        psnr_result = psnr(target_np[0, 0, :, :], outputs_np[0, 0, :, :])
        ssim_result = ssim(target_np[0, 0, :, :], outputs_np[0, 0, :, :], data_range=outputs_np.max() - outputs_np.min())
        # 保存MSE和PSNR到文件
        metrics_file.write(f"{i}: MSE={mse_result}, PSNR={psnr_result}, SSIM={ssim_result}\n")

metrics_file.close()

