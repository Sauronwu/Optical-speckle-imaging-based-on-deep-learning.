import os

import torch
from matplotlib import pyplot as plt
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from net import  *
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

main_folder_path = 'results_folder'
os.makedirs(main_folder_path, exist_ok=True)

class Data_test(Dataset):
    def __init__(self,filepath):
        data = np.load(filepath)
        self.len = data['padded_test'].shape[0]
        self.object = torch.from_numpy(data['padded_test'])
        self.speckle = torch.from_numpy(data['scattered_test'])

    def __getitem__(self, index):
        return self.object[index], self.speckle[index]

    def __len__(self):
        return self.len


test_data = Data_test('testing_data.npz')

for i in range(0,100):
    a, b = test_data[i]
    target, inputs = a.to(device), b.to(device)
    target = target.view(1,1,512,512)
    inputs = inputs.view(1,1,512,512)

    net = torch.load('trained_model.pth') # trained
    net = net.to(device)
    weight_type = next(net.parameters()).dtype
    # 将 inputs 张量的数据类型更改为与权重相同的类型
    inputs = inputs.to(weight_type)
    target = target.to(weight_type)


    # 创建主文件夹
    main_folder_path = 'results_folder'
    os.makedirs(main_folder_path, exist_ok=True)


    # 创建子文件夹
    sub_folder_path = os.path.join(main_folder_path, f'results_{i}')
    os.makedirs(sub_folder_path, exist_ok=True)


    # 推理并保存结果图片
    with torch.no_grad():
        outputs = net(inputs)
        outputs_np = outputs.cpu().numpy()
        inputs_np = inputs.cpu().numpy()
        target_np = target.cpu().numpy()

        # 保存图片
        plt.imsave(os.path.join(sub_folder_path, 'output.png'), outputs_np[0, 0, :, :], cmap='gray')
        plt.imsave(os.path.join(sub_folder_path, 'input.png'), inputs_np[0, 0, :, :], cmap='gray')
        plt.imsave(os.path.join(sub_folder_path, 'target.png'), target_np[0, 0, :, :], cmap='gray')

