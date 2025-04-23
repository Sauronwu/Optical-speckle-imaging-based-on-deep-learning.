for i in range (0,101):
    import torch
    from torch.optim import optimizer
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from net import  *
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    a, b = test_data[i]
    target, inputs = a.to(device), b.to(device)
    target = target.view(1,1,512,512)
    inputs = inputs.view(1,1,512,512)

    net = torch.load('trained_model_mnist.pth') # trained
    net = net.to(device)
    weight_type = next(net.parameters()).dtype
    # 将 inputs 张量的数据类型更改为与权重相同的类型
    inputs = inputs.to(weight_type)
    target = target.to(weight_type)

    with torch.no_grad():
        outputs = net(inputs)
        outputs_cpu = outputs.cpu()
        outputs_np = outputs_cpu.numpy()
        inputs_cpu = inputs.cpu()
        inputs_np = inputs_cpu.numpy()
        target_cpu = target.cpu()
        target_np = target_cpu.numpy()

    import matplotlib.pyplot as plt

    # 假设你已经有了 outputs_np、inputs_np 和 target_np 这三个 NumPy 数组

    # 创建一个画布和子图
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 显示 outputs_np
    axs[0].imshow(outputs_np[0, 0, :, :], cmap='gray')
    axs[0].set_title('Outputs')

    # 显示 inputs_np
    axs[1].imshow(inputs_np[0, 0, :, :], cmap='gray')
    axs[1].set_title('Inputs')

    # 显示 target_np
    axs[2].imshow(target_np[0, 0, :, :], cmap='gray')
    axs[2].set_title('Target')

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.5)

    # 显示画布
    plt.show()