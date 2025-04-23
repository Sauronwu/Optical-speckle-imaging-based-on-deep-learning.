import torch
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from net import  *
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Data_train(Dataset):
    def __init__(self,filepath):
        data = np.load(filepath)
        self.len = data['padded_train'].shape[0]
        self.object = torch.from_numpy(data['padded_train'])
        self.speckle = torch.from_numpy(data['scattered_train'])

    def __getitem__(self, index):
        return self.object[index], self.speckle[index]

    def __len__(self):
        return self.len

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

lr = 0.000001
train_data = Data_train('training_data_50.npz')
# net = net_model().to(device) # untrained
net = torch.load("trained_model.pth") # trained



test_data = Data_test('testing_data.npz')
train_loader = DataLoader(train_data, 1, True)
test_loader = DataLoader(test_data, 1, True)


net = net.to(device)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)

def train(epoch):
    running_loss = 0.0
    for batch_idx, (target, inputs) in enumerate(train_loader,0):
        target, inputs = target.to(device), inputs.to(device)
        target = target.unsqueeze(1)
        inputs = inputs.unsqueeze(1)

        weight_type = next(net.parameters()).dtype
        # 将 inputs 张量的数据类型更改为与权重相同的类型
        inputs = inputs.to(weight_type)
        target = target.to(weight_type)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if batch_idx %10 == 9 :
        print(f"epoch:{epoch + 1}, batch:{batch_idx +1}, loss:{running_loss}, lr:{lr} ")
        running_loss = 0

# 指定要训练的轮数
num_epochs = 1  # 根据需要调整这个值

# 循环每一轮
for epoch in range(num_epochs):
    # 在每一轮中调用train函数
    train(epoch)



# 指定保存的文件路径
save_path = 'trained_model.pth'

# 使用 torch.save 保存 checkpoint 到文件
torch.save(net, save_path)

# 打印保存成功的消息
print(f'Model saved to {save_path}')

