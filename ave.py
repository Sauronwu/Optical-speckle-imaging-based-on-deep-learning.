import re

# 从文件中读取内容
with open('results_folder/metrics.txt', 'r') as file:
    data = file.read()

# 使用正则表达式提取MSE数值
mse_values = re.findall(r'MSE=([\d.]+)', data)

# 输出提取到的MSE数值
print(mse_values)
# 将字符串列表转换为浮点数列表
mse_values = [float(mse) for mse in mse_values]

# 计算平均值
average_mse = sum(mse_values) / len(mse_values)

# 输出平均值
print("平均MSE值为:", average_mse)
