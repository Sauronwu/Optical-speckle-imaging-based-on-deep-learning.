import re
import matplotlib.pyplot as plt
from PIL import Image

# 从文件中读取并提取指标数据
metrics_file = open('results_folder/metrics.txt', 'r')

mse = []
psnr = []
ssim = []

for line in metrics_file:
    m = re.search('MSE=([0-9]+\.[0-9]+)', line)
    p = re.search('PSNR=([0-9]+\.[0-9]+)', line)
    s = re.search('SSIM=([0-9]+\.[0-9]+)', line)

    if m and p and s:
        mse.append(float(m.group(1)))
        psnr.append(float(p.group(1)))
        ssim.append(float(s.group(1)))

metrics_file.close()

# 创建MSE图表，并保存为图片
plt.plot(mse, label='MSE', color='b')
plt.title('Mean Squared Error (MSE)')
plt.xlabel('Image index')
plt.ylabel('Value')
plt.savefig('MSE_Plot.png')
plt.close()

# 创建PSNR图表，并保存为图片
plt.plot(psnr, label='PSNR', color='g')
plt.title('Peak Signal-to-Noise Ratio (PSNR)')
plt.xlabel('Image index')
plt.ylabel('Value')
plt.savefig('PSNR_Plot.png')
plt.close()

# 创建SSIM图表，并保存为图片
plt.plot(ssim, label='SSIM', color='r')
plt.title('Structural Similarity Index (SSIM)')
plt.xlabel('Image index')
plt.ylabel('Value')
plt.savefig('SSIM_Plot.png')
plt.close()
