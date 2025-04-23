from keras.datasets import mnist
import numpy as np
from scipy.ndimage import zoom

# 一、加载MNIST数据集
(object_train, label_train), (_, _) = mnist.load_data()
size_train_set = 600 # 训练集的大小
upto = 600*50
begin = upto-600
object_train = object_train[begin:upto]
label_train = label_train[begin:upto]
object_train = object_train / 255 # 归一化
upsample_factor = 256/28 # 上采样
object_train = np.array([zoom(image, zoom=(upsample_factor, upsample_factor), order=1) for image in object_train])

# 三、探针
camera_size = 512 # 接收屏大小
padded_train = np.zeros((size_train_set, camera_size, camera_size))
for i in range(size_train_set): # 将每张图像居中放置在probe中
    center_image = object_train[i]
    padded_image = np.pad(center_image, ((camera_size - 256) // 2, (camera_size - 256) // 2), mode='constant')
    padded_train[i] = padded_image

# 二、相位调制
seed_value = 30
np.random.seed(seed_value)
random_phases = np.random.uniform(0, 2 * np.pi, size=(512, 512))
diffuser = np.exp(1j * random_phases)
padded_train_complex = padded_train.astype(np.complex64)
diffused_train = padded_train_complex * diffuser


# 四、衍射开始
mm = 1e-3
um = 1e-6

M = 512 # 相机的大小
lamb = 0.6328*um # 波长
r = 10*mm # 孔径
L = r/256*camera_size # 256是图像的大小
dx = L/ M # 空间域的采样间隔
Z = M*dx*dx/ lamb # 特征距离
ps = lamb*Z/ L # pixelsize ccd上的像素间隔
ccdsize = camera_size*ps

def AS(U_in,lamb,Z):
    x = np.linspace(-M / 2, M / 2 - 1, M)
    i =   x
    j =   x
    [Fx, Fy] = np.meshgrid(i, j, indexing='ij')
    Fx =  Fx * dx
    Fy =  Fy * dx
    Fx =  Fx / L / dx
    Fy =  Fy / L / dx
    #Fx = Fx * deta
    #Fy = Fy * deta
    k = 2 * np.pi / lamb

    U_out = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(U_in)))
    H = np.exp(1j * k * Z * np.sqrt(1 - (lamb * Fx) ** 2 - (lamb * Fy) ** 2))
    #H = np.exp(1j * 2 * np.pi * Z * np.sqrt(1/(lamb**2) -  Fx ** 2 -  Fy ** 2))
    U_out = U_out * H

    U_out = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(U_out)))
    I1 = np.abs(U_out) * np.abs(U_out)
    I1 = (I1 / np.max(np.max(I1)))
    return I1

scattered_train = np.zeros_like(diffused_train)  # 创建一个与 diffused 大小相同的数组来存储 scattered 结果

for i in range(size_train_set):
    scattered_train[i] = AS(diffused_train[i], lamb, Z)

scattered_train = abs(scattered_train)

np.savez('training_data_50.npz', padded_train=padded_train, scattered_train=scattered_train)










