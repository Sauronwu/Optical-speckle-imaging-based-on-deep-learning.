from keras.datasets import mnist
import numpy as np
from scipy.ndimage import zoom

# 一、加载MNIST数据集
(_,_), (object_test, label_test) = mnist.load_data()
size_test_set = 100 # 训练集的大小
object_test = object_test[:size_test_set]
label_test = label_test[:size_test_set]
object_test = object_test / 255 # 归一化
upsample_factor = 256/28 # 上采样
object_test = np.array([zoom(image, zoom=(upsample_factor, upsample_factor), order=1) for image in object_test])

# 三、探针
camera_size = 512 # 接收屏大小
padded_test = np.zeros((size_test_set, camera_size, camera_size))
for i in range(size_test_set): # 将每张图像居中放置在probe中
    center_image = object_test[i]
    padded_image = np.pad(center_image, ((camera_size - 256) // 2, (camera_size - 256) // 2), mode='constant')
    padded_test[i] = padded_image

# 二、相位调制
seed_value = 30
np.random.seed(seed_value)
random_phases = np.random.uniform(0, 2 * np.pi, size=(512, 512))
diffuser = np.exp(1j * random_phases)
padded_test_complex = padded_test.astype(np.complex64)
diffused_test = padded_test_complex * diffuser


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

scattered_test = np.zeros_like(diffused_test)  # 创建一个与 diffused 大小相同的数组来存储 scattered 结果

for i in range(size_test_set):
    scattered_test[i] = AS(diffused_test[i], lamb, Z)

scattered_test = abs(scattered_test)

np.savez('testing_data.npz', padded_test=padded_test, scattered_test=scattered_test)

