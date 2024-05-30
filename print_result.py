import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def load_images(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, filename)
            image = io.imread(img_path)
            images[filename] = image
            print(f"Loaded {filename} with shape {image.shape}")  # 打印图片尺寸和类型
    return images

def calculate_metrics(images, complement_images):
    psnr_values = []
    ssim_values = []
    filenames = []
    for filename in images.keys():
        if filename in complement_images:
            img_original = images[filename]
            img_complement = complement_images[filename]
            print(f"Processing {filename} - Original shape: {img_original.shape}, Complement shape: {img_complement.shape}")
            psnr_values.append(psnr(img_original, img_complement))
            # 更新 SSIM 调用，指定 channel_axis
            ssim_values.append(ssim(img_original, img_complement, multichannel=True, channel_axis=-1))
            filenames.append(filename)
    return filenames, psnr_values, ssim_values


def plot_metrics(filenames, values, title, ylabel, file_name):
    """绘制指标的折线图，并保存到文件。"""
    plt.figure(figsize=(10, 5))
    plt.plot(filenames, values, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Filename')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name)  # 保存图像到文件
    plt.show()

# 加载图片
images = load_images('images')
complement_images = load_images('complement')

# 计算 PSNR 和 SSIM
filenames, psnr_vals, ssim_vals = calculate_metrics(images, complement_images)

# 分别绘制 PSNR 和 SSIM 图
plot_metrics(filenames, psnr_vals, 'PSNR Values', 'PSNR', 'psnr_plot.png')
plot_metrics(filenames, ssim_vals, 'SSIM Values', 'SSIM', 'ssim_plot.png')
