import os

from PIL import Image


def corrupt_image(image, mask_size=(50, 50)):
    """在图像中心生成一个白色的矩形"""
    w, h = image.size
    # 计算遮挡的起始位置，使其位于图像中心
    x = w // 2 - mask_size[0] // 2
    y = h // 2 - mask_size[1] // 2
    # 创建一个白色矩形遮挡图像
    for i in range(mask_size[0]):
        for j in range(mask_size[1]):
            image.putpixel((x + i, y + j), (255, 255, 255))


def process_images(input_dir, output_dir, mask_size=(50, 50)):
    """处理指定文件夹内的所有图像"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(input_dir, image_name)
            image = Image.open(img_path)
            corrupt_image(image, mask_size)
            # 保存到输出目录
            image.save(os.path.join(output_dir, image_name))


# 使用示例
input_directory = 'images'  # 将此路径替换为你的输入目录
output_directory = 'damaged'  # 将此路径替换为你的输出目录
process_images(input_directory, output_directory)
