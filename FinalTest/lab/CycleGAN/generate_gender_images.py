from imageio import imread, imsave
import cv2
import glob, os
from tqdm import tqdm

data_dir = 'faces'
male_dir = 'faces/male'
female_dir = 'faces/female'

# 创建文件夹
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(male_dir):
    os.mkdir(male_dir)
if not os.path.exists(female_dir):
    os.mkdir(female_dir)

WIDTH = 256
HEIGHT = 256

def read_process_save(read_path, save_path):
    import imageio.v2 as imageio
    image = imageio.imread(read_path)

    # 中心裁剪为正方形（以短边为基准）
    h = image.shape[0]
    w = image.shape[1]
    if h > w:
        image = image[h // 2 - w // 2: h // 2 + w // 2, :, :]
    else:
        image = image[:, w // 2 - h // 2: w // 2 + h // 2, :]

    # 缩放至目标尺寸
    image = cv2.resize(image, (WIDTH, HEIGHT))
    imsave(save_path, image)

target = 'Male'
with open('list_attr_celeba.txt', 'r') as fr:
    lines = fr.readlines()
    all_tags = lines[0].strip('\n').split()
    for i in tqdm(range(1, len(lines))):
        line = lines[i].strip('\n').split()
        if int(line[all_tags.index(target) + 1]) == 1:
            read_process_save(os.path.join('celeba', line[0]), os.path.join(male_dir, line[0])) # 男
        else:
            read_process_save(os.path.join('celeba', line[0]), os.path.join(female_dir, line[0])) # 女