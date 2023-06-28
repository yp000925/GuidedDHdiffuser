import os
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# random.seed(11)
random.seed(28)
data_dir = '/Users/zhangyunping/PycharmProjects/PLholo/experimental_data/cell'
output_dir = '/Users/zhangyunping/PycharmProjects/improved-diffusion-main/data/augmented_cell2'
crop_size = 768
crop_num = 1000
rotation = [0, 90, 180, 270]
image_files = os.listdir(data_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for i in range(crop_num):
    image_file = random.choice(image_files)
    if image_file.endswith('.tiff'):
        image = Image.open(os.path.join(data_dir, image_file))
        width, height = image.size
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        right = left + crop_size
        bottom = top + crop_size
        cropped_img = image.crop((left, top, right, bottom))
        rotation_angle = random.choice(rotation)
        rotated_img = cropped_img.rotate(rotation_angle)
        new_fname = f"{i}.png"
        new_img = rotated_img.convert('RGB').resize((256, 256))
        new_img.save(os.path.join(output_dir, new_fname))
        if (i+1) % 100 == 0:
            print(f"{i} images have been generated.")
    else:
        print(image_file)
        i-=1
        pass
