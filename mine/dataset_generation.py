import os

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
import pandas as pd

foreground_org = Image.open("./img_template/grid1.png")
background_org = Image.open("./gen_dataset.png")

background_size = 512
# dataset_sizes = [10000, 1000]
dataset_sizes = [20, 10]

def draw_circles(image, x, y, r=1):
    draw = ImageDraw.Draw(image)
    leftUpPoint = (x - r, y - r)
    rightDownPoint = (x + r, y + r)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, fill=(255, 0, 0, 255))



for split_dataset_size, split in zip(dataset_sizes, ['TRAIN', 'TEST']):

    os.mkdir(f'./gen_dataset/images_{split}')
    os.mkdir(f'./gen_dataset/points_{split}')

    dataset_info = {'img': [],
                    'x_min': [],
                    'y_min': [],
                    'x_max': [],
                    'y_max': [],
                    'label': []}

    for i in range(split_dataset_size):
        grid_size = np.random.randint(50, 512)
        max_offset = background_size - grid_size
        foreground = foreground_org.resize((grid_size, grid_size))
        background = background_org.resize((background_size, background_size))
        x_offset = np.random.randint(0, max_offset)
        y_offset = np.random.randint(0, max_offset)
        background.paste(foreground, (x_offset, y_offset), foreground)
        bbox = (x_offset, y_offset, x_offset + grid_size, y_offset + grid_size)
        background.save(f'./gen_dataset/images_{split}/{i}.png')
        dataset_info['img'] += [f'images_{split}/{i}.png']

        dataset_info['x_min'] += [bbox[0]]
        dataset_info['y_min'] += [bbox[1]]
        dataset_info['x_max'] += [bbox[2]]
        dataset_info['y_max'] += [bbox[3]]

        dataset_info['label'] += [20]

        draw_circles(background, bbox[2], bbox[3])
        draw_circles(background, bbox[0], bbox[1])
        background.save(f'./gen_dataset/points_{split}/{i}.png')

    dataset_df = pd.DataFrame(data=dataset_info)
    dataset_df.to_csv(f'./gen_dataset/{split}_dataset_info.csv', header=True, index=False)