import os

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
import pandas as pd

grid_org = Image.open("img_template/grid.png")
legends_org = Image.open("img_template/legends.png")
background_org = Image.open("img_template/blank.png")

background_size = 512
dataset_sizes = [50000, 5000]
# dataset_sizes = [20, 10]

def draw_circles(image, x, y, r=1, color=(255, 0, 0, 255)):
    draw = ImageDraw.Draw(image)
    leftUpPoint = (x - r, y - r)
    rightDownPoint = (x + r, y + r)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, fill=color)



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
        if i % 100 == 0:
            print(f'{i} / {split_dataset_size}')
        legend_size = np.random.randint(200, 512)
        grid_size = int(legend_size * 0.9)
        division_factor = np.random.randint(1, 5)
        recolored_grid = Image.fromarray(np.array(grid_org) // division_factor)

        legends = legends_org.resize((legend_size, legend_size))
        grid = recolored_grid.resize((grid_size, grid_size))

        rand_num_circles = np.random.randint(0, 12)
        for _ in range(rand_num_circles):
            random_x_circle = np.random.randint(0, grid_size)
            random_y_circle = np.random.randint(0, grid_size)
            random_circle_r = np.random.randint(grid_size // 40, grid_size // 30)
            random_R = np.random.randint(0, 256)
            random_G = np.random.randint(0, 256)
            random_B = np.random.randint(0, 256)
            draw_circles(grid,
                         x=random_x_circle,
                         y=random_y_circle,
                         r=random_circle_r,
                         color=(random_R, random_G, random_B, 255))

        legends.paste(grid, (legend_size // 10, legend_size // 50), grid)

        background = background_org.resize((background_size, background_size))

        max_offset = background_size - legend_size
        x_offset = np.random.randint(0, max_offset)
        y_offset = np.random.randint(0, max_offset)
        background.paste(legends, (x_offset, y_offset), legends)
        bbox = (x_offset + legend_size // 10,
                y_offset + legend_size // 50,
                x_offset + legend_size // 10 + grid_size,
                y_offset + legend_size // 50 + grid_size)

        dataset_info['img'] += [f'images_{split}/{i}.png']

        dataset_info['x_min'] += [bbox[0]]
        dataset_info['y_min'] += [bbox[1]]
        dataset_info['x_max'] += [bbox[2]]
        dataset_info['y_max'] += [bbox[3]]

        dataset_info['label'] += [20]

        background.save(f'./gen_dataset/images_{split}/{i}.png')

        draw_circles(background, bbox[2], bbox[3])
        draw_circles(background, bbox[0], bbox[1])
        background.save(f'./gen_dataset/points_{split}/{i}.png')
        # background.save(f'./gen_dataset/points_{split}/{i}.png')

    dataset_df = pd.DataFrame(data=dataset_info)
    dataset_df.to_csv(f'./gen_dataset/{split}_dataset_info.csv', header=True, index=False)