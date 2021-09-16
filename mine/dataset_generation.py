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

def draw_rectangle(image, x, y, r=1, color=(255, 0, 0, 255)):
    draw = ImageDraw.Draw(image)

    r *= 1.5

    random_quarter = np.random.randint(0, 4)
    random_pos = np.random.uniform(0, r)

    a = 4 * r # length
    b = 2 * r # width


    if random_quarter == 0: # top left
        bottom_right_x = x - random_pos
        bottom_right_y = y - np.sqrt((r**2) - (random_pos ** 2))

        top_left_x = bottom_right_x - a
        top_left_y = bottom_right_y - b

    elif random_quarter == 1: # top right
        bottom_right_x = x + random_pos + a
        bottom_right_y = y - np.sqrt((r ** 2) - (random_pos ** 2))

        top_left_x = bottom_right_x - a
        top_left_y = bottom_right_y - b

    elif random_quarter == 2: # bottom right

        top_left_x = x + random_pos
        top_left_y = y + np.sqrt((r ** 2) - (random_pos ** 2))

        bottom_right_x = top_left_x + a
        bottom_right_y = top_left_y + b

    elif random_quarter == 3: # bottom left
        top_left_x = x - random_pos - a
        top_left_y = y + np.sqrt((r ** 2) - (random_pos ** 2))

        bottom_right_x = top_left_x + a
        bottom_right_y = top_left_y + b

    leftUpPoint = (top_left_x, top_left_y)
    rightDownPoint = (bottom_right_x, bottom_right_y)
    twoPointList = [leftUpPoint, rightDownPoint]

    draw.rectangle(twoPointList, fill=color)


def draw_circles(image, x, y, r=1, color=(255, 0, 0, 255)):
    draw = ImageDraw.Draw(image)

    r_bigger = r * 1.5

    leftUpPoint = (x - r_bigger, y - r_bigger)
    rightDownPoint = (x + r_bigger, y + r_bigger)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, fill=color)

    r_bigger = r_bigger * 0.90
    leftUpPoint = (x - r_bigger, y - r_bigger)
    rightDownPoint = (x + r_bigger, y + r_bigger)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, fill=(255, 255, 255, 255))

    leftUpPoint = (x - r, y - r)
    rightDownPoint = (x + r, y + r)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, fill=color)

    draw_rectangle(image, x, y, r, color)


base_ds = './gen_dataset/new_shit_ha'
os.mkdir(base_ds)
for split_dataset_size, split in zip(dataset_sizes, ['TRAIN', 'TEST']):

    os.mkdir(os.path.join(base_ds, f'images_{split}'))
    os.mkdir(os.path.join(base_ds, f'points_{split}'))

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
        division_factor = np.random.randint(1, 4)
        recolored_grid = Image.fromarray(np.array(grid_org) // division_factor)

        legends = legends_org.resize((legend_size, legend_size))
        grid = recolored_grid.resize((grid_size, grid_size))

        rand_num_circles = np.random.randint(0, 12)

        legends.paste(grid, (legend_size // 10, legend_size // 50), grid)

        background = background_org.resize((background_size, background_size))

        max_offset = background_size - legend_size
        x_offset = np.random.randint(0, max_offset)
        y_offset = np.random.randint(0, max_offset)
        background.paste(legends, (x_offset, y_offset), legends)

        red_color = (215, 33, 33, 255)
        green_color = (61, 155, 53, 255)
        blue_color = (9, 40, 85, 255)
        orange_color = (244, 128, 4, 255)
        you_color = (76, 78, 77, 255)
        purple_color = (68, 45, 123, 255)

        colors = [red_color,
                  green_color,
                  blue_color,
                  orange_color,
                  purple_color,
                  you_color]

        for color in colors:
            random_x_circle = np.random.randint(int(x_offset) + legend_size // 10, grid_size + int(x_offset) + legend_size // 10)
            random_y_circle = np.random.randint(int(y_offset) + legend_size // 50, grid_size + int(y_offset) + legend_size // 50)
            random_circle_r = np.random.randint(grid_size // 40, grid_size // 30)

            draw_circles(background,
                         x=random_x_circle,
                         y=random_y_circle,
                         r=random_circle_r,
                         color=color)

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

        background.save(os.path.join(base_ds, f'images_{split}/{i}.png'))

        draw_circles(background, bbox[2], bbox[3])
        draw_circles(background, bbox[0], bbox[1])
        background.save(os.path.join(base_ds, f'points_{split}/{i}.png'))
        # background.save(f'./gen_dataset/points_{split}/{i}.png')

    dataset_df = pd.DataFrame(data=dataset_info)
    dataset_df.to_csv(os.path.join(base_ds, f'{split}_dataset_info.csv'), header=True, index=False)