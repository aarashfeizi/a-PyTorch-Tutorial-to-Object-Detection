import os
import random

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
import pandas as pd

grid_org = Image.open("img_template/grid.png")
legends_org = Image.open("img_template/legends.png")
background_org = Image.open("img_template/blank.png")
title1 = Image.open("img_template/title1.png")
title2 = Image.open("img_template/title2.png")
title3 = Image.open("img_template/title3.png")

background_size = 512
dataset_sizes = [50000, 5000]
# dataset_sizes = [20, 10]


base_ds = './xy_dataset2'
os.mkdir(base_ds)

LPC_color = (215, 33, 33, 255)
GPC_color = (61, 155, 53, 255)
CPC_color = (9, 40, 85, 255)
BQ_color = (17, 60, 112, 255)
NDP_color = (244, 128, 4, 255)
YOU_color = (76, 78, 77, 255)
PPC_color = (68, 45, 123, 255)

colors = [NDP_color,
          GPC_color,
          YOU_color,
          LPC_color,
          BQ_color,
          CPC_color,
          PPC_color]

color_names = ['NDP',
               'GPC',
               'YOU',
               'LPC',
               'BQ',
               'CPC',
               'PPC']


def draw_bar_graph(image, x, y, l_x=50, l_y=5, gap=20):
    draw = ImageDraw.Draw(image)

    random_color_order = np.random.permutation(colors)

    for clr in random_color_order:

        if tuple(clr) == YOU_color:
            continue

        absolute_leftUpPoint = (x, y)
        absolute_rightDownPoint = (x + l_x, y + l_y)
        absolute_twoPointList = [absolute_leftUpPoint, absolute_rightDownPoint]

        bottom_right_x = np.random.uniform(x, x + l_x)

        rightDownPoint = (bottom_right_x, y + l_y)

        twoPointList = (absolute_leftUpPoint, rightDownPoint)

        draw.rectangle(absolute_twoPointList, fill=(221, 221, 221, 225))
        draw.rectangle(twoPointList, fill=tuple(clr))

        circle_center_x = x - gap * (2 / 5)
        circle_center_y = y - l_y / 4
        r = gap * (2 / 5) - 5
        leftUpPoint = (circle_center_x - r, circle_center_y - r)
        rightDownPoint = (circle_center_x + r, circle_center_y + r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw.ellipse(twoPointList, fill=tuple(clr))

        y += gap


def draw_rectangle(image, x, y, r=1, color=(255, 0, 0, 255)):
    draw = ImageDraw.Draw(image)

    r *= 1.5

    random_quarter = np.random.randint(0, 4)
    random_pos = np.random.uniform(0, r)

    a = 4 * r  # length
    b = 2 * r  # width

    if random_quarter == 0:  # top left
        bottom_right_x = x - random_pos
        bottom_right_y = y - np.sqrt((r ** 2) - (random_pos ** 2))

        top_left_x = bottom_right_x - a
        top_left_y = bottom_right_y - b

    elif random_quarter == 1:  # top right
        bottom_right_x = x + random_pos + a
        bottom_right_y = y - np.sqrt((r ** 2) - (random_pos ** 2))

        top_left_x = bottom_right_x - a
        top_left_y = bottom_right_y - b

    elif random_quarter == 2:  # bottom right

        top_left_x = x + random_pos
        top_left_y = y + np.sqrt((r ** 2) - (random_pos ** 2))

        bottom_right_x = top_left_x + a
        bottom_right_y = top_left_y + b

    elif random_quarter == 3:  # bottom left
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


# create general dataset
def create_dataset():
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

            resize_factor = legend_size / legends_org.size[0]

            division_factor = np.random.randint(1, 4)
            recolored_grid = Image.fromarray(np.array(grid_org) // division_factor)

            legends = legends_org.resize((legend_size, legend_size))
            grid = recolored_grid.resize((grid_size, grid_size))

            rand_title = np.random.randint(0, 3)

            title = None

            diffs = (int((429 - 359) * resize_factor), int((500 - 367) * resize_factor))
            if rand_title == 0:
                title_size_pair = (int(title1.size[0] * resize_factor),
                                   int(title1.size[1] * resize_factor))
                title = title1.resize(title_size_pair)
            elif rand_title == 1:
                title_size_pair = (int(title2.size[0] * resize_factor),
                                   int(title2.size[1] * resize_factor))
                title = title2.resize(title_size_pair)
            elif rand_title == 2:
                title_size_pair = (int(title3.size[0] * resize_factor),
                                   int(title3.size[1] * resize_factor))
                title = title3.resize(title_size_pair)

            if title_size_pair[0] > 512 or title_size_pair[1] > 512:
                title = None

            legends.paste(grid, (legend_size // 10, legend_size // 50), grid)

            if np.random.random() < 0.5:
                twice_big = False
                background = background_org.resize((background_size, background_size))
            else:
                twice_big = True
                background = background_org.resize((background_size * 2, background_size))

            max_offset = background_size - legend_size
            x_offset = np.random.randint(0, max_offset)
            y_offset = np.random.randint(0, max_offset)
            background.paste(legends, (x_offset, y_offset), legends)

            if title:
                x_offset_title = int(x_offset - diffs[0])
                y_offset_title = int(y_offset - diffs[1])

                background.paste(title, (x_offset_title, y_offset_title), title)

            if twice_big:
                if title:
                    bar_x_offset = x_offset + legends.size[0] + diffs[0] * 2
                else:
                    bar_x_offset = x_offset + legends.size[0] * 1.2

                bar_y_offset = y_offset + legends.size[0] * 0.25
                draw_bar_graph(background, bar_x_offset, bar_y_offset,
                               l_x=legends.size[0],
                               l_y=legends.size[1] / 40,
                               gap=legends.size[1] / 7)

            for color in colors:
                random_x_circle = np.random.randint(int(x_offset) + legend_size // 10,
                                                    grid_size + int(x_offset) + legend_size // 10)
                random_y_circle = np.random.randint(int(y_offset) + legend_size // 50,
                                                    grid_size + int(y_offset) + legend_size // 50)
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

        dataset_df = pd.DataFrame(data=dataset_info)
        dataset_df.to_csv(os.path.join(base_ds, f'{split}_dataset_info.csv'), header=True, index=False)


# create xy dataset
for split_dataset_size, split in zip(dataset_sizes, ['TRAIN', 'TEST']):

    os.mkdir(os.path.join(base_ds, f'images_{split}'))
    os.mkdir(os.path.join(base_ds, f'points_{split}'))

    color_names = ['NDP',
                   'GPC',
                   'YOU',
                   'LPC',
                   'BQ',
                   'CPC',
                   'PPC']

    dataset_info = {'img': [],
                    'x_min_NDP': [],
                    'y_min_NDP': [],
                    'x_max_NDP': [],
                    'y_max_NDP': [],
                    'label_NDP': [],
                    'x_min_GPC': [],
                    'y_min_GPC': [],
                    'x_max_GPC': [],
                    'y_max_GPC': [],
                    'label_GPC': [],
                    'x_min_YOU': [],
                    'y_min_YOU': [],
                    'x_max_YOU': [],
                    'y_max_YOU': [],
                    'label_YOU': [],
                    'x_min_LPC': [],
                    'y_min_LPC': [],
                    'x_max_LPC': [],
                    'y_max_LPC': [],
                    'label_LPC': [],
                    'x_min_BQ': [],
                    'y_min_BQ': [],
                    'x_max_BQ': [],
                    'y_max_BQ': [],
                    'label_BQ': [],
                    'x_min_CPC': [],
                    'y_min_CPC': [],
                    'x_max_CPC': [],
                    'y_max_CPC': [],
                    'label_CPC': [],
                    'x_min_PPC': [],
                    'y_min_PPC': [],
                    'x_max_PPC': [],
                    'y_max_PPC': [],
                    'label_PPC': []}

    for i in range(split_dataset_size):
        if i % 100 == 0:
            print(f'{i} / {split_dataset_size}')
        legend_size = np.random.randint(480, 512)
        grid_size = int(legend_size * 0.9)

        resize_factor = legend_size / legends_org.size[0]

        division_factor = np.random.randint(1, 4)
        recolored_grid = Image.fromarray(np.array(grid_org) // division_factor)

        legends = legends_org.resize((legend_size, legend_size))
        grid = recolored_grid.resize((grid_size, grid_size))

        background = background_org.resize((background_size, background_size))
        legends.paste(grid, (legend_size // 10, legend_size // 50), grid)

        max_offset = background_size - legend_size
        x_offset = np.random.randint(-1 * int(0.1 * legend_size) - 5, max_offset + 5)
        y_offset = np.random.randint(-5, max_offset + int(0.1 * legend_size) + 5)
        background.paste(legends, (x_offset, y_offset), legends)

        all_bboxes = []
        for idx, (colorname, color) in enumerate(zip(color_names, colors)):
            if colorname == 'BQ':
                if random.random() < 0.5:
                    dataset_info[f'x_min_{colorname}'] += [None]
                    dataset_info[f'y_min_{colorname}'] += [None]
                    dataset_info[f'x_max_{colorname}'] += [None]
                    dataset_info[f'y_max_{colorname}'] += [None]

                    dataset_info[f'label_{colorname}'] += [None]

                    continue

            random_x_circle = np.random.randint(int(x_offset) + legend_size // 10,
                                                grid_size + int(x_offset) + legend_size // 10)
            random_y_circle = np.random.randint(int(y_offset) + legend_size // 50,
                                                grid_size + int(y_offset) + legend_size // 50)
            random_circle_r = np.random.randint(grid_size // 40, grid_size // 30)

            draw_circles(background,
                         x=random_x_circle,
                         y=random_y_circle,
                         r=random_circle_r,
                         color=color)

            bbox = (random_x_circle - random_circle_r,
                    random_y_circle - random_circle_r,
                    random_x_circle + random_circle_r,
                    random_y_circle + random_circle_r)

            all_bboxes.append(bbox)

            dataset_info[f'x_min_{colorname}'] += [bbox[0]]
            dataset_info[f'y_min_{colorname}'] += [bbox[1]]
            dataset_info[f'x_max_{colorname}'] += [bbox[2]]
            dataset_info[f'y_max_{colorname}'] += [bbox[3]]

            dataset_info[f'label_{colorname}'] += [idx + 1]

        dataset_info['img'] += [f'images_{split}/{i}.png']

        background.save(os.path.join(base_ds, f'images_{split}/{i}.png'))

        for bbox in all_bboxes:
            draw_circles(background, bbox[2], bbox[3])
            draw_circles(background, bbox[0], bbox[1])
        background.save(os.path.join(base_ds, f'points_{split}/{i}.png'))
        # background.save(f'./gen_dataset/points_{split}/{i}.png')

    dataset_df = pd.DataFrame(data=dataset_info)
    dataset_df.to_csv(os.path.join(base_ds, f'{split}_dataset_info.csv'), header=True, index=False)
