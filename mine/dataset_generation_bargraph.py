import os
import random

import cv2
import numpy as np
from PIL import Image, ImageFont
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
# dataset_sizes = [50, 1]


base_ds = './bar_dataset'
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

selectable_colors = [NDP_color,
          GPC_color,
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

    random_color_order = np.random.permutation(selectable_colors)

    number_of_bars = np.random.randint(1, len(selectable_colors))
    bboxes = []

    for clr in random_color_order[:number_of_bars]:

        offset = np.random.uniform(0, gap / 5)
        absolute_leftUpPoint = (x, y - offset)
        absolute_rightDownPoint = (x + l_x, y - offset + l_y)
        absolute_twoPointList = [absolute_leftUpPoint, absolute_rightDownPoint]

        bottom_right_x = np.random.uniform(x, x + l_x)
        rightDownPoint = (bottom_right_x, y - offset + l_y)
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

        percent_x_left = x + l_x + r
        percent_leftUpPoint = (percent_x_left, circle_center_y - r)
        percent_rightDownPoint = (percent_x_left + 2 * r, circle_center_y + r)
        percent_twoPointList = [percent_leftUpPoint, percent_rightDownPoint]
        draw.rectangle(percent_twoPointList, fill=(221, 221, 221, 225))

        border_width = r / 8
        percent_leftUpPoint = (percent_x_left + border_width, circle_center_y - r + border_width)
        percent_rightDownPoint = (percent_x_left + 2 * r - border_width, circle_center_y + r - border_width)
        percent_twoPointList = [percent_leftUpPoint, percent_rightDownPoint]
        draw.rectangle(percent_twoPointList, fill=(255, 255, 255, 225))

        percent_text_leftUpPoint = (percent_x_left + r/3, circle_center_y - r + r/2)
        font = ImageFont.truetype("../calibril.ttf", int(r))
        draw.text(percent_text_leftUpPoint, f'{np.random.randint(0, 100)}%', fill=(0, 0, 0, 0), font=font)

        bbox = (leftUpPoint[0],
                leftUpPoint[1],
                percent_rightDownPoint[0],
                percent_rightDownPoint[1])

        bboxes.append(bbox)

        y += gap

    return bboxes

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

        os.mkdir(os.path.join(base_ds, f'bar_images_{split}'))
        os.mkdir(os.path.join(base_ds, f'bar_points_{split}'))

        dataset_info = {'img': [],
                        'x_min': [],
                        'y_min': [],
                        'x_max': [],
                        'y_max': [],
                        'label': []}

        for i in range(split_dataset_size):
            if i % 100 == 0:
                print(f'{i} / {split_dataset_size}')
            legend_size = np.random.randint(200, 340)
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
                twice_big = True
            else:
                twice_big = False


            max_offset = background_size - (legend_size * 1.5)
            if twice_big:
                shift_size = 0
                x_shift = legends.size[0]
                background = background_org.resize((background_size * 2, background_size))
            else:
                shift_size = 0
                x_shift = 0
                background = background_org.resize((background_size, background_size))

            x_offset = np.random.randint(0 - shift_size, max_offset - shift_size)
            y_offset = np.random.randint(0, max_offset * 2)
            if twice_big:
                background.paste(legends, (x_offset, y_offset), legends)

            if title:
                bar_x_offset = x_offset + x_shift + diffs[0] * 2
            else:
                bar_x_offset = x_offset + x_shift * 1.2

            bar_y_offset = y_offset + legends.size[0] * 0.25

            if title:
                x_offset_title = int(bar_x_offset - diffs[0])
                y_offset_title = int(bar_y_offset - diffs[1])

                background.paste(title, (x_offset_title, y_offset_title), title)

            bboxs = draw_bar_graph(background, bar_x_offset, bar_y_offset,
                           l_x=legends.size[0],
                           l_y=legends.size[1] / 40,
                           gap=legends.size[1] / 7)
            if twice_big:
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

            for bbox in bboxs:
                dataset_info['img'] += [f'bar_images_{split}/{i}.png']

                dataset_info['x_min'] += [bbox[0]]
                dataset_info['y_min'] += [bbox[1]]
                dataset_info['x_max'] += [bbox[2]]
                dataset_info['y_max'] += [bbox[3]]

                dataset_info['label'] += [20]

            background.save(os.path.join(base_ds, f'bar_images_{split}/{i}.png'))

            for bbox in bboxs:
                draw_circles(background, bbox[2], bbox[3])
                draw_circles(background, bbox[0], bbox[1])

            background.save(os.path.join(base_ds, f'bar_points_{split}/{i}.png'))

        dataset_df = pd.DataFrame(data=dataset_info)
        dataset_df.to_csv(os.path.join(base_ds, f'bar_{split}_dataset_info.csv'), header=True, index=False)


create_dataset()
