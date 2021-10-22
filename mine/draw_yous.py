import os

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import utils

grid_location = './img_template/grid.png'
you_label_location = '../../new_results/img_results_results/model_annotated_points/'

you_labels = pd.read_csv(os.path.join(you_label_location, 'you_labels.csv'))
loss_threshs = [i for i in range(10)]
loss_threshs.append(0.5)

plt.rcParams['figure.dpi'] = 500

colors = {'ndp': '#EA8435',
          'gpc': '#54933E',
          'cpc': '#022043',
          'ppc': '#452C74',
          'lpc': '#CE302D'}


for loss_thresh in loss_threshs:
    all = 0
    valid = 0
    plt.figure()
    plt.grid()

    for abs_positions in [utils.abs_positions_1, utils.abs_positions_2, utils.abs_positions_3]:
        for k, v in abs_positions.items():
            plt.scatter(v[0] - 4, -1 * (v[1] - 4), color=colors[k], s=50)

    for row in you_labels.iterrows():
        all += 1
        # print(row[1].loss)
        if row[1].loss <= loss_thresh:
            valid += 1
            you = row[1].you[1:-1].split()
            plt.scatter(float(you[0]) - 4, -1 * (float(you[1]) - 4), alpha=0.3, s=20, color='#514A5C')

    plt.title(f'Threshold: {loss_thresh}, valid_points: {valid} / {all}')
    plt.savefig(f'plots/threshold_{loss_thresh}.png')
    # plt.show()