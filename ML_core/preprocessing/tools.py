from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import DateCategories


root = 'C:/Users/tolik/information_technology/third_year/practice_project/CoreAnalysis-ML'
data = pd.read_csv('{}/data_for_study/data.csv'.format(root))
path_to_labels = 'C:/Users/tolik/information_technology/third_year/practice_project/CoreAnalysis-ML/data_for_study/labels'




def convert_into_rgb(labelformat, mask, colormap = DateCategories.labels_colors):
    # labelformat = 'ultra' or 'day'

    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    if labelformat == 'ultra':
        for l in range(3):
            idx = mask == l
            r[idx] = colormap[l, 0]
            g[idx] = colormap[l, 1]
            b[idx] = colormap[l, 2]
    else:
        for l in range(14):
            idx = mask == l
            r[idx] = colormap[l, 0]
            g[idx] = colormap[l, 1]
            b[idx] = colormap[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def change_segments(photo_id, directory_name, colors):
    temp_df = data[data.photo_id == photo_id]
    num_of_sg = len(temp_df)
    task_id = temp_df.task_id.unique()[0]

    # load original masks
    segmented = np.load('{}/data_for_study/matrixes/matrix_{}__{}.npz'.format(root, str(photo_id), task_id))
    segmented = segmented['data']

    # changing masks
    for i in range(num_of_sg):
        sg_type = temp_df[temp_df.segment_num == i]['segment_value'].unique()[0]
        value = colors[sg_type]
        segmented[segmented == i] = value

    # save as png image
    if directory_name == 'ultraviolet':
        mask = segmented - 20
        rgb = convert_into_rgb('ultra', mask)
        im = Image.fromarray(rgb)
        im.save('{}_png/ultraviolet/label_{}.png'.format(path_to_labels, photo_id))
    else:
        mask = segmented - 70
        rgb = convert_into_rgb('day', mask)
        im = Image.fromarray(rgb)
        im.save('{}_png/daylight/label_{}.png'.format(path_to_labels, photo_id))

    # save as npz file
    outfile = '{}/{}/label_{}'.format(path_to_labels,
                                      directory_name,
                                      str(photo_id))
    np.savez(outfile, x=mask)
    return mask


def show_colormap(colormap=DateCategories.labels_colors):
    colormap = colormap / 255.
    fig, axes = plt.subplots(nrows=7, ncols=2)
    axes[0, 0].set(title='Переслаивание пород',
                   fc=colormap[0])
    axes[0, 1].set(title='Алевролит глинистый',
                   fc=colormap[1])
    axes[1, 0].set(title='Песчаник',
                   fc=colormap[2])
    axes[1, 1].set(title='Аргиллит',
                   fc=colormap[3])
    axes[2, 0].set(title='Глинисто-кремнистая',
                   fc=colormap[4])
    axes[2, 1].set(title='Песчаник глинистый',
                   fc=colormap[5])
    axes[3, 0].set(title='Уголь',
                   fc=colormap[6])
    axes[3, 1].set(title='Аргиллит углистый',
                   fc=colormap[7])
    axes[4, 0].set(title='Алевролит',
                   fc=colormap[8])
    axes[4, 1].set(title='Карбонатная порода',
                   fc=colormap[9])
    axes[5, 0].set(title='Известняк',
                   fc=colormap[10])
    axes[5, 1].set(title='Глина аргиллитоподобная',
                   fc=colormap[11])
    axes[6, 0].set(title='Разлом',
                   fc=colormap[12])
    axes[6, 1].set(title='Проба',
                   fc=colormap[13])
    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])

    plt.subplots_adjust(wspace=2, hspace=2)
    plt.show()
    # plt.savefig('colormap for distribution of types.png')


def rgb_hist(image_path):
    image = cv2.imread(image_path)
    blue_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    green_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    red_hist = cv2.calcHist([image], [2], None, [256], [0, 256])

    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(blue_hist, 'b-')
    ax[1].plot(green_hist, 'g-')
    ax[2].plot(red_hist, 'r-')
    plt.show()
    plt.savefig('{}/rgb_hist_{}.png'.format())