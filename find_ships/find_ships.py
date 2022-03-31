import json, glob, argparse, os, sys
from datetime import datetime

from statistics import mean, stdev
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from keras.utils import np_utils

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

def read_args():
    parser=argparse.ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg)
    pwargs=vars(parser.parse_args())
    print(pwargs)
    return pwargs


def tf_load_img(path):
    print(datetime.now(), 'Loading: ', path, flush = True)
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    return img.numpy()

def cutting(picture_tensor, x, y):
    area_study = picture_tensor[y:y+80, x:x+80, 0:3]
    area_study = area_study / 255
    sys.stdout.write('\rX:{0} Y:{1}  '.format(x, y))
    return area_study

def not_near(x, y, s, coordinates):
    result = True
    for e in coordinates:
        if x+s > e[0][0] and x-s < e[0][0] and y+s > e[0][1] and y-s < e[0][1]:
            result = False
    return result

def show_ship(picture_tensor, x, y, acc, thickness=5):
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[y+i][x-th][ch] = -1
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[y+i][x+th+80][ch] = -1
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[y-th][x+i][ch] = -1

    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[y+th+80][x+i][ch] = -1
    return picture_tensor

def find_ships(img):
    step = 10; coordinates = []
    height = img.shape[0]
    width = img.shape[1]
    imgs = []; xs_list = []; ys_list = []
    batch_size = 1000
    k = 0
    for y in range(int((height-(80-step))/step)):
        for x in range(int((width-(80-step))/step) ):
            k += 1
            ys = y*step
            xs = x*step
            area = img[ys:ys+80, xs:xs+80, 0:3] / 250
            imgs.append(area)
            xs_list.append(xs)
            ys_list.append(ys)
            if k == batch_size:
                imgs = np.asarray(imgs)
                results = model.predict(imgs)
                imgs = []
                k = 0
                for ri,r in enumerate(results):
                    if r[1] > 0.95: #and not_near(x*step,y*step, 88, coordinates):
                        coordinates.append([xs_list[ri], ys_list[ri], r[1]])
                xs_list = []; ys_list = []

    return coordinates


def remove_duplicates(c, min_dist = 80):
    min_dist = min_dist*min_dist
    new_c = [c[0]]
    for ci in c:
        dist = []
        for cj in new_c:
            dist.append((ci[0]-cj[0])**2+(ci[1]-cj[1])**2)

        if min(dist) > min_dist:
            new_c.append([ci[0], ci[1], ci[2]])
    return new_c


def process_image(img):
    # Find ships:
    coordinates = find_ships(img)

    coordinates = remove_duplicates(coordinates)
    scores = [ c[2] for c in coordinates ]

    stats = {
        'out:num_ships': str(len(scores)),
        'out:max_score': str(max(scores)),
        'out:mean_score': str(np.mean(scores)),
        'out:std_score': str(np.std(scores))
    }

    print('Statistics:', stats, flush = True)


    for e in coordinates:
        img = show_ship(img, e[0], e[1], e[2])

    return img, stats


if __name__ == '__main__':
    args = read_args()

    # Load scene images
    img = tf_load_img(args['img_path'])

    # Load model:
    model = keras.models.load_model(args['model_dir'])

    # Create output directory:
    os.makedirs(os.path.dirname(args['img_path_out']), exist_ok = True)

    img, stats = process_image(img)

    # Save image:
    img = Image.fromarray(img)
    img.save(args['img_path_out'])

    with open(args['img_path_out'].replace('.png', '.json'), 'w') as fp:
        json.dump(stats, fp, indent = 4)
