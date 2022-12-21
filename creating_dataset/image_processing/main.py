# Planify
import json
import time
import pickle
from io import BytesIO
from os.path import isfile, join

import pandas as pd
from multiprocessing import Pool

import requests
import tqdm
import PIL.Image
from PIL import Image
from shapely import Polygon
from shapely.geometry import MultiPolygon
from libs.door import get_doors
from libs.helpers import draw_polygons, rects_to_polygons, line_length, \
    get_inner_polygon, draw_multi_polygons, imshow, imwrite, approx_contours
from libs.lines import get_wall_lines, get_wall_lines_polys
from libs.rooms import rooms_polygons, get_rooms, window_rects, rooms
from libs.wall import get_wall_width
import os
import cv2
import urllib
import numpy as np

pd.set_option('display.max_columns', None)

from_urls = True

save_dir = "planimgs2d"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

IMG_PATH = 'planimgs'

BLURRED_IMG_PATH = None


def url_to_numpy(url, name):
    img = Image.open(BytesIO(requests.get(url).content))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, img)
    return img


def vectorize_plan(img_name):
    global IMG_PATH, BLURRED_IMG_PATH, from_urls
    #####################################################################################
    # Blurring and splitting

    if from_urls:
        pid = img_name.split('/')[4]
        img = url_to_numpy(img_name, pid + '.png')
    else:
        pid = img_name.split('.')[0]
        img = cv2.imread(os.path.join(IMG_PATH, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rimg = img.copy()

    if BLURRED_IMG_PATH is None:
        blurred_img = cv2.pyrMeanShiftFiltering(img, 20, 20)
    else:
        blurred_img = cv2.imread(os.path.join(BLURRED_IMG_PATH, img_name))

    r, g, b = cv2.split(img)
    rb, gb, bb = cv2.split(blurred_img)

    ####################################################################################
    # Getting walls width
    wall_width = get_wall_width(r, g, b, 0)
    wall_width = wall_width * 2 // 3

    #####################################################################################
    # Getting rooms
    rooms_contours = get_rooms(rb, gb, bb, img, include_balacony=True, as_contours=True, wall_width=wall_width)

    # rooms_polys = rooms_polygons(img, rooms_contours)
    rooms_polys_smooth = rooms_polygons(rooms_contours, smooth=True)

    rooms_mask = draw_polygons(rooms_polys_smooth, r.copy() * 0, (255, 0, 0))

    ####################################################################################
    # Getting walls itself
    lines = get_wall_lines(img, rooms_mask)

    lines = [l for l in lines if line_length(l) > 3]

    wall_polys, xs, ys, points = get_wall_lines_polys(lines, wall_width)

    rooms_polys_smooth = rooms_polygons(rooms_contours, smooth=True, xsys=(xs, ys), eps=wall_width, multi_poly=True)
    img = draw_polygons(rooms_polys_smooth, img, (255, 0, 0))

    ####################################################################################
    # inner mask
    inner_mask_real = cv2.inRange(~img, (0, 0, 0), (10, 10, 10))
    inner_mask_real = cv2.morphologyEx(~inner_mask_real, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_RECT, (wall_width, wall_width)))
    inner_mask_real = cv2.morphologyEx(inner_mask_real, cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_RECT, (wall_width, wall_width)))

    # imshow(inner_mask_real)

    inner_mask_con, _ = cv2.findContours(inner_mask_real, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # outer = draw_contours(inner, outer_contour, simple=True)
    polys = []

    inner_mask_con = approx_contours(inner_mask_con, xsys=(xs, ys), eps=wall_width)
    inner_mask_con = approx_contours(inner_mask_con, xsys=(xs, ys), eps=wall_width)
    inner_mask_con = approx_contours(inner_mask_con, xsys=(xs, ys), eps=wall_width)
    inner_mask_con = approx_contours(inner_mask_con, xsys=(xs, ys), eps=wall_width)

    contour = np.squeeze(inner_mask_con)

    polys.append(Polygon(contour))

    mp = MultiPolygon(polys)

    ####################################################################################
    # Getting walls itself
    c1, c2, c3 = cv2.split(img)

    # ###################################################################################################################;
    # Window and door

    window_rec = window_rects(r, g, b, room_wall_mask=c1 | c2 | c3, width=wall_width, points=points)
    door_rec = get_doors(r, g, b, rimg, width=wall_width, points=points)

    window_poly = MultiPolygon(rects_to_polygons(window_rec))
    door_poly = MultiPolygon(rects_to_polygons(door_rec))

    inner_poly = get_inner_polygon(list(rooms_polys_smooth.values()) + [wall_polys, door_poly, window_poly], wall_width,
                                   img.shape[:2])


    data = {
        **{r: None for r in rooms},
        **rooms_polys_smooth,
        'pid': pid,
        'door': door_poly,
        'window': window_poly,
        'wall': wall_polys,
        'inner': inner_poly,
        'inner_actual': mp,
        'wall_lines': lines,
        'size': img.shape[:2],
        'wall_width': wall_width,
        'img_name': img_name,
        'points': points,
        'xsys': (xs, ys),

    }

    # imshow(draw_multi_polygons(inner_poly, img.shape[:2]))
    return data


# print(len(vectorize_plan('download.jpg')['bathroom'].geoms))

# for img in os.listdir('planimgs'):
#     try:
#         print(img)
#         poly = vectorize_plan(img)['stair']
#         i = cv2.imread(IMG_PATH + '/' + img)
#         i = draw_multi_polygons(poly, i.shape[:2])
#         imwrite(IMG_PATH + '/' + img.replace('.', 'f.'), i)
#     except:
#         ...
#


def safe_vec(img):
    # return vectorize_plan(img)
    try:
        return vectorize_plan(img)
    except:
        return None


# df = pd.read_pickle(f'dataset_1.pkl')


js_areas = json.load(open('images_areas.json'))


def get_area():
    for p in js_areas:
        a = p[1][-1]
        b = p[2]

        if len(a) != len(b):
            print(p)


# print(safe_vec('0008004_0000000.jpg'))
#
if __name__ == '__main__':
    images = [f for f in os.listdir(IMG_PATH) if isfile(join(IMG_PATH, f))]

    pickle.dump(0, open("index.txt", 'wb'))

    if from_urls:
        with open('./floor_plans_urls.pickle', 'rb') as handle:
            images = pickle.load(handle)

    parts = 12
    siz = len(images) // parts
    current_part = pickle.load(open("index.txt", 'rb'))

    for pi in range(current_part, parts):
        with Pool(1) as p:
            start = pi * siz
            end = min(len(images), (pi + 1) * siz)

            print(start, end)
            time.sleep(2)
            data = [i for i in list(tqdm.tqdm(p.imap(safe_vec, images[start: end]), total=len(images[start: end]))) if
                    i is not None]
            p.close()

            dataframe = pd.DataFrame(data)

            dataframe.to_pickle(f'dataset__{pi}.pkl', protocol=4)
            df = pd.read_pickle(f'dataset__{pi}.pkl')

            pickle.dump(pi + 1, open("index.txt", 'wb'))

# p.to_pickle('dataset.pkl')
# df = pd.read_pickle('dataset__3.pkl')
#
# imshow(draw_multi_polygons(df['window'][0], (2000, 2000)))
# # print(p['window'])
# print(p['servant'])
