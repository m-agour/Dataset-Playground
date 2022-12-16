"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import time

import cv2
import cv2 as cv
import numpy as np
from shapely.geometry import Polygon, LineString, MultiPolygon

from helpers import draw_contours, get_contours, imshow, approx_lines, draw_polygons, approx_contours, \
    contours_to_lines, rects_to_polygons
from rooms import rooms_polygons


def sketch(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel


def get_wall_contours(src, rooms_mask, wall_width=10):
    src = cv2.GaussianBlur(src, (3, 3), 0)

    # r, g, b = cv2.split(src)
    # mask = cv2.inRange(b, 0, 70) & cv2.inRange(g, 0, 52) & cv2.inRange(r, 0, 119)
    # mask = cv2.dilate(mask, (wall_width, wall_width))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (wall_width* 4, wall_width * 4))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # mask = cv2.dilate(mask, (wall_width, wall_width))
    # imshow(mask)

    src = cv2.GaussianBlur(src, (5, 5), 0)

    src = cv2.bilateralFilter(src, 5, 5, 20, None)

    lab = cv2.cvtColor(src, cv.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    wimg = cv2.inRange(l, 50, 120) & cv2.inRange(a, 90, 140) & cv2.inRange(b, 122, 135) & ~ rooms_mask

    wimg = cv2.GaussianBlur(wimg, (3, 3), 0)
    thinned = cv2.ximgproc.thinning(wimg)

    contours = get_contours(thinned, 0, True)
    thinned *= 0
    approxed_contours = approx_contours(contours, eps=wall_width // 2)
    approxed_contours = approx_contours(approxed_contours, eps=wall_width // 2)
    approxed_contours = approx_contours(approxed_contours, eps=wall_width // 2)
    approxed_contours = approx_contours(approxed_contours, eps=wall_width // 2)

    c = (draw_contours(thinned, approxed_contours, 0, 255, simple=True) > 0).astype(np.uint8)*255

    approxed_contours = get_contours(c, 0, True)
    # linesP = cv.HoughLinesP(thinned, 1, np.pi / 180, 7, None, 0, 9)

    # if linesP is not None:
    #     linesP = approx_lines(linesP, eps=2)
    #     linesP = approx_lines(linesP, eps=4)
    #     linesP = approx_lines(linesP, eps=5)
    #     return linesP

    return approxed_contours


def get_wall_lines(src, rooms_mask, wall_width=10):
    contours = get_wall_contours(src, rooms_mask, wall_width)
    return contours_to_lines(contours)


def remove_lines_in_rooms(lines_shapes, rooms_geometry):
    lines = []

    for l in lines_shapes:
        f = False
        for g in rooms_geometry.values():
            for room_geom in g:
                if room_geom.contains(l):
                    f = True
                    print('skipping')
                    break
        if not f:
            lines.append(l)
    return lines


def get_wall_lines_polys(lines, width):
    xs = []
    ys = []
    points = {}
    rects = []
    for i in range(0, len(lines)):
        l = lines[i].coords
        # cv.line(src, np.int32(l[0]), np.int32(l[1]), (0, 0, 255), 9, cv.LINE_AA)
        x1, y1 = np.int32(l[0])
        x2, y2 = np.int32(l[1])

        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        w += width
        x -= width // 2
        h += width
        y -= width // 2

        xe = x + w
        ye = y + h
        xs.extend([x, xe])
        ys.extend([y, ye])

        horiz = w > h

        points[(x, y)] = {'up': True, 'left': True, 'horiz': horiz}
        points[(xe, y)] = {'up': True, 'left': False, 'horiz': horiz}
        points[(x, ye)] = {'up': False, 'left': True, 'horiz': horiz}
        points[(xe, ye)] = {'up': False, 'left': False, 'horiz': horiz}
        rects.append((x, y, w, h))
        # cv2.rectangle(img, (x, y, w, h), (255, 255, 255), -1)
    return MultiPolygon(rects_to_polygons(rects)), sorted(set(xs)), sorted(set(ys)), points
