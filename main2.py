"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import time
from pyproj import Geod, geod
import cv2
import cv2 as cv
import numpy as np
from scipy.signal import get_window
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import linemerge
import geopandas
from door import get_doors
from helpers import draw_contours, get_contours, imshow, approx_lines, draw_polygons, contours_to_lines, draw_lines, \
    imwrite, rects_to_polygons
from lines import remove_lines_in_rooms, get_wall_lines, get_wall_lines_polys
from rooms import rooms_polygons, get_rooms, window_rects
from wall import get_wall_width

# Loading image
# img = cv2.imread('Mo/0014482_0000000.jpg')
img = cv2.imread('Mo/0034158_0000000.jpeg')
img = cv2.imread('Mo/0014334_0000000.jpg')
# img = cv2.imread('Mo/0001911_0000000.jpg')
img = cv2.imread('Mo/0007635_0000000.jpg')
# img = cv2.imread('Mo/0008004_0000000.jpg')
# img = cv2.imread('Mo/0007635_0000000.jpg')
# img = cv2.imread('Mo/0014334_0000000.jpg')
img = cv2.imread('Mo/0001911_0000000.jpg')

#####################################################################################
# Blurring and splitting


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rimg = img.copy()
blurred_img = cv2.pyrMeanShiftFiltering(img, 20, 20)
# blurred_img = cv2.imread('blurred.png')

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
print(lines)


# # k
# lines = linemerge(lines)
# imshow(draw_lines(lines, img, (255, 0, 0)))

# def merge_lines(lines):
#     tbd = []
#     for i in range(len(lines)-1):
#         curr = lines[i]
#         nex = lines[i+1]
#         aft = lines[i+2]
#         if curr[0] == nex[0] == aft[0]:
#             tbd.insert(0, i+1)
#         if curr[0] == nex[0] == aft[0]:
#             tbd.insert(0, i + 1)
#     for i in tbd:
#         del lines[i]


# p1, p2 = line.coords[0])

def line_length(line):
    return np.linalg.norm(np.array(line.coords[0]) - np.array(line.coords[1]))


lines = [l for l in lines if line_length(l) > 3]
print(lines)

# imshow(draw_lines(lines, img, (255, 0, 0)))

# lines = remove_lines_in_rooms(lines, rooms_polys_smooth )

# imshow(draw_lines(lines, img, (255, 0, 0)))

# lines_new = []
# for pol in lines:
#     boundary = pol.boundary
#     if boundary.type == 'MultiLineString':
#         for line in boundary:
#             lines_new.append(line)
#     else:
#         lines_new.append(boundary)

# for line in lines_new:
#     print(line.wkt)

img *= 0
wall_polys, xs, ys, points = get_wall_lines_polys(lines, wall_width)

# imshow(wall_img)

# img *= 0
rooms_polys_smooth = rooms_polygons(rooms_contours, smooth=True, xsys=(xs, ys), eps=wall_width, multi_poly=True)
img = draw_polygons(rooms_polys_smooth, img, (255, 0, 0))

# wall_mask = np.zeros(img.shape[:2]).astype(np.uint8)
# wall_mask, xs, ys, points = draw_wall_lines(wall_mask, lines, wall_width)
# new_wall_contours = get_contours(wall_mask, 0, True)
# wall_mask = np.zeros(img.shape[:2]).astype(np.uint8)
# wall_mask = draw_contours(wall_mask, new_wall_contours, min_threshold=0, simple=True)


# contour_points = []
# for m in new_wall_contours:
#     contour_points += list(m.squeeze())
# print(contour_points)
# imshow(wall_mask)

####################################################################################
# Getting walls itself
c1, c2, c3 = cv2.split(img)

# ####################################################################################################################;
# Window and door

window_rec = window_rects(r, g, b, room_wall_mask=c1 | c2 | c3, width=wall_width, points=points)
door_rec = get_doors(r, g, b, rimg, width=wall_width, points=points)


# imshow(img)


# [cv2.rectangle(img, r, (255, 255, 0), -1) for r in window_rec]
# [cv2.rectangle(img, r, (0, 255, 255), -1) for r in door_rec]
# [cv2.circle(img, r, 5, (0, 255, 255), -1) for r in points]


window_poly = MultiPolygon(rects_to_polygons(window_rec))
door_poly = MultiPolygon(rects_to_polygons(door_rec))

# imshow(img)


data = [{**rooms_polys_smooth, 'door': door_poly, 'window': window_poly, 'wall_lines': lines, 'wall_poly': wall_polys, 'size': img.shape[:2], 'wall_width': wall_width}]
print(data)

d = geopandas.GeoDataFrame(data)
print(d['door'][0])


def draw_multi_polygons(mpoly, shape):
    img = np.zeros(shape)
    polys = list(mpoly)
    for poly in polys:
        img = cv2.drawContours(img, np.int32([poly.exterior.coords]), -1, 255, -1)
    return img


# [print(poly.centroid) for poly in d['door'][0]]
imshow(draw_multi_polygons(d['door'][0], img.shape[:2]))
