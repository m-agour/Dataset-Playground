import numpy as np
import cv2
import rasterio as rasterio

from door import get_doors
from helpers import cluster_points, rule_rooms, imwrite, imshow, draw_contours, approx_contours
from rooms import window_rects, get_room, rooms, get_rooms
from wall import get_wall
import time
import datetime

import numpy as np
from shapely.geometry import Polygon
t = time.perf_counter()
img = cv2.imread('Mo/0014482_0000000.jpg')

# img = cv2.imread('Mo/0014482_0000000.jpg')
# img = cv2.imread('Mo/0034158_0000000.jpeg')


img = cv2.imread('Mo/0014334_0000000.jpg')
img = cv2.imread('Mo/0001911_0000000.jpg')
img = cv2.imread('Mo/0007635_0000000.jpg')
# img = cv2.imread('Mo/0008004_0000000.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
bimg = cv2.pyrMeanShiftFiltering(img, 20, 20)
# bimg = cv2.imread('blurred.png')

# imwrite('blurred.png', bimg)

r, g, b = cv2.split(img)
rb, gb, bb = cv2.split(bimg)

wallmask, width, pts = get_wall(r, g, b, threshold=0)
rms_mask = np.zeros_like(img)
wwd_mask = np.zeros_like(img)

# room = rooms["balacony"]
# rng = room['range']
# color = room['color']
# threshold = room['threshold']
# kernel = tuple(np.array(room['kernel']) * width // 4)
# dm = room['default_morphing']

# rm = get_room(rb, gb, bb, threshold, rng, kernel, dm)

opt_width = 8 * width // 10
winimg, wrects = window_rects(r, g, b, img, width=opt_width, real_width=width, points=pts)
# dmask = get_doors(r, g, b, img, width=opt_width, width=width, points=pts)

wwd_mask[np.where(dmask > 0)] = np.array([100, 100, 20]).astype(np.uint8)
wwd_mask[np.where(winimg > 0)] = np.array([255, 0, 255]).astype(np.uint8)
wwd_mask[np.where(wallmask > 0)] = np.array([255, 255, 255]).astype(np.uint8)
kernel = np.ones((width // 2, width // 2), np.uint8)

[cv2.circle(wwd_mask, p, 1, (255, 55, 12), -1) for p in pts]

# wwd_mask = rule_rooms(wwd_mask, pts, width)
imwrite('wwd.png', wwd_mask)

# wwd_mask = cv2.erode(wwd_mask, kernel, iterations=1)


# imshow(get_rooms(rb, gb, bb, img, include_balacony=True, as_contours=False))
rooms_contours = get_rooms(rb, gb, bb, img, include_balacony=True, as_contours=True)
# print(rooms_contours[1][0])
i = wwd_mask.copy()
i *= 0

polygons = []
for cc in rooms_contours:
    c = approx_contours(cc, eps=3)
    c = approx_contours(c, eps=3)
    c = approx_contours(c, eps=3)
    c = approx_contours(c, eps=3)
    # f = rule_rooms(rms_mask, pts, width)
    for contour in c:
        contour = np.squeeze(contour)
        polygons.append(Polygon(contour))
    m = draw_contours(i, c, simple=True)
    i[np.where(m > 0)] = 255

imshow(i)
indices = np.where(wwd_mask != [0, 0, 0])

rms_mask[indices] = wwd_mask[indices]

imwrite('native.png', rms_mask)

# [cv2.line(f, (x, 0), (x, 3000), (50, 100, 3), 1) for x in anchors[0]]
# [cv2.line(f, (0, y), (3000, y), (50, 100, 3), 1) for y in anchors[1]]


#
# f[np.where(rm > 0)] = np.array(color).astype(np.uint8)



contours, hierarchy = cv2.findContours(wallmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
testcontour = contours[0]

# print(contours)
contour = np.squeeze(contours[1])
print(Polygon(contour).contains( Polygon(contour)))

# print(polygon)

import shapely.geometry as sg
import shapely.ops as so
import matplotlib.pyplot as plt

# r1 = sg.Polygon([(0,0),(0,1),(1,1),(1,0),(0,0)])
# r2 = sg.box(0.5,0.5,1.5,1.5)
# r3 = sg.box(4,4,5,5)

# new_shape = so.unary_union([r1, r2, r3])
fig, axs = plt.subplots()
axs.set_aspect('equal', 'datalim')

xs, ys = polygon.exterior.xy
axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')

# img = rasterio.features.rasterize([polygon], out_shape=(60, 50))
# plt.imshow(img)
plt.show()

print(list(zip(polygon.exterior.xy[0], polygon.exterior.xy[1])))
img = np.zeros_like(img)

points = np.int32([np.asarray((list(zip(polygon.exterior.xy[0], polygon.exterior.xy[1]))))])

print(polygon.exterior.coords)

t = time.perf_counter()

[cv2.drawContours(img, np.int32([polygon.exterior.coords]), -1, (0, 255, 255), -1) for i in range(1000)]

print(time.perf_counter() - t)

# image = cv2.polylines(img, points,
#                       1, (255, 0, 0),
#                       -1)
cv2.imshow('', img)
cv2.waitKey(0)

# rooms
# wimg = cv2.morphologyEx(wimg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
# if name in ["general", "bathroom"]:
#     rm = cv2.dilate(rm, np.ones((width // 2, width // 2), np.uint8), iterations=1)
# elif name not in []:
#     rm = cv2.dilate(rm, np.ones((width // 3, width // 3), np.uint8), iterations=2)
