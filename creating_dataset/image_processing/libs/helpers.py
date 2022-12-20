import random
import time
from datetime import datetime

import numpy as np
import cv2
from shapely.geometry import Polygon, LineString, LinearRing, MultiLineString, MultiPolygon


def rule_rooms(img, points, wall_width):
    anchors = [[i[0] - wall_width // 4 for i in points], [i[1] - wall_width // 4 for i in points]]
    anchors[0].extend([0, img.shape[0]])
    anchors[1].extend([0, img.shape[1]])

    anchors[0] = np.unique(cluster_points(anchors[0], 4))
    anchors[1] = np.unique(cluster_points(anchors[1], 4))
    for ix in range(len(anchors[0]) - 1):
        for iy in range(len(anchors[1]) - 1):
            try:
                x1, x2 = anchors[0][ix:ix + 2]
                y1, y2 = anchors[1][iy:iy + 2]
                a = img[y1:y2 + 1, x1: x2 + 1, :]
                colors, count = np.unique(a.reshape(-1, a.shape[-1]), axis=0, return_counts=True)
                dom_color = colors[count.argmax()]
                img[y1:y2 + 1, x1: x2 + 1, :] = dom_color
            except:
                ...
                # print(x1, x2, y1, y2, len(img), len(img[0]))
    return img


def draw_contours(img, contours, min_threshold=100, color=None, kernel=None, simple=False):
    if simple:
        out = np.zeros(img.shape[:2])
        cv2.drawContours(out, contours, -1, 255, -1)
        return out

    out = np.zeros_like(img)

    t = time.perf_counter()

    for cc in contours:
        timg = np.zeros_like(img)
        c = cc[0]
        area = cc[1]
        if area > min_threshold:
            cv2.drawContours(timg, [c], -1, 255, -1)
            if kernel:
                timg = cv2.morphologyEx(timg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
        out = out | timg
    # print(time.perf_counter() - t)

    return out.astype(np.uint8) * 255


def get_center(cntor):
    #     M = cv2.moments(cntor)
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])

    #     return (cX, cY)
    sh = cntor.shape
    s = np.mean(cntor.reshape(sh[0], 2), axis=0)
    return s.astype(int)


def get_contours(img, min_threshold=100, simple=False):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if simple:
        return [c for c in contours if cv2.contourArea(c) >= min_threshold]

    contours = [[c, cv2.contourArea(c), cv2.boundingRect(c)] for c in contours]
    contours = [c for c in contours if c[1] >= min_threshold]
    contours = sorted(contours, key=lambda c: c[1], reverse=True)
    return contours


def cluster_points(in_points, eps=8, as_dict=False):
    anc = []
    clusters = []
    if as_dict:
        out_dict = {}
    points_sorted = sorted(in_points)

    curr_point = points_sorted[0]

    curr_cluster = [curr_point]

    for point in points_sorted[1:]:
        if point <= curr_point + eps:
            curr_cluster.append(point)
        else:
            clusters.append(curr_cluster)
            curr_cluster = [point]
            curr_point = point
    clusters.append(curr_cluster)

    for c in clusters:
        mn = int(np.median(c))
        anc.append(mn)
        if as_dict:
            out_dict = {**out_dict, **{i: mn for i in c}}
    if as_dict:
        return out_dict
    return anc


def approx_lines(lines, eps=8):
    xs = []
    ys = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    xs = sorted(xs)
    ys = sorted(ys)
    xs = cluster_points(xs, as_dict=True, eps=eps)
    ys = cluster_points(ys, as_dict=True, eps=eps)

    for li in range(len(lines)):
        lines[li][0][0] = xs[lines[li][0][0]]
        lines[li][0][1] = ys[lines[li][0][1]]
        lines[li][0][2] = xs[lines[li][0][2]]
        lines[li][0][3] = ys[lines[li][0][3]]
    return lines


def imwrite(name, img):
    cv2.imwrite(name, img)
    cv2.imwrite('./Mo/temp/' + str(datetime.now()) + '.jpg', img)


def imshow(img, key=True):
    cv2.imshow('', img)
    if key:
        cv2.waitKey(0)


def approx_contours(contour, eps=6, xsys=None):
    if not xsys:
        xs = []
        ys = []
        for c in contour:
            for s in c:
                for f in s:
                    x, y = f
                    xs.append(x)
                    ys.append(y)

        xs = sorted(xs)
        ys = sorted(ys)
    else:
        xs, ys = xsys
    xs = cluster_points(xs, as_dict=True, eps=eps)
    ys = cluster_points(ys, as_dict=True, eps=eps)

    for ci in range(len(contour)):
        for li in range(len(contour[ci])):
            for pi in range(len(contour[ci][li])):
                if xsys:
                    x = contour[ci][li][pi][0]
                    xn = min(xs, key=lambda xo: abs(xo - x))
                    if abs(xs[xn] - x) <= eps:
                        contour[ci][li][pi][0] = np.int32(xn)

                    y = contour[ci][li][pi][1]
                    yn = min(ys, key=lambda yo: abs(yo - y))
                    if abs(yn - y) <= eps:
                        contour[ci][li][pi][1] = np.int32(yn)
                else:
                    contour[ci][li][pi][0] = np.int32(xs[contour[ci][li][pi][0]])
                    contour[ci][li][pi][1] = np.int32(ys[contour[ci][li][pi][1]])
    return contour


def contours_to_lines(contours):
    lines = []
    for c in contours:
        l = len(c)
        i = 0
        while i < l - 1:
            p1 = c[i][0]
            p2 = c[i + 1][0]
            if (p1 == p2).all():
                i += 1
                continue
            lines.append(LineString((p1, p2)))
            i += 1
    return lines
    # polygons = []
    # for cont in contours:
    #     if len(cont) > 2:
    #         cont = np.squeeze(cont)
    #         polygons.append((cont))
    # return polygons


def draw_polygons(polygon, img, color):
    if isinstance(polygon, list):
        for p in polygon:
            img = cv2.drawContours(img, np.int32([p.exterior.coords]), -1, color, -1)
        return img
    elif isinstance(polygon, dict):
        for name in polygon:
            if isinstance(polygon[name], MultiPolygon):
                polys = polygon[name].geoms
            else:
                polys = polygon[name]
            for p in polys:
                img = cv2.drawContours(img, np.int32([p.exterior.coords]), -1, color, -1)
        return img
    return cv2.drawContours(img, np.int32([polygon.exterior.coords]), -1, color, -1)


def draw_lines(lines, img, color):
    if isinstance(lines, (MultiLineString, list)):
        for p in lines:
            color = (random.randrange(0, 255), random.randrange(0, 256), random.randrange(0, 255))
            img = cv2.line(img, np.int32(p.coords[0]), np.int32(p.coords[1]), color, 1)
        return img
    elif isinstance(lines, dict):
        for name in lines:
            for p in lines[name]:
                img = cv2.line(img, np.int32(p.coords[0]), np.int32(p.coords[1]), color, 1)
        return img
    return cv2.line(img, np.int32(lines.coords[0]), np.int32(lines.coords[1]), color, 1)


def rects_to_polygons(rects):
    polys = []
    for rect in rects:
        x, y, w, h = rect
        polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        polys.append(Polygon(polygon))
    return polys


def draw_multi_polygons(mpoly, shape):
    img = np.zeros(shape)
    polys = mpoly.geoms
    for poly in polys:
        img = cv2.drawContours(img, np.int32([poly.exterior.coords]), -1, 255, -1)
    return img


def line_length(line):
    return np.linalg.norm(np.array(line.coords[0]) - np.array(line.coords[1]))


def get_inner_polygon(polygons, wall_width, shape):
    masks = []
    for p in polygons:
        if isinstance(p, MultiPolygon):
            masks.append(draw_multi_polygons(p, shape))

    inner = compine_masks(masks)
    inner = cv2.morphologyEx(inner, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (wall_width, wall_width)))

    outer_contour, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # outer = draw_contours(inner, outer_contour, simple=True)
    polys = []
    for c in outer_contour:
        contour = np.squeeze(c)
        polys.append(Polygon(contour))
    return MultiPolygon(polys)


def compine_masks(masks):
    return (np.sum(masks, axis=0) > 0).astype(np.uint8) * 255
# def cluster_points2(in_points, img, x, eps=8, rect=(0, 0, 3000, 3000)):
#     anc = []
#     clusters = []
#
#     points_sorted = sorted(in_points)
#
#     curr_point = points_sorted[0]
#
#     curr_cluster = [curr_point]
#
#     for point in points_sorted[1:]:
#         if point <= curr_point + eps:
#             curr_cluster.append(point)
#         else:
#             clusters.append(curr_cluster)
#             curr_cluster = [point]
#             curr_point = point
#     clusters.append(curr_cluster)
#
#     for c in clusters:
#         m = 10 ** 9
#         temp = []
#         for v in c:
#             if x:
#                 sa = np.sum(img[y:y + h + 1, v - eps: v + eps + 1])
#                 if sa < m:
#                     m = sa
#                 temp.append((v, sa))
#             else:
#                 sa = np.sum(img[v - eps: v + eps + 1, x:x + w + 1])
#                 if sa < m:
#                     m = sa
#                 temp.append((v, sa))
#
#         mn = int(2 * np.median(c) / 2)
#         anc.append(mn)
#         # if x:
#         #     cv2.line(img, [mn, y], [mn,y+h ], (0, 0, 0), 2)
#
#     return anc
