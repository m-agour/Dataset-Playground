import numpy as np
import cv2
import random

from helpers import get_contours, draw_contours


def window_rects(r, g, b, img, width=13, real_width=0, points={}):
    rimg = cv2.inRange(r, 236, 255) & cv2.inRange(g, 200, 217) & cv2.inRange(b, 159, 255)

    contours, _ = cv2.findContours(rimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = [[c, cv2.contourArea(c), cv2.boundingRect(c)] for c in contours]

    threshold = width * width * 0.5

    contours = [c for c in contours if c[1] > threshold]
    contours = sorted(contours, key=lambda c: c[1], reverse=True)

    xs = []
    ys = []
    hrects = []
    vrects = []

    temp_points_h = [p for p in points if points[p]['horiz'] == 1]
    temp_points_v = [p for p in points if points[p]['horiz'] == 0]

    rimg *= 0
    for c in contours:
        x, y, w, h = np.array(c[2]) - width // 4

        center = (x + w // 2, y+h//2)

        horiz = w > h
        ps = temp_points_h if not horiz else temp_points_v
        if points:
            v1, v2, v3, v4 = [np.array(p) for p in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]]
            c_point = None
            min_dist = 10 ** 9
            for p in ps:
                if horiz:
                    if points[p]['up']:
                        verts = [v1, v2]
                    else:
                        verts = [v3, v4]
                if not horiz:
                    if points[p]['left']:
                        verts = [v1, v3]
                    else:
                        verts = [v2, v4]
                for v in verts:
                    d = np.linalg.norm(v - np.array(p))
                    if d < min_dist:
                        min_dist = d
                        c_point = p

            axis_only = False
            if horiz:
                w = w + 3 * width // 4
                x += width // 4
                y += width // 4

                x -= width // 2
                w += width

            else:
                h = h + 3 * width // 4
                # y += width // 4
                # x += width // 4

                y -= width // 2
                h += width

            offset = width // 4
            th = width * 3
            if w > h:
                if min_dist > th:
                    c_point = sorted(points, key=lambda p: min([abs(p[1] - y), abs(p[1] - (y + h))]))[0]
                    axis_only = 1

                if min([abs(c_point[1] - y), abs(c_point[1] - (y + h))]) < th and (
                        axis_only or (min_dist <= th)):
                    h = width + offset // 2
                    up = points[c_point]['up']

                    if (c_point[0], c_point[1] + real_width) in points:
                        y = c_point[1]
                    elif (c_point[0], c_point[1] - real_width) in points:
                        y = c_point[1] - h
                    elif (c_point[0] - real_width, c_point[1]) in points or (
                            c_point[0] + real_width, c_point[1]) in points:
                        if up:
                            y = c_point[1]
                        else:
                            y = c_point[1] - h
                    else:
                        if center[1] < c_point[1]:
                            y = c_point[1]
                        else:
                            y = c_point[1] - h
                    h = width + offset // 2
                    hrects.append([x - offset, y - offset, w, h])
                    continue
            else:
                if min_dist > th:
                    c_point = sorted(points, key=lambda p: min([abs(p[0] - x), abs(p[0] - (x + w))]))[0]
                    axis_only = 1

                if min([abs(c_point[0] - x), abs(c_point[0] - (x + w))]) < th and (
                        axis_only or (min_dist <= th)):
                    w = width + offset // 2
                    left = points[c_point]['left']

                    if (c_point[0] + real_width, c_point[1]) in points:
                        x = c_point[0]
                    elif (c_point[0] - real_width, c_point[1]) in points:
                        x = c_point[0] - w

                    elif (c_point[0], c_point[1] - real_width) in points or (
                            c_point[0], c_point[1] + real_width) in points:
                        if left:
                            x = c_point[0]
                        else:
                            x = c_point[0] - w
                    else:
                        if center[0] < c_point[0]:
                            x = c_point[0]
                        else:
                            x = c_point[0] - w
                    vrects.append([x - offset, y - offset, w, h])

                    continue

        continue
        offset = width // 2
        x, y, w, h = np.array(c[2])
        if h > w:
            ws = w // 2
            a = img[y: y + h + 1, x - ws: x + 1]
            b = img[y: y + h + 1, x + w: x + ws + w + 1]

            ac = np.linalg.norm(a.mean(axis=(0, 1)) - np.array([215, 205, 190]))
            bc = np.linalg.norm(b.mean(axis=(0, 1)) - np.array([215, 205, 190]))

            if ac < bc:
                xf = x - ws
                wf = width
            else:
                xf = x - w
                wf = width

            widths.append(wf)
            vrects.append([xf - offset, y - 2, wf, h + 4])
        else:
            hs = h // 3
            a = img[y - hs: y + 1, x: x + w + 1]
            b = img[y + h: y + hs + h + 1, x: x + w + 1]
            ac = np.linalg.norm(a.mean(axis=(0, 1)) - np.array([215, 205, 190]))
            bc = np.linalg.norm(b.mean(axis=(0, 1)) - np.array([215, 205, 190]))
            if ac < bc:
                yf = y - hs
                hf = width
            else:
                yf = y - h
                hf = width
            widths.append(hf)
            hrects.append([x - 2, yf - offset, w + 4, hf])

    for r in hrects:
        cv2.rectangle(rimg, r, (255, 0, 0), -1)

    for r in vrects:
        cv2.rectangle(rimg, r, (255, 0, 0), -1)
    return rimg, vrects + hrects


def get_room(r, g, b, threshold, rng, kernel, default_morphing=True):
    wimg = cv2.inRange(b, *rng[0]) & cv2.inRange(g, *rng[1]) & cv2.inRange(r, *rng[2])
    # imshow(wimg)
    contours = get_contours(wimg, threshold)
    wimg *= 0
    if default_morphing:
        wimg = draw_contours(wimg, contours, color=255, kernel=kernel)
    else:
        wimg = draw_contours(wimg, contours, color=255, kernel=None)
        wimg = cv2.morphologyEx(wimg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
    return wimg


rooms = {

    'pool': {
        'range': ((159, 202), (150, 187), (13, 70)),
        'color': (255, 255, 0),
        'threshold': 3000,
        'kernel': (30, 30),
        'default_morphing': True
    },
    'garden': {
        'range': ((73, 91), (147, 155), (130, 142)),
        'color': (255, 255, 0),
        'threshold': 3000,
        'kernel': (30, 30),
        'default_morphing': True
    },
    'parking': {
        'range': ((188, 196), (194, 199), (189, 200)),
        'color': (255, 100, 50),
        'threshold': 3000,
        'kernel': (30, 30),
        'default_morphing': True
    },
    'stair': {
        'range': ((232, 255), (205, 255), (228, 252)),
        'color': (100, 100, 100),
        'threshold': 500,
        'kernel': (50, 50),
        'default_morphing': False
    },
    'general': {
        'range': ((205, 212), (216, 223), (219, 255)),
        'color': (0, 255, 0),
        'threshold': 3000,
        'kernel': (20, 20),
        'default_morphing': True
    },
    'bedroom': {
        'range': ((138, 153), (167, 179), (202, 219)),
        'color': (255, 0, 0),
        'threshold': 1000,
        'kernel': (10, 10),
        'default_morphing': True
    },
    'bathroom': {
        'range': ((112, 125), (142, 164), (152, 166)),
        'color': (0, 255, 255),
        'threshold': 1000,
        'kernel': (10, 10),
        'default_morphing': True
    },
    'balacony': {
        'range': ((166, 196), (198, 221), (212, 253)),
        'color': (0, 0, 255),
        'threshold': 2000,
        'kernel': (20, 20),
        'default_morphing': True
    },
    'veranda': {
        'range': ((211, 223), (216, 232), (231, 242)),
        'color': (0, 70, 70),
        'threshold': 2000,
        'kernel': (20, 20),
        'default_morphing': True
    },

}
