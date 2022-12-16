import cv2
import numpy as np

from helpers import get_center, imshow


def get_doors(r, g, b, img, width, min_th=50, points={}):
    width = width

    imgf = cv2.bilateralFilter(img, 10, 30, 30, None)

    imgf_hsv = cv2.cvtColor(imgf, cv2.COLOR_RGB2HSV)
    _, s, _ = cv2.split(imgf_hsv)
    r, g, b = cv2.split(imgf)
    dmask = np.zeros_like(b)

    mask = ((b < 100) & (g < 100) & (s > 90) & (r > 60) & (b < 170)).astype(np.uint8) * 255
    # imshow(mask)
    door_corners = cv2.goodFeaturesToTrack(mask, 100, 0.4, 4)
    door_corners = np.int0(door_corners)
    door_pts = []

    for corner in door_corners:
        x, y = corner.ravel()
        door_pts.append([x, y])

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = [[c, cv2.contourArea(c), cv2.boundingRect(c)] for c in contours]
    contours = [c for c in contours if c[1] >= min_th]
    contours = sorted(contours, key=lambda c: c[1], reverse=True)

    [cv2.rectangle(dmask, r[2], 255, -1) for r in contours]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(dmask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [[c, cv2.contourArea(c), cv2.boundingRect(c)] for c in contours]
    contours = [c for c in contours if c[1] > min_th]
    contours = sorted(contours, key=lambda c: c[1], reverse=True)

    dmask *= 0

    # temp_points_h = [p for p in points if points[p]['horiz'] == 1]
    # temp_points_v = [p for p in points if points[p]['horiz'] == 0]
    vrects = []
    hrects = []
    dmask *= 0
    for c in contours:
        x, y, w, h = np.array(c[2])

        center = get_center(c[0])

        cv2.circle(dmask, center, 17, 255, -1)

        def dist(p1, p2):
            return np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])

        # door_pts_curr = [p for p in door_pts if dist(p, center) < width * 6]
        # center = np.median(door_pts_curr, axis=0)

        horiz = w > h
        # ps = temp_points_h if not horiz else temp_points_v
        ps = points
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

            # H  O  R  I  Z  O  N  T  A  L
            # offset = width // 8
            th = max(w, h) + width
            if w > h:
                if min_dist > 1.5 * th:
                    c_point = sorted(points, key=lambda p: min([abs(p[1] - y), abs(p[1] - (y + h))]))[0]
                    axis_only = 1

                curr_d = min([abs(c_point[1] - y), abs(c_point[1] - (y + h))])
                if curr_d <= 1.5 * th or (axis_only and (curr_d <= 2 * th)):
                    h = width
                    up = points[c_point]['up']

                    if (c_point[0], c_point[1] + width) in points:
                        y = c_point[1]
                    elif (c_point[0], c_point[1] - width) in points:
                        y = c_point[1] - h
                    elif (c_point[0] - width, c_point[1]) in points or (
                            c_point[0] + width, c_point[1]) in points:
                        if up:
                            y = c_point[1]
                        else:
                            y = c_point[1] - h
                    else:
                        if center[1] < c_point[1]:
                            y = c_point[1]
                        else:
                            y = c_point[1] - h
                    h = width
                    hrects.append([x, y, w, h])
                    # cv2.circle(dmask, c_point, 17, 255, -1)
                    # hrects.append([int(center[0]-2), int(center[1]-2), 4, 4])

                    continue
            else:
                # V  E  R  T  I  C  A  L
                if min_dist > 1.5 * th:
                    c_point = sorted(points, key=lambda p: min([abs(p[0] - x), abs(p[0] - (x + w))]))[0]
                    axis_only = 1

                curr_d = min([abs(c_point[0] - x), abs(c_point[0] - (x + w))])
                if curr_d <= 1.5 * th or (axis_only and (curr_d <= 2 * th)):
                    w = width
                    left = points[c_point]['left']

                    if (c_point[0] + width, c_point[1]) in points:
                        x = c_point[0]
                    elif (c_point[0] - width, c_point[1]) in points:
                        x = c_point[0] - w

                    elif (c_point[0], c_point[1] - width) in points or (
                            c_point[0], c_point[1] + width) in points:
                        if left:
                            x = c_point[0]
                        else:
                            x = c_point[0] - w
                    else:
                        if center[0] < c_point[0]:
                            x = c_point[0]
                        else:
                            x = c_point[0] - w
                    vrects.append([x, y, w, h])
                    # cv2.circle(dmask, c_point, 17, 255, -1)
                    # hrects.append([int(center[0]-2), int(center[1]-2), 4, 4])


                    continue
    # for r in hrects:
    #     # r[3] = wmeanl
    #     cv2.rectangle(dmask, r, (255, 0, 0), -1)
    #
    # for r in vrects:
    #     # r[2] = wmean
    #     cv2.rectangle(dmask, r, (255, 0, 0), -1)
    return hrects + vrects
