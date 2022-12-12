import cv2
import numpy as np

from helpers import get_center


def get_doors(r, g, b, img, width, real_width, min_th=50, points={}):
    width = real_width

    imgf = cv2.bilateralFilter(img, 10, 30, 30, None)

    imgf_hsv = cv2.cvtColor(imgf, cv2.COLOR_RGB2HSV)
    _, s, _ = cv2.split(imgf_hsv)

    dmask = np.zeros_like(b)

    mask = ((b < 100) & (g < 100) & (s > 100) & (r > 60) & (b < 170)).astype(np.uint8) * 255

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
    #
    # for i, c in enumerate(contours):
    #
    #     contour = c[0]
    #     area = c[1]
    #     brect = c[2]
    #
    #     masks[i] = np.ones(b.shape)
    #
    #     x, y, w, h = brect
    #     inc = 7
    #     door_width = width
    #
    #     def dist(p1, p2):
    #         return np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])
    #
    #     horizontal = True if h < w else False
    #     com = get_center(contour)
    #     thr = width * 3
    #
    #     com = get_center(contour)
    #
    #     door_pts_curr = [p for p in door_pts if dist(p, com) < thr]
    #     # com = np.mean(door_pts_curr, axis=0)
    #
    #     com = np.median(door_pts_curr, axis=0)
    #
    #     if horizontal:
    #         y1 = y
    #         y2 = y + h
    #         if abs(y1 - com[1]) < abs(y2 - com[1]):
    #             y_f = y1 + door_width // 3
    #         else:
    #             y_f = y2 - door_width // 3
    #         door_big_rect = (x - inc, int(y_f - door_width / 2), w + 2 * inc, door_width)
    #
    #     else:
    #         x1 = x
    #         x2 = x + w
    #         if abs(x1 - com[0]) < abs(x2 - com[0]):
    #             x_f = x1 + door_width // 3
    #         else:
    #             x_f = x2 - door_width // 3
    #         door_big_rect = (x_f - door_width // 2, y - inc, door_width, h + 2 * inc)
    #         # cv2.line(dmask, ((door_width + 2 * x_f)//2, 0), ((door_width + 2 * x_f)//2, 3000), 255 ,1)
    #
    #     cv2.rectangle(dmask, door_big_rect, 255, -1)
    temp_points_h = [p for p in points if points[p]['horiz'] == 1]
    temp_points_v = [p for p in points if points[p]['horiz'] == 0]
    vrects = []
    hrects = []
    dmask *= 0
    for c in contours:
        x, y, w, h = np.array(c[2])

        center = get_center(c[0])

        def dist(p1, p2):
            return np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])

        door_pts_curr = [p for p in door_pts if dist(p, center) < width * 2]
        # com = np.mean(door_pts_curr, axis=0)
        center = np.median(door_pts_curr, axis=0)
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
            # if horiz:
            #     w = w + 3 * width // 4
            #     x += width // 4
            #     y += width // 4
            #
            #     # x -= width // 2
            #     w += width
            #
            # else:
            #     h = h + 3 * width // 4
            #     # y += width // 4
            #     # x += width // 4
            #
            #     # y -= width // 2
            #     h += width

            offset = 0

            # offset = width // 8
            th = width * 3
            if w > h:
                if min_dist > 1.5*th:
                    c_point = sorted(points, key=lambda p: min([abs(p[1] - y), abs(p[1] - (y + h))]))[0]
                    axis_only = 1

                curr_d = min([abs(c_point[1] - y), abs(c_point[1] - (y + h))])
                if curr_d <= 1.5*th or (axis_only and (curr_d <= 2*th)):
                    h = width*3
                    up = points[c_point]['up']

                    if (c_point[0], c_point[1] + real_width) in points:
                        y = c_point[1]
                    elif (c_point[0], c_point[1] - real_width) in points:
                        y = c_point[1] - h - offset
                    elif (c_point[0] - real_width, c_point[1]) in points or (
                            c_point[0] + real_width, c_point[1]) in points:
                        if up:
                            y = c_point[1]
                        else:
                            y = c_point[1] - h - offset
                    else:
                        if center[1] < c_point[1]:
                            y = c_point[1]
                        else:
                            y = c_point[1] - h - offset
                    h = width + offset // 2
                    hrects.append([x - offset, y - offset, w, h])
                    continue
            else:
                if min_dist > 1.5*th:
                    c_point = sorted(points, key=lambda p: min([abs(p[0] - x), abs(p[0] - (x + w))]))[0]
                    axis_only = 1

                curr_d = min([abs(c_point[0] - x), abs(c_point[0] - (x + w))])
                if curr_d <= 1.5*th or (axis_only and (curr_d <= 2*th)):
                    w = width + offset // 2
                    h = h*3
                    left = points[c_point]['left']

                    if (c_point[0] + real_width, c_point[1]) in points:
                        x = c_point[0]
                    elif (c_point[0] - real_width, c_point[1]) in points:
                        x = c_point[0] - w - offset

                    elif (c_point[0], c_point[1] - real_width) in points or (
                            c_point[0], c_point[1] + real_width) in points:
                        if left:
                            x = c_point[0]
                        else:
                            x = c_point[0] - w - offset
                    else:
                        if center[0] < c_point[0]:
                            x = c_point[0]
                        else:
                            x = c_point[0] - w - offset
                    vrects.append([x - offset, y - offset, w, h])
                    # cv2.circle(dmask, c_point, 17, 255, -1)

                    continue
    for r in hrects:
        # r[3] = wmeanl
        cv2.rectangle(dmask, r, (255, 0, 0), -1)

    for r in vrects:
        # r[2] = wmean
        cv2.rectangle(dmask, r, (255, 0, 0), -1)
    return dmask
