import random
import numpy as np
import cv2


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
                print(x1, x2, y1, y2, len(img), len(img[0]))
    return img


def draw_contours(img, contours, min_threshold=100, color=None, kernel=None):
    out = np.zeros_like(img)

    if color is None:
        color = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
    for cc in contours:
        timg = np.zeros_like(img)
        c = cc[0]
        area = cc[1]
        if area > min_threshold:
            cv2.drawContours(timg, [c], -1, 255, -1)
            if kernel:
                timg = cv2.morphologyEx(timg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
        out[np.where(timg > 0)] = color
    return out


def get_center(cntor):
    #     M = cv2.moments(cntor)
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])

    #     return (cX, cY)
    sh = cntor.shape
    s = np.mean(cntor.reshape(sh[0], 2), axis=0)
    return s.astype(int)


def get_contours(img, min_threshold=100, max_threshold=10 * 9):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        mn = int(np.mean(c))
        anc.append(mn)
        if as_dict:
            out_dict = {**out_dict, **{i: mn for i in c}}
    if as_dict:
        return out_dict
    return anc
#
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
