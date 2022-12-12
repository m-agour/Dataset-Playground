import random

import cv2
import numpy as np

from helpers import get_contours, draw_contours, cluster_points

from sklearn.cluster import MeanShift, estimate_bandwidth


def get_wall(r, g, b, threshold=50):
    wimg = cv2.inRange(r, 255 - 254, 255 - 157) & cv2.inRange(g, 255 - 198, 255 - 156) & cv2.inRange(b, 255 - 255,
                                                                                                     255 - 0)
    pts = {}
    #
    contours = get_contours(wimg, threshold)

    wimg *= 0

    wimg = draw_contours(wimg, contours, color=255, kernel=(30, 30))

    blur = cv2.GaussianBlur(wimg, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 20))
    horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 70))
    vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    wimg *= 0
    # Combine masks and remove lines
    table_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)
    wimg[np.where(table_mask == 255)] = 255
    wimg = cv2.erode(wimg, np.ones((4, 4)), iterations=3)

    wimg = ~wimg

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    vimg = cv2.morphologyEx(wimg, cv2.MORPH_OPEN, horizontal_kernel, iterations=3) & wimg

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 17))
    himg = cv2.morphologyEx(wimg, cv2.MORPH_OPEN, vertical_kernel, iterations=3) & wimg

    mut = vimg & himg
    voimg = vimg - mut
    hoimg = himg - mut

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    hoimg = cv2.morphologyEx(hoimg, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    voimg = cv2.morphologyEx(voimg, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    f = np.zeros_like(wimg)
    hs = []

    contours_h = get_contours(hoimg, 0)
    contours_v = get_contours(voimg, 0)
    contours = contours_h + contours_v
    hoimg *= 0
    voimg *= 0

    mwallwidth = round(np.median([c[2][3] for c in contours_v] + [c[2][2] for c in contours_h]))
    inc = mwallwidth
    lines_mask = np.zeros_like(f)

    xs, ys = [], []
    for cc in contours:
        x, y, w, h = cc[2]
        xs.extend([x, x + w])
        ys.extend([y, y + h])

    xs = sorted(xs)
    ys = sorted(ys)

    xs = cluster_points(xs, mwallwidth // 2, as_dict=True)
    ys = cluster_points(ys, mwallwidth // 2, as_dict=True)
    print(xs, ys)

    for cc in contours_h:
        x, y, w, h = cc[2]

        x = xs[x]
        y = ys[y]

        if w * 1.3 < mwallwidth:
            continue
        w = mwallwidth
        cv2.rectangle(f, (x, y), (x + w, y + h), 255, -1)
        hs.extend([x, x + w])

        if min(w, h) > mwallwidth * 0.8:
            pts[(x, y)] = {'up': True, 'left': True, 'horiz': True}
            pts[(x + w, y)] = {'up': True, 'left': False, 'horiz': True}
            pts[(x, y + h)] = {'up': False, 'left': True, 'horiz': True}
            pts[(x + w, y + h)] = {'up': False, 'left': False, 'horiz': True}

        cv2.line(lines_mask, (x, y - inc), (x, y + h + inc), 100, 1)
        cv2.line(lines_mask, (x + w, y - inc), (x + w, y + h + inc * 2), 100, 1)
        # cv2.line(lines_mask, (x - inc, y), (x + w + inc, y), 100, 1)
        # cv2.line(lines_mask, (x - inc, y + h), (x + w + inc, y + h), 100, 1)

    for cc in contours_v:
        x, y, w, h = cc[2]
        x = xs[x]
        y = ys[y]
        if h * 1.3 < mwallwidth:
            continue

        h = mwallwidth
        if min(w, h) > mwallwidth * 0.8:
            pts[(x, y)] = {'up': True, 'left': True, 'horiz': False}
            pts[(x + w, y)] = {'up': True, 'left': False, 'horiz': False}
            pts[(x, y + h)] = {'up': False, 'left': True, 'horiz': False}
            pts[(x + w, y + h)] = {'up': False, 'left': False, 'horiz': False}

        cv2.rectangle(f, (x, y), (x + w, y + h), 255, -1)
        cv2.line(lines_mask, (x - inc, y), (x + w + inc, y), 100, 1)
        cv2.line(lines_mask, (x - inc, y + h), (x + w + inc, y + h), 100, 1)
        # cv2.line(lines_mask, (x, y - inc), (x, y + h + inc), 100, 1)
        # cv2.line(lines_mask, (x + w, y - inc), (x + w, y + h + inc), 100, 1)
        hs.extend([y, y + h])

    fb = f.copy()
    f[lines_mask > 0] = 255

    contours = get_contours(mut, 50)
    mut *= 0
    for cc in contours:
        x, y, w, h = cc[2]
        cv2.rectangle(mut, (x, y), (x + w, y + h), 255, -1)

    f = f | mut

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mwallwidth // 2, mwallwidth // 4))
    fm = cv2.morphologyEx(f, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)

    fm = cv2.morphologyEx(fm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, mwallwidth//8)), iterations=1)
    fm = cv2.morphologyEx(fm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (mwallwidth//8, 1)), iterations=1)

    f = fm
    f = cv2.erode(f, np.ones((mwallwidth // 4, mwallwidth // 4), np.uint8), iterations=1)
    # f = cv2.erode(f, np.ones((mwallwidth//8 ,mwallwidth//8), np.uint8), iterations=2)

    rows, cols = f.shape
    M = np.float32([[1, 0, -mwallwidth // 6], [0, 1, -mwallwidth // 6]])
    f = cv2.warpAffine(f, M, (cols, rows))

    # wimg = cv2.erode(wimg, np.ones((mwallwidth//4 ,mwallwidth//4), np.uint8), iterations=1)
    wimg = cv2.merge((wimg, wimg, wimg))
    img_blur = cv2.GaussianBlur(f, (3, 3), 0)

    # cv2.Sobel(img_blur, sobelx, cv2.CV_64F, 1, 0, 5);
    # cv2.Sobel(img_blur, sobely, cv2.CV_64F, 0, 1, 5);
    # cv2.Sobel(img_blur, sobelxy, cv2.CV_64F, 1, 1, 5);

    # cv2.imshow("Sobel X", sobelx)
    # cv2.waitKey(0)
    # cv2.imshow("Sobel Y", sobely)
    # cv2.waitKey(0)
    # cv2.imshow("Sobel XY using Sobel() function", sobelxy)
    # cv2.waitKey(0)

    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0,
                       ksize=1)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1,
                       ksize=1)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1,
                        ksize=1)  # Combined X and Y Sobel Edge Detection

    # print(img_blur)
    # edges = cv2.Canny(img_blur.astype(np.uint8), 100, 200, )
    # cv2.imshow("Canny edge detection", edges);
    # cv2.waitKey(0)
    #
    # # Display Sobel Edge Detection Images
    # cv2.imshow('Sobel X', sobelx)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel Y', sobely)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel Y', sobelxy)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel Y', ((sobelx > 0) | (sobely > 0)).astype(np.uint8) * 255)
    # cv2.waitKey(0)
    contours = get_contours(f, 0)


    return f, mwallwidth, pts

# img = cv2.imread('Mo/0013147_0000000.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# r, g, b = cv2.split(img)
# x, y, pts = get_wall(r, g, b)
# print(pts)
# cv2.imshow('oh', x)
# cv2.waitKey(0)
