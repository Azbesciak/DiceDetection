from __future__ import division

import traceback

from pylab import *
import skimage as ski
from skimage import data, io, filters, exposure, measure
from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import disk
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray, gray2rgb
from skimage.filters.edges import convolve
from matplotlib import pylab as plt
import numpy as np
from skimage import feature
from numpy import array
from IPython.display import display
from ipywidgets import interact, interactive, fixed
from ipywidgets import *
from ipykernel.pylab.backend_inline import flush_figures
from multiprocessing.pool import ThreadPool
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

dicesToRead = [
    '01', '02', '03', '04', '05',
    '06', '07', '08', '09', '10',
    '11', '12', '13', '14', '15',
    '16', '17', '18', '19', '20',
    '21'
]

dicesToRead = [
    '18', '13'
]
# dicesToRead = [
#     '08'
# ]

params_for_dices = [
    {'gamma': 0.4, 'sig': 2.7, 'l': 91, 'u': 90, 'edgeFunc': lambda img, p: get_edges(img, p)},
    {'sig': 4, 'low': 0.05, 'high': 0.3, 'edgeFunc': lambda img, p: just_canny_and_dilation(img, p)},
    # {'l': 0.6, 'u': 15.4, 'tresh': 0.1, 'edgeFunc': lambda img, p: simple_gray(img, p)},
    # {'gamma': 0.5, 'sig': 1.4, 'l': 0, 'u': 100, 'edgeFunc': lambda img, p: get_edges(img, p)},
    # {'low': 0.05, 'high': 0.3, 'sig': 3, 'edgeFunc': lambda img, p: edges_by_sharp_color(img, p)},
    # {'l': 0.6, 'u': 15.4, 'tresh': 0.4, 'lev': 0.19, 'edgeFunc': lambda img, p: edges_with_contours(img, p)}
]

params_for_dotes = [
    {'sig': 0.4, 'low': 0.1, 'high': 0.3, 'edgeFunc': lambda img, p: just_canny_and_dilation(img, p)},
    {'gamma': 1, 'sig': 1, 'l': 0, 'u': 100, 'edgeFunc': lambda img, p: get_edges(img, p)}
]

dices = [io.imread('./dices/dice{0}.jpg'.format(i)) for i in dicesToRead]


def drawDiceImage(i, img):
    plt.subplot(2, 3, i)
    plt.imshow(img)


def drawDiceImageAligned(total, i, img):
    in_row = int(total / 3) + 1
    ax = plt.subplot(in_row, int(total / in_row), i)
    plt.imshow(img)
    return ax


def get_edges(img, p):
    img = rgb2gray(img)
    if 'l' in p and 'u' in p:
        pp, pk = np.percentile(img, (p['l'], p['u']))
        img = exposure.rescale_intensity(img, in_range=(pp, pk))
    if 'gamma' in p:
        img = exposure.adjust_gamma(img, p['gamma'])
        # img = img ** p['gamma']
    img = ski.feature.canny(img, sigma=p['sig'])
    return img


def simple_gray(img, p):
    img = rgb2gray(img)
    pp, pk = np.percentile(img, (p['l'], p['u']))
    img = exposure.rescale_intensity(img, in_range=(pp, pk))
    img = filters.prewitt(img)
    img[img > p['tresh']] = 1
    img[img <= p['tresh']] = 0
    return img


def edges_by_sharp_color(img, p):
    img = rgb2hsv(img)
    for x in range(len(img)):
        for y in range(len(img[0])):
            img[x][y] = [img[x][y][0], 1, 1]
    img = hsv2rgb(img)
    img = rgb2gray(img)
    img = ski.feature.canny(img, sigma=p['sig'], low_threshold=p['low'], high_threshold=p['high'])
    return img


def edges_with_contours(img, p):
    pp, pk = np.percentile(img, (p['l'], p['u']));
    img = exposure.rescale_intensity(img, in_range=(pp, pk))
    img = rgb2gray(img)
    blackWhite = np.zeros([len(img), len(img[0])]) + 1 - img
    contours = measure.find_contours(blackWhite, p['lev'])
    for contours in contours:
        for con in contours:
            blackWhite[int(con[0])][int(con[1])] = 1

    blackWhite[blackWhite < p['tresh']] = 0
    blackWhite[blackWhite >= p['tresh']] = 1
    return blackWhite


def just_canny_and_dilation(img, p):
    img = rgb2gray(img)
    img = ski.morphology.dilation(img)
    img = ski.feature.canny(img, sigma=p['sig'], low_threshold=p['low'], high_threshold=p['high'])
    return img


def get_rectangles_with_dim(rectangles, dim_upper):
    values = [x for x in rectangles if dim_upper >= x["width"] / x["height"] >= 1 / dim_upper]
    sort_by_key(values, 'rarea')
    return values


def sort_by_key(values, key, rev=True):
    values.sort(key=lambda x: x[key], reverse=rev)


def try_to_find_dices(img):
    filtered_dices = []
    for p in params_for_dices:
        try:
            dices_candidates = look_for_dices(img, p)
            filtered_dices.extend(dices_candidates)
        except ValueError:
            pass

    return filtered_dices


def look_for_dices(img, params):
    image = params['edgeFunc'](img, params)
    regions = find_regions(image)
    return filter_dices(regions)


def parse_image(img):
    dices = try_to_find_dices(img)

    fig, ax = plt.subplots(figsize=(10, 6))
    if len(dices) > 0:
        firstOk = dices[0]
        # filtered = [x for x in dices if
        #             x['height'] >= firstOk['height'] / 2 and x['width'] >= firstOk['width'] / 2]
        total_length = len(dices)
        if total_length > 0:
            dots_on_dices = prepare_dice_to_draw(dices, img)
            draw_dices(ax, zip(dots_on_dices, dices), img)
    ax.imshow(img)


def prepare_dice_to_draw(filtered, img):
    dots = []
    for f in filtered:
        dots_on_dice = find_on_dice(img, f)
        for dot in dots_on_dice:
            new_x = dot['rect'].get_x() + f['minx']
            new_y = dot['rect'].get_y() + f['miny']
            dot['rect'].set_xy((new_x, new_y))
        dots.append(dots_on_dice)
    return dots


def draw_dices(ax, dices, img):
    for (dots, dice) in dices:
        dots_amount = len(dots)
        if dots_amount > 0 or True:
            for dot in dots:
                ax.add_patch(dot['rect'])
            size = len(img) / 250
            cv2.putText(img, str(dots_amount), (dice['minx'], dice['miny']), 2, fontScale=size,  # 3
                        color=(0, 140, 150), thickness=max(int(size), 2))
            ax.add_patch(dice['rect'])


def filter_dices(regions):
    values = []
    for region in regions:
        validate_region(region, values, lambda rect: rect['area'] >= 70, 'orange')
    values = get_rectangles_with_dim(values, 2)
    return values


def find_regions(image):
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(2))
    cleared = clear_border(bw)
    label_image = label(cleared)
    return regionprops(label_image)


def validate_region(region, values, validation_fun, color='blue'):
    miny, minx, maxy, maxx = region.bbox
    height = maxy - miny
    width = maxx - minx
    rect = mpatches.Rectangle((minx, miny), width, height,
                              fill=False, edgecolor=color, linewidth=2)
    rect_repr = {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy, "rect": rect, "area": region.area,
                 "width": width, "height": height, 'rarea': width * height, 'fill': region.convex_area}
    if validation_fun(rect_repr):
        values.append(rect_repr)


def is_one_value_image(img):
    return len(list(set([x for sublist in img for x in sublist]))) <= 1


def get_regions_from_dice(dice_img, par):
    regions = []
    for p in par:
        img = p['edgeFunc'](dice_img, p)
        if is_one_value_image(img):
            return []
        regions.extend(find_regions(img))
    return regions


def find_on_dice(org_img, dice):
    dice_img_copy = org_img[dice['miny']:dice['maxy'], dice['minx']:dice['maxx']]
    regions = get_regions_from_dice(dice_img_copy, params_for_dotes)

    valid_regions = []
    img_size = len(dice_img_copy) * len(dice_img_copy[0])
    for region in regions:
        validate_region(region, valid_regions, lambda rect: img_size * .005 <= rect['rarea'] <= img_size * .15)
    ratio = 1.4
    filtered = get_rectangles_with_dim(valid_regions, ratio)

    if len(filtered) > 0:
        filtered = filter_dots(filtered, ratio, dice)

    return filtered


def filter_dots(filtered, ratio, dice):
    filtered = remove_in_corners(filtered, dice)
    filtered = remove_mistaken_dots(filtered, ratio)
    filtered = remove_overlaped(filtered)
    filtered = remove_smaller_than_half_of_the_biggest(filtered)
    filtered = remove_the_farthest_if_more_than_six(filtered)
    return filtered


def remove_in_corners(filtered, dice):
    res = []
    bound_lim = 0.05
    width_bound = dice['width'] * bound_lim
    height_bound = dice['height'] * bound_lim
    for f in filtered:
        if not ((width_bound > f['minx'] and height_bound > f['miny']) or
                (width_bound > f['minx'] and dice['height'] - height_bound < f['maxy']) or
                (dice['width'] - width_bound < f['maxx'] and height_bound > f['miny']) or
                (dice['width'] - width_bound < f['maxx'] and dice['height'] - height_bound < f['maxy'])):
            res.append(f)
    return res


def remove_the_farthest_if_more_than_six(filtered):
    for f in filtered:
        f['center'] = {
            'x': int((f['minx'] + f['maxx']) / 2),
            'y': int((f['miny'] + f['maxy']) / 2)
        }
    for f1 in filtered:
        f1['total_dist'] = 0
        for f2 in filtered:
            f1['total_dist'] += get_distance_between(f1, f2)
    sort_by_key(filtered, 'total_dist', False)
    if len(filtered) > 6:
        filtered = filtered[0:6]
    return filtered


def get_distance_between(f1, f2):
    return sqrt(
        abs(f1['center']['x'] - f2['center']['x']) ** 2 +
        abs(f1['center']['y'] - f2['center']['y']) ** 2
    )


def remove_smaller_than_half_of_the_biggest(filtered):
    rareas = get_rareas(filtered)
    if len(rareas) < 1:
        return []
    max_area = max(rareas)
    filtered = [f for f in filtered if f['rarea'] > 0.5 * max_area]
    return filtered


def remove_mistaken_dots(filtered, ratio):
    by_rarea = get_rareas(filtered)
    if len(by_rarea) < 1:
        return []
    center_point = np.percentile(by_rarea, 80)
    filtered_first = [f for f in filtered if center_point * 1 / ratio <= f['rarea'] <= center_point * ratio]
    if len(filtered_first) < 0.3 * len(filtered):
        ratio = ratio ** 2
        filtered_first = [f for f in filtered if center_point * 1 / ratio <= f['rarea'] <= center_point * ratio]
        if len(filtered_first) < 0.3 * len(filtered):
            center_point = sum(by_rarea) / len(by_rarea)
            filtered = [f for f in filtered if center_point * 1 / ratio <= f['rarea'] <= center_point * ratio]
    else:
        filtered = filtered_first
    return filtered


def get_rareas(filtered):
    return [f['rarea'] for f in filtered]


def remove_overlaped(filtered):
    res = []
    for f1 in filtered:
        isOk = True
        for f2 in filtered:
            if f1 != f2:
                if f1['rarea'] < f2['rarea']:
                    if (sum([
                        f2['miny'] <= f1['miny'] <= f2['maxy'] and f2['minx'] <= f1['minx'] <= f2['maxx'],
                        f2['miny'] <= f1['maxy'] <= f2['maxy'] and f2['minx'] <= f1['minx'] <= f2['maxx'],
                        f2['miny'] <= f1['miny'] <= f2['maxy'] and f2['minx'] <= f1['maxx'] <= f2['maxx'],
                        f2['miny'] <= f1['maxy'] <= f2['maxy'] and f2['minx'] <= f1['maxx'] <= f2['maxx']
                    ]) >= 2):
                        isOk = False
                        break
        if isOk:
            res.append(f1)
    return res


def look_for_dices_on_image():
    fig = plt.figure(facecolor="black", figsize=(60, 60))
    for i, image in enumerate(dices):
        try:
            parse_image(image)
        except Exception:
            print("error with {0}".format(i))
            traceback.print_exc()
    plt.tight_layout()
    plt.show()
    fig.savefig("dices.pdf", facecolor="black")
    plt.close()


look_for_dices_on_image()
