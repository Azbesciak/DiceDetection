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
    '11', '12', '13', '14'
]

# dicesToRead = [
#     '18', '13'
# ]
dicesToRead = [
    '06'
]

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
    pp, pk = np.percentile(img, (p['l'], p['u']))
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
        image_area = len(img) * len(img[0])
        filtered = [x for x in dices if x['rarea'] > image_area * 0.005]
        filtered = remove_with_single_color(filtered, img)
        sort_by_key(filtered, 'rarea')
        filtered = remove_outliers_on_field(filtered, 'width', 1.6)
        filtered = remove_outliers_on_field(filtered, 'height', 1.6)
        total_length = len(filtered)
        if total_length > 0:
            dices = filter_dices_candidates(filtered)
            dots_on_dices = prepare_dice_to_draw(dices, img)
            dices_to_draw = filter_dices_to_draw(zip(dots_on_dices, dices))
            draw_dices(ax, dices_to_draw, img)
    ax.imshow(img)


def remove_with_single_color(candidates, img):
    filtered = [x for x in candidates if is_multi_color(get_img_fragment(x, img))]
    return filtered


def is_multi_color(img):
    img_copy = np.copy(img)
    vals = []
    ar_range = int(255 / 5) + 1
    for x in range(len(img_copy)):
        for y in range(len(img_copy[0])):
            r = int(img_copy[x][y][0] / ar_range) * ar_range
            g = int(img_copy[x][y][1] / ar_range) * ar_range
            b = int(img_copy[x][y][2] / ar_range) * ar_range
            img_copy[x][y] = [r, g, b]
            vals.append((r, g, b))
    unique_vals = len(set(vals))
    return not exposure.is_low_contrast(img_copy) and unique_vals > 10


def filter_dices_candidates(candidates):
    res = []
    for c1 in candidates:
        should_add = True
        for c2 in candidates:
            if has_lower_real_area(c1, c2) and has_common_field(c1, c2):
                extend_dice_area(c2, c1)
                should_add = False
        if should_add:
            res.append(c1)
    return res


def has_common_field(f1, f2):
    min_x = max(0, min(f1['maxx'], f2['maxx']) - max(f1['minx'], f2['minx']))
    min_y = max(0, min(f1['maxy'], f2['maxy']) - max(f1['miny'], f2['miny']))
    common_part = min_x * min_y
    minimum_real_area = min(f1['rarea'], f2['rarea'])
    return common_part >= minimum_real_area * 0.25


def extend_dice_area(extended, consumed):
    extended['minx'] = min(extended['minx'], consumed['minx'])
    extended['maxx'] = max(extended['maxx'], consumed['maxx'])
    extended['miny'] = min(extended['miny'], consumed['miny'])
    extended['maxy'] = max(extended['maxy'], consumed['maxy'])
    extended['height'] = extended['maxy'] - extended['miny']
    extended['width'] = extended['maxx'] - extended['minx']
    extended['rarea'] = extended['width'] * extended['height']
    extended['rect'].set_x(extended['minx'])
    extended['rect'].set_y(extended['miny'])
    extended['rect'].set_width(extended['width'])
    extended['rect'].set_height(extended['height'])


def filter_dices_to_draw(candidates):
    candidates = [(dts, dcs) for (dts, dcs) in candidates if len(dts) > 0]
    return candidates


def prepare_dice_to_draw(dices_region, img):
    dots = []
    for dice_reg in dices_region:
        dots_on_dice = find_on_dice(img, dice_reg)
        for dot in dots_on_dice:
            move_rectangle(dot, dice_reg)
        dots.append(dots_on_dice)
    return dots


def move_rectangle(container, cords):
    new_x = container['rect'].get_x() + cords['minx']
    new_y = container['rect'].get_y() + cords['miny']
    container['rect'].set_xy((new_x, new_y))


def draw_dices(ax, dices, img):
    for (dots, dice) in dices:
        dots_amount = len(dots)
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
    dice_img_copy = get_img_fragment(dice, org_img)
    regions = get_regions_from_dice(dice_img_copy, params_for_dotes)

    valid_regions = []
    img_size = len(dice_img_copy) * len(dice_img_copy[0])
    for region in regions:
        validate_region(region, valid_regions, lambda rect: img_size * .005 <= rect['rarea'] <= img_size * .15)

    if len(valid_regions) > 0:
        return filter_dots(valid_regions, dice)
    return []


def get_img_fragment(cords, org_img):
    return org_img[cords['miny']:cords['maxy'], cords['minx']:cords['maxx']]


def filter_dots(filtered, dice):
    ratio = 1.4
    filtered = get_rectangles_with_dim(filtered, ratio)
    filtered = remove_too_small_and_too_big(filtered, dice)
    filtered = remove_in_corners(filtered, dice)
    filtered = remove_mistaken_dots(filtered, ratio)
    filtered = remove_overlaped(filtered)
    filtered = remove_outliers_on_field(filtered, 'fill')
    filtered = remove_smaller_than_half_of_the_biggest(filtered)
    filtered = remove_the_farthest_if_more_than_six(filtered)
    return filtered


def remove_outliers_on_field(filtered, field_name, accept_ratio=1.2):
    if len(filtered) == 0:
        return []
    filings = get_by_field_name(filtered, field_name)
    comparing_idx = int(len(filings) * 0.25)
    to_compare = filings[comparing_idx]
    dif = to_compare * (accept_ratio - 1)
    return [f for f in filtered if to_compare + dif >= f[field_name] >= to_compare - dif]


def remove_too_small_and_too_big(filtered, dice):
    return [x for x in filtered if dice['rarea'] / 6 >= x['rarea'] >= dice['rarea'] * 0.01]


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
    rareas = get_by_field_name(filtered, 'rarea')
    if len(rareas) < 1:
        return []
    max_area = max(rareas)
    filtered = [f for f in filtered if f['rarea'] > 0.5 * max_area]
    return filtered


def remove_mistaken_dots(filtered, ratio):
    by_rarea = get_by_field_name(filtered, 'rarea')
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


def get_by_field_name(filtered, field_name):
    return [f[field_name] for f in filtered]


def remove_overlaped(filtered):
    res = []
    for f1 in filtered:
        isOk = True
        for f2 in filtered:
            if has_lower_real_area(f1, f2) and does_include(f1, f2, 2):
                isOk = False
                break
        if isOk:
            res.append(f1)
    return res


def has_lower_real_area(smaller, bigger):
    return not(smaller['minx'] == bigger['minx'] and smaller['miny'] == bigger['miny'] and
           smaller['maxx'] == bigger['maxx'] and smaller['maxy'] == bigger['maxy']) and \
           smaller['rarea'] <= bigger['rarea']


def does_include(inner, outer, corners):
    return sum([
        outer['miny'] <= inner['miny'] <= outer['maxy'] and outer['minx'] <= inner['minx'] <= outer['maxx'],
        outer['miny'] <= inner['maxy'] <= outer['maxy'] and outer['minx'] <= inner['minx'] <= outer['maxx'],
        outer['miny'] <= inner['miny'] <= outer['maxy'] and outer['minx'] <= inner['maxx'] <= outer['maxx'],
        outer['miny'] <= inner['maxy'] <= outer['maxy'] and outer['minx'] <= inner['maxx'] <= outer['maxx']
    ]) >= corners


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
