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
    '16','17', '19', '20'
]

# dicesToRead = [
   # '17',
   # '11',
   #  '01',
   #  '14',
    # '02'
    # '08',
    # '09',
    # '16',
    # '07',
    # '19','20'
# ]

params_for_dices = [
    {'gamma': 0.4, 'sig': 2.7, 'l': 91, 'u': 90, 'edgeFunc': lambda img, p: get_edges(img, p)},
    {'sig': 4, 'low': 0.05, 'high': 0.3, 'edgeFunc': lambda img, p: just_canny_and_dilation(img, p)},
    {'tresh': 0.8, 'edgeFunc': lambda img, p: get_by_hsv_value(img, p)},
    {'edgeFunc': lambda img, p: splashing_image(img, p)},
    {'gamma': 5, 'tresh': 0.05, 'edgeFunc': lambda img, p: sobel_and_scharr_connected(img, p)},
    {'l': 10, 'u': 90, 'edgeFunc': lambda img, p: high_percentile_shrink(img, p)}
]

params_for_dotes = [
    {'sig': 0.4, 'low': 0.1, 'high': 0.3, 'edgeFunc': lambda img, p: just_canny_and_dilation(img, p)},
    {'gamma': 1, 'sig': 1, 'l': 0, 'u': 100, 'edgeFunc': lambda img, p: get_edges(img, p)},
    # {'edgeFunc': lambda img, p: double_sobel(img, p)},
    # {'gamma': 4, 'close': 2, 'median': 2, 'sobel': True, 'edgeFunc': lambda img, p: dots_filter(img, p)},
    # {'gamma': 4, 'close': 2, 'median': 2, 'edgeFunc': lambda img, p: dots_filter(img, p)}
]


dices = [io.imread('./dices/dice{0}.jpg'.format(i)) for i in dicesToRead]


def draw_dice_image(i, img):
    plt.subplot(1, 2, i)
    plt.imshow(img)


def draw_dice_image_aligned(total, i, img):
    in_row = int(total / 3) + 1
    ax = plt.subplot(in_row, int(total / in_row), i)
    plt.imshow(img)
    return ax


def debug_show(img1, img2):
    draw_dice_image(1, img1)
    draw_dice_image(2, img2)
    plt.show()


def get_by_hsv_value(img, p):
    img = rgb2hsv(img)
    img[img[:, :, 2] > p['tresh']] = 1
    img[img[:, :, 2] <= p['tresh']] = 0
    img = rgb2gray(img)
    img = ski.morphology.erosion(img, square(2))
    img = ski.morphology.erosion(img, square(3))
    img = ski.morphology.opening(img)
    return img


def get_edges(img, p):
    img = rgb2gray(img)
    if 'l' in p and 'u' in p:
        pp, pk = np.percentile(img, (p['l'], p['u']))
        img = exposure.rescale_intensity(img, in_range=(pp, pk))
    if 'gamma' in p:
        img = exposure.adjust_gamma(img, p['gamma'])
    img = ski.feature.canny(img, sigma=p['sig'])
    return img


def just_canny_and_dilation(img, p):
    img = rgb2gray(img)
    img = ski.morphology.dilation(img)
    img = ski.feature.canny(img, sigma=p['sig'], low_threshold=p['low'], high_threshold=p['high'])
    return img


def high_percentile_shrink(img, p):
    temp = rgb2gray(img)
    temp = temp ** 3
    temp = ski.exposure.equalize_adapthist(temp)
    if np.average(temp) > 0.5:
        temp = 1 - temp
    temp = apply_threshold(temp)
    return temp


def sobel_and_scharr_connected(img, p):
    temp = img
    temp = rgb2gray(temp)

    temp = temp ** p['gamma']
    temp1 = ski.filters.sobel(temp)
    temp2 = ski.filters.scharr(temp)
    temp = temp1 + temp2
    temp[temp > p['tresh']] = 1
    temp = ski.morphology.dilation(temp)
    return temp


def splashing_image(img, p):
    temp = -img
    temp = rgb2gray(temp)
    temp = (temp * 2) ** 0.5 / 2
    temp = ski.morphology.erosion(temp, square(10))
    temp = 1 - temp
    temp = filters.median(temp, square(25))
    temp = apply_threshold(temp)
    temp = filters.median(temp, square(50))
    return temp


def dots_filter(img, p):
    temp = -img
    temp = rgb2gray(temp)
    temp = (temp ** p['gamma'])
    temp = ski.morphology.closing(temp, square(p['close']))
    temp = apply_threshold(temp)
    temp = filters.median(temp, square(p['median']))
    if 'sobel' in p:
        temp = filters.sobel(temp)
        temp = apply_threshold(temp)
    return temp


def double_sobel(img, p):
    temp = rgb2gray(img)
    temp = ski.filters.median(temp, square(10))
    temp1 = apply_threshold(temp)
    temp = rgb2gray(img) - temp1
    temp[temp < 0] = 0
    temp = ski.filters.median(temp, square(10))
    temp = apply_threshold(temp)
    temp = 1 - temp
    temp = temp1 - temp
    temp[temp < 0] = 0
    temp = ski.filters.sobel(temp)
    return temp


def apply_threshold(img):
    thresh = threshold_otsu(img)
    img[img <= thresh] = 0
    img[img > thresh] = 1
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
    dices_candidates = try_to_find_dices(img)
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(dices_candidates) > 0:
        dices = filter_dices_after_discovering(dices_candidates, img)
        dots_on_dices = prepare_dice_to_draw(dices, img)
        dices_to_draw = filter_dices_to_draw(zip(dots_on_dices, dices))
        draw_dices(ax, dices_to_draw, img)
    ax.imshow(img)


def filter_dices_after_discovering(candidates, img):
    min_coverage = .0025
    width, height = len(img), len(img[0])
    image_area = width * height
    filtered = [x for x in candidates if x['rarea'] > image_area * min_coverage]
    filtered = remove_with_single_color(filtered, img)
    sort_by_key(filtered, 'rarea')
    filtered = remove_outliers_on_field(filtered, 'width', 1.6, False)
    filtered = remove_outliers_on_field(filtered, 'height', 1.6, False)
    filtered = remove_outliers_on_field(filtered, 'rarea', 1.8, False)
    dices = merge_if_multiple_same_detections(filtered)
    dices = remove_outliers_on_field(dices, 'rarea', 1.8, False)
    return dices


def merge_if_multiple_same_detections(candidates):
    # needed 2x loops, because there is possibility that two
    # areas are very close, are connected with another, but in one loop wont connect,
    # for example 3 rectangles a - b - c, common part a, b == 60%, b, c == 60%, a,c == 20 -> 20 is too low
    candidates = concat_if_condition_met(candidates, lambda c1, c2: is_overlaped_by(c1, c2))
    candidates = concat_if_condition_met(candidates, lambda c1, c2: is_overlaped_by(c1, c2))
    return candidates


def is_overlaped_by(c1, c2):
    return has_lower_real_area(c1, c2) and has_common_field(c1, c2, 0.25)


def is_the_same(c1, c2, accept=0):
    return (c1['minx'] - accept <= c2['minx'] <= c1['minx'] + accept and
            c1['miny'] - accept <= c2['miny'] <= c1['miny'] + accept and
            c1['maxx'] - accept <= c2['maxx'] <= c1['maxx'] + accept and
            c1['maxy'] - accept <= c2['maxy'] <= c1['maxy'] + accept)


def remove_with_single_color(candidates, img):
    filtered = [x for x in candidates if is_multi_color(get_img_fragment(x, img))]
    return filtered


def is_multi_color(img):
    img_copy = np.copy(img)
    vals = []
    ar_range = int(255 / 5) + 1
    for x in range(len(img_copy)):
        for y in range(len(img_copy[0])):
            res = discrete_colors(ar_range, img_copy, x, y)
            vals.append(res)
    unique_vals = len(set(vals))
    return not exposure.is_low_contrast(img_copy) and unique_vals > 10


def discrete_colors(range, img, x, y):
    r = discrete_val(img[x][y][0], range)
    g = discrete_val(img[x][y][1], range)
    b = discrete_val(img[x][y][2], range)
    img[x][y] = [r, g, b]
    return b, g, r


def discrete_val(val, range):
    return int(val / range) * range


def concat_if_condition_met(candidates, condition):
    res = []
    for c1 in candidates:
        should_add = True
        for c2 in candidates:
            if condition(c1, c2):
                extend_dice_area(c2, c1)
                should_add = False
        if should_add:
            res.append(c1)
    return res


def has_common_field(f1, f2, field_ratio):
    min_x = max(0, min(f1['maxx'], f2['maxx']) - max(f1['minx'], f2['minx']))
    min_y = max(0, min(f1['maxy'], f2['maxy']) - max(f1['miny'], f2['miny']))
    common_part = min_x * min_y
    minimum_real_area = min(f1['rarea'], f2['rarea'])
    return common_part >= minimum_real_area * field_ratio


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
        cv2.putText(img, str(dots_amount), (dice['minx'], dice['miny']), 2, fontScale=size,
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
        if not is_one_value_image(img):
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
        return filter_dots(valid_regions, dice, dice_img_copy)
    return []


def get_img_fragment(cords, org_img):
    return org_img[cords['miny']:cords['maxy'], cords['minx']:cords['maxx']]


def filter_dots(filtered, dice, img):
    ratio = 1.4
    filtered = get_rectangles_with_dim(filtered, ratio)
    filtered = remove_too_small_and_too_big(filtered, dice)
    filtered = remove_in_corners(filtered, dice)
    filtered = remove_mistaken_dots(filtered, ratio)
    filtered = concat_if_condition_met(filtered, lambda c1, c2: has_lower_real_area(c1, c2) and does_include(c1, c2, 2))
    filtered = merge_if_multiple_same_detections(filtered)
    # [{'param': 'fill', 'ratio': 1.15}, {'param':'rarea', 'ratio': 1.1}, {'param':}]
    filtered = remove_outliers_on_field(filtered, [('fill', 1.15), ('rarea', 1.1), ('area', 1.15)])
    filtered = remove_the_farthest_if_more_than_six(filtered)
    return filtered


def look_for_dots_on_img(filtered, img):
    res = []
    for f in filtered:
        scalar = 0.1
        width_bound = int(f['width'] * scalar)
        height_bound = int(f['height'] * scalar)
        cor = {
            'minx': max(0, f['minx'] - width_bound),
            'maxx': min(len(img[0]) - 1, f['maxx'] + width_bound),
            'miny': max(0, f['miny'] - height_bound),
            'maxy': min(len(img) - 1, f['maxy'] + height_bound),
        }
        min_dim = int(min(cor['maxx'] - cor['minx'], cor['maxy'] - cor['miny']) / 2)
        if min_dim > 0:

            fragment = get_img_fragment(cor, img)
            edges = dots_filter(fragment, {'gamma': 4, 'close': 2, 'median': 2})

            if is_filled_circle(edges/255):
                res.append(f)
    return res


def is_filled_circle(zero_one_img):
    max_x = len(zero_one_img) - 1
    max_y = len(zero_one_img[0]) - 1
    cx = int(max_x / 2)
    cy = int(max_y / 2)
    r = int(min(cx, cy)/3)
    inside = []
    outside = []
    for x in range(0, max_x + 1):
        for y in range(0, max_y + 1):
            current_r = sqrt((cx - x) ** 2 + (cy - y) ** 2)
            if current_r <= r:
                inside.append(zero_one_img[x][y])
            elif current_r >= r*4:
                outside.append(zero_one_img[x][y])

    if len(inside) == 0:
        return False
    inside_color = sum(inside)/len(inside)
    outside_color = 1 - inside_color
    if len(outside) > 0:
        outside_color = sum(outside)/len(outside)

    return ((0.9 <= inside_color and outside_color <= 0.4) or
            (0.9 <= outside_color and inside_color <= 0.4))


def remove_outliers_on_field(filtered, fields, accept_ratio=None, should_sort=True, percent=0.25, allow_more=False):
    if len(filtered) == 0:
        return []

    if type(fields) is str and accept_ratio is not None:
        fields = [(fields, accept_ratio)]
    indexes = []
    for (field_name, ration) in fields:
        field_values = get_by_field_name(filtered, field_name)
        if should_sort:
            field_values.sort(reverse=True)
        comparing_idx = int(len(field_values) * percent)
        to_compare = field_values[comparing_idx]
        dif = to_compare * (ration - 1)
        indexes.extend([i for (i, f) in enumerate(filtered)
                        if f[field_name] >= to_compare - dif and (allow_more or to_compare + dif >= f[field_name])])
    indexes = list(set(indexes))
    return [filtered[i] for i in indexes]


def remove_too_small_and_too_big(filtered, dice):
    return [x for x in filtered if dice['rarea'] / 10 >= x['rarea'] >= dice['rarea'] * 0.005]


def remove_in_corners(filtered, dice):
    res = []
    bound_lim = 0.05
    width_bound = dice['width'] * bound_lim
    height_bound = dice['height'] * bound_lim
    for f in filtered:
        if not ((width_bound > f['minx'] and height_bound > f['miny']) or
                (width_bound > f['minx'] and dice['height'] - height_bound < f['maxy']) or
                (dice['width'] - width_bound < f['maxx'] and height_bound > f['miny']) or
                (dice['width'] - width_bound < f['maxx'] and dice['height'] - height_bound < f['maxy']) or
                f['minx'] <= 1 or f['maxx'] >= dice['width'] - 1 or
                f['miny'] <= 1 or f['maxy'] >= dice['height'] - 1):
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
    for i1, f1 in enumerate(filtered):
        isOk = True
        for i2, f2 in enumerate(filtered[i1:]):
            if has_lower_real_area(f1, f2) and does_include(f1, f2, 2):
                isOk = False
                break
        if isOk:
            res.append(f1)
    return res


def has_lower_real_area(smaller, bigger):
    return not(is_the_same(smaller, bigger)) and is_smaller(smaller, bigger)


def is_smaller(smaller, bigger):
    return smaller['rarea'] <= bigger['rarea']


def does_include(inner, outer, corners):
    return sum([
        outer['miny'] <= inner['miny'] <= outer['maxy'] and outer['minx'] <= inner['minx'] <= outer['maxx'],
        outer['miny'] <= inner['maxy'] <= outer['maxy'] and outer['minx'] <= inner['minx'] <= outer['maxx'],
        outer['miny'] <= inner['miny'] <= outer['maxy'] and outer['minx'] <= inner['maxx'] <= outer['maxx'],
        outer['miny'] <= inner['maxy'] <= outer['maxy'] and outer['minx'] <= inner['maxx'] <= outer['maxx']
    ]) >= corners


def look_for_dices_on_image():
    for i, image in enumerate(dices):
        try:
            parse_image(image)
        except Exception:
            print("error with {0}".format(i))
            traceback.print_exc()
    plt.tight_layout()
    plt.show()
    plt.close()


look_for_dices_on_image()
