from __future__ import division
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

dicesToRead = [
    '01'
   #  , '02', '03','04','05',
   # '06','07','08', '09', '10',
   # '11', '12','13','14','15',
   # '16','17', '18', '19', '20',
   # '21'
]
# dicesToRead = ['08']


def drawDiceImage(i, img):
    plt.subplot(6, 3, i)
    plt.imshow(img)

def drawDiceImageAligned(total, i, img):
    in_row = int(total / 3)
    ax = plt.subplot(in_row, int(total/in_row), i)
    plt.imshow(img)
    return ax


dices = [io.imread('./dices/dice{0}.jpg'.format(i)) for i in dicesToRead]


def getEdges(img, gamma=0.7, sig=3, l=0, u=100):
    img = rgb2gray(img)
    pp, pk = np.percentile(img,(l,u))
    img = exposure.rescale_intensity(img,in_range=(pp,pk))
    from skimage import feature
    img = img ** gamma
    img = ski.feature.canny(img, sigma=sig)
    return img

def get_rectangles_with_dim(rectangles, dim_upper):
    values = [x for x in rectangles if dim_upper >= x["width"] / x["height"] >= 1/dim_upper]
    sort_by_key(values, 'rarea')
    return values


def sort_by_key(values, key):
    values.sort(key=lambda x: x[key], reverse=True)


def parse_image(gamma, img, l, sig, u):
    image = getEdges(img, gamma, sig, l, u)
    regions = find_regions(image)
    # ax.imshow(img)
    values = []
    for region in regions:
        validate_region(region, values, lambda area: area >= 70)
    values = get_rectangles_with_dim(values, 2)

    if len(values) > 0:
        firstOk = values[0]

        filtered = [x for x in values if x['height'] >= firstOk['height'] / 2 and x['width'] >= firstOk['width'] / 2]
        print([x['area'] for x in filtered])
        i = 1
        fig, ax = plt.subplots(figsize=(10, 6))
        for f in filtered:
            find_on_dice(img, f, i, ax)
            i += 1
            # ax.add_patch(f['rect'])  # <- zastąp, zakomentuj poniższe
            # ax.imshow(img[f['miny']:f['maxy'], f['minx']:f['maxx']])
    # return fig


def find_regions(image):
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(2))
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    # label image regions
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=image)
    # fig, ax = plt.subplots(figsize=(10, 6))
    return regionprops(label_image)


def validate_region(region, values, validation_fun):
    if validation_fun(region.area):
        miny, minx, maxy, maxx = region.bbox
        height = maxy - miny
        width = maxx - minx
        rect = mpatches.Rectangle((minx, miny), width, height,
                                  fill=False, edgecolor='blue', linewidth=2)
        values.append({"minx": minx, "miny": miny, "maxx": maxx,
                       "maxy": maxy, "rect": rect, "area": region.area,
                       "width": width, "height": height, 'rarea': width * height})
        #                 ax.add_patch(rect)

# EXPERIMENTAL
# def findCircles(image, edges):
#     # Detect two radii
#     hough_radii = np.arange(2, 15, 2)
#     hough_res = hough_circle(edges, hough_radii)
#
#     accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
#                                                total_num_peaks=6)
#
#     # Draw them
#     # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
#     image = gray2rgb(image)
#     for center_y, center_x, radius in zip(cy, cx, radii):
#         circy, circx = circle_perimeter(center_y, center_x, radius)
#         x = []
#         y = []
#         for i in range(len(circy)):
#             if 0 <= circy[i] < len(image) and 0 <= circx[i] < len(image[0]):
#                 x.append(circx[i])
#                 y.append(circy[i])
#         # if 0 <= circy <= len(image) and 0 <= circx <= len(image[0]):
#         image[y, x] = (220, 20, 20)
#     return image


# def find_on_dice(org_img, dice, i, ax):
#     # fig, ax = plt.subplots(figsize=(10, 6))
#     dice_img_copy = org_img[dice['miny']:dice['maxy'], dice['minx']:dice['maxx']]
#     dice_img = getEdges(dice_img_copy, 1, 1)
#     circles = findCircles(dice_img_copy, dice_img)
#     drawDiceImageAligned(6, i, circles)

def find_on_dice(org_img, dice, i, ax):
    # fig, ax = plt.subplots(figsize=(10, 6))

    dice_img_copy = org_img[dice['miny']:dice['maxy'], dice['minx']:dice['maxx']]
    ax = drawDiceImageAligned(6, i, dice_img_copy)
    dice_img = getEdges(dice_img_copy, 1, 1)
    regions = find_regions(dice_img)
    ax.imshow(dice_img_copy)
    values = []
    for region in regions:
        validate_region(region, values, lambda area: 10 <= area <= 30)
    print(i)
    filtered = get_rectangles_with_dim(values, 1.4)
    center_point = int(len(filtered) / 2)
    center = filtered[center_point]
    filtered = [f for f in filtered if center['rarea'] * 1/1.4 <= f['rarea'] <= center['rarea'] * 1.4]
    for value in filtered:
        print(value['rect'])
        ax.add_patch(value['rect'])





def drawDices(gamma=0.4, sig=2.7, l=91, u=90):  # RECTANGLES
    fig = plt.figure(facecolor="black", figsize=(60, 60))
    i = 1
    for i, image in enumerate(dices):
        try:
            parse_image(gamma, image, l, sig, u)
        except ValueError:
            print("error with {0}".format(i))
    plt.tight_layout()
    plt.show()
    fig.savefig("dices.pdf", facecolor="black")
    plt.close()

# interact(drawDices, gamma=(0.1, 2, 0.1), sig=(0.1, 4, 0.1), l=(0, 100, 1), u=(0, 100, 1))
drawDices()
