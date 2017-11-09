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
    '01', '02', '03','04','05',
   '06','07','08', '09', '10',
   '11', '12','13','14','15',
   '16','17', '18', '19', '20',
   '21'
]
# dicesToRead = ['08']


def drawDiceImage(i, img):
    plt.subplot(6,3,i)
    plt.imshow(img)


dices = [io.imread('./dices/dice{0}.jpg'.format(i)) for i in dicesToRead]


def getEdges(img, gamma = 0.7, sig=3, l=0, u=100):
    img = rgb2gray(img)
    pp, pk = np.percentile(img,(l,u))
    img = exposure.rescale_intensity(img,in_range=(pp,pk))
    from skimage import feature
    img = img ** gamma
    img = ski.feature.canny(img, sigma=sig)
    return img


def parse_image(gamma, img, l, sig, u):
    image = getEdges(img, gamma, sig, l, u)
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(2))
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    # label image regions
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=image)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    values = []
    for region in regionprops(label_image):
        if region.area >= 70:
            miny, minx, maxy, maxx = region.bbox
            height = maxy - miny
            width = maxx - minx
            rect = mpatches.Rectangle((minx, miny), width, height,
                                      fill=False, edgecolor='blue', linewidth=2)
            values.append({"minx": minx, "miny": miny, "maxx": maxx,
                           "maxy": maxy, "rect": rect, "area": region.area,
                           "width": width, "height": height})
            #                 ax.add_patch(rect)
    values = [x for x in values if 2 >= x["width"] / x["height"] >= 0.5]
    values.sort(key=lambda x: x['area'], reverse=True)
    if len(values) > 0:
        firstOk = values[0]
        i = 1
        filtered = [x for x in values if x['height'] >= firstOk['height'] / 2 and x['width'] >= firstOk['width'] / 2]
        print([x['area'] for x in filtered])
        for f in filtered:
            ax.add_patch(f['rect'])  # <- zastąp, zakomentuj poniższe
            # ax.imshow(img[f['miny']:f['maxy'], f['minx']:f['maxx']])
            i += 1
    return fig


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
    # fig.savefig("dices.pdf", facecolor="black")
    plt.close()

# interact(drawDices, gamma=(0.1, 2, 0.1), sig=(0.1, 4, 0.1), l=(0, 100, 1), u=(0, 100, 1))
drawDices()
