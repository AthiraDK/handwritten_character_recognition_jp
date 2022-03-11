import numpy as np
from PIL import Image, ImageEnhance

def binarize(image):
    # Apply otsu threshold to convert a gray scale to a binary image
    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(image)
    binary = image > thresh
    return binary

def resize_img(image, W=64, H=64):
    # This code has been inspired from the original implementation
    image = Image.fromarray(image)
    new_img = Image.new('1', (W, H))
    new_img.paste(image, (0, 0))
    iE = Image.eval(new_img, lambda x: not x)
    iE.thumbnail((W, H))
    ret = np.asarray(iE.getdata()).reshape(W, H)
    return ret

def test_center(image, W=64, H=63, size_add = 12):
    # This code has been inspired from the original implementation
    image = Image.fromarray(image)
    image = image.resize((W + size_add, H + size_add))
    center_img = image.crop((size_add / 2,
                  size_add / 2,
                  W + size_add / 2,
                  H + size_add / 2))
    center_img = np.array(center_img)
    center_img = resize_img(center_img)
    return center_img
