import numpy as np
from PIL import Image, ImageEnhance

def binarize_img(image):
    # Apply otsu threshold to convert a gray scale to a binary image
    from skimage.filters import threshold_otsu
    if isinstance(image, Image.Image):
        image = np.array(image)
    thresh = threshold_otsu(image)
    binary = image > thresh
    return binary

def resize_img(image, W=64, H=64):
    # This code has been inspired from the original implementation
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    new_img = Image.new('1', (W, H))
    new_img.paste(image, (0, 0))
    iE = new_img
    # iE = Image.eval(new_img, lambda x: not x) # inverts the image
    iE.thumbnail((W, H))
    ret = np.asarray(iE.getdata()).reshape(W, H)
    return ret

def center_img(image, W=64, H=64, size_add=12):
    # This code has been inspired from the original implementation
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.resize((W + size_add, H + size_add))
    center_img = image.crop((size_add / 2,
                  size_add / 2,
                  W + size_add / 2,
                  H + size_add / 2))
    center_img = np.array(center_img)
    # center_img = resize_img(center_img, W, H)
    return center_img
