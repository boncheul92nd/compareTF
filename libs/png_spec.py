import numpy as np
from PIL import Image

def scale_image(img, base_height, base_width):

    if (base_width == None):
        wpercent = (base_height / float(img.size[1]))
        h_size = int((float(img.size[0]) * float(wpercent)))
        img = img.resize((h_size, base_height), Image.ANTIALIAS)
    else:
        img = img.resize((base_height, base_width), Image.ANTIALIAS)
    return img


def logspec_to_png(out_img, fname, scale_height=None, scale_width=None):

    if scale_height is not None:
        savimg = Image.fromarray(out_img)
        outimg = scale_image(savimg, scale_height, scale_width)

    shift = np.amax(outimg) - np.amin(outimg)
    SC2 = 255 * (outimg - np.amin(outimg)) / shift
    savimg2 = Image.fromarray(np.flipud(SC2))

    pngimg = savimg2.convert('L').transpose(Image.ROTATE_90)
    pngimg = pngimg.transpose(Image.FLIP_LEFT_RIGHT)
    pngimg.save(fname, pnginfo=None)