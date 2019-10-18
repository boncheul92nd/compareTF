import numpy as np
from PIL import PngImagePlugin
import json
from PIL import Image

def scale_image(img, base_height, base_width):

    if (base_width == None):
        wpercent = (base_height / float(img.size[1]))
        h_size = int((float(img.size[0]) * float(wpercent)))
        img = img.resize((h_size, base_height), Image.ANTIALIAS)
    else:
        img = img.resize((base_height, base_width), Image.ANTIALIAS)
    return img


def logspec_to_png(out_img, fname, scale_height=None, scale_width=None, lwinfo=None):

    info = PngImagePlugin.PngInfo()
    lwinfo = lwinfo or {}
    lwinfo['oldmin'] = str(np.amin(out_img))
    lwinfo['oldmax'] = str(np.amax(out_img))
    lwinfo['newmin'] = '0'
    lwinfo['newmax'] = '255'
    info.add_text('meta', json.dumps(lwinfo))

    if scale_height is not None:
        savimg = Image.fromarray(out_img).transpose(Image.ROTATE_90)
        outimg = scale_image(savimg, scale_height, scale_width)

    shift = np.amax(outimg) - np.amin(outimg)
    SC2 = 255 * (outimg - np.amin(outimg)) / shift
    savimg2 = Image.fromarray(np.flipud(SC2))

    pngimg = savimg2.convert('L').transpose(Image.ROTATE_90)
    pngimg.save(fname, pnginfo=info)

