import numpy as np
from PIL import PngImagePlugin
import json
from PIL import Image

def scale_image(img, base_width, base_height):

    if (base_width == None):
        w_percent = (base_height / float(img.size[1]))
        h_size = int((float(img.size[0]) * float(w_percent)))
        img = img.resize((h_size, base_height), Image.ANTIALIAS)
    else:
        img = img.resize((base_width, base_height), Image.ANTIALIAS)

    return img


def log_spec_to_png(out_img, fname, scale_width=None, scale_height=None, lwinfo=None):

    info = PngImagePlugin.PngInfo()
    lwinfo = lwinfo or {}
    lwinfo['old_min'] = str(np.amin(out_img))
    lwinfo['old_max'] = str(np.amax(out_img))
    lwinfo['new_min'] = '0'
    lwinfo['new_max'] = '255'
    info.add_text('meta', json.dumps(lwinfo))

    if scale_height is not None:
        original_img = Image.fromarray(out_img)
        out_img = scale_image(original_img, scale_width, scale_height)

    shift = np.amax(out_img) - np.amin(out_img)
    sc2 = 255 * (out_img - np.amin(out_img)) / shift
    sav_img2 = Image.fromarray(np.flipud(sc2))

