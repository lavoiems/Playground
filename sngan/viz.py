'''Visualization.

'''

import imageio
import scipy
import matplotlib
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import visdom
from sklearn.manifold import TSNE
from tile_images import tile_raster_images

visualizer = None
matplotlib.use('Agg')

_options = dict(
    use_tanh=False,
    quantized=False,
    img=None,
    label_names=None,
    is_caption=False,
    is_attribute=False
)


CHAR_MAP = ['_', '\n', ' ', '!', '"', '%',  '&', "'", '(', ')', ',', '-', '.', '/',
            '0', '1', '2', '3', '4', '5', '8', '9', ':', ';', '=', '?', '\\', '`',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '*', '*', '*']


def setup(server, port, use_tanh=None, quantized=None, img=None, label_names=None,
          is_caption=False, is_attribute=False, env='main'):
    global visualizer
    visualizer = visdom.Visdom(server=server, port=port, env=env)

    global _options
    if use_tanh is not None:
        _options['use_tanh'] = use_tanh
    if quantized is not None:
        _options['quantized'] = quantized
    if img is not None:
        _options['img'] = img
    if label_names is not None:
        _options['label_names'] = label_names
    _options['is_caption'] = is_caption
    _options['is_attribute'] = is_attribute
    if is_caption and is_attribute:
        raise ValueError('Cannot be both attribute and caption')


def dequantize(images):
    images = np.argmax(images, axis=1).astype('uint8')
    images_ = []
    for image in images:
        img2 = Image.fromarray(image)
        img2.putpalette(_options['img'].getpalette())
        img2 = img2.convert('RGB')
        images_.append(np.array(img2))
    images = np.array(images_).transpose(0, 3, 1, 2).astype(floatX) / 255.
    return images


def save_images(images, num_x, num_y, out_file=None, labels=None,
                margin_x=5, margin_y=5, image_id=0, caption='', title=''):
    if labels is not None:
        if _options['is_caption']:
            margin_x = 80
            margin_y = 50
        elif _options['is_attribute']:
            margin_x = 25
            margin_y = 200
        elif _options['label_names'] is not None:
            margin_x = 20
            margin_y = 25
        else:
            margin_x = 5
            margin_y = 12

    if out_file is None:
        pass
    else:

        if _options['quantized']:
            images = dequantize(images)
        elif _options['use_tanh']:
            images = 0.5 * (images + 1.)

        images = images * 255.

        dim_c, dim_x, dim_y = images.shape[-3:]
        if dim_c == 1:
            arr = tile_raster_images(
                X=images, img_shape=(dim_x, dim_y), tile_shape=(num_x, num_y),
                tile_spacing=(margin_y, margin_x), bottom_margin=margin_y)
            fill = 255
        else:
            arrs = []
            for c in xrange(dim_c):
                arr = tile_raster_images(
                    X=images[:, c].copy(), img_shape=(dim_x, dim_y),
                    tile_shape=(num_x, num_y),
                    tile_spacing=(margin_y, margin_x),
                    bottom_margin=margin_y)
                arrs.append(arr)
            arr = np.array(arrs).transpose(1, 2, 0)
            fill = (255, 255, 255)
        im = Image.fromarray(arr)
        if labels is not None:
            try:
                font = ImageFont.truetype(
                    '/usr/share/fonts/truetype/freefont/FreeSans.ttf', 9)
            except:
                font = ImageFont.truetype(
                    '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf', 9)

            idr = ImageDraw.Draw(im)
            for i, label in enumerate(labels):
                x_ = (i % num_x) * (dim_x + margin_x)
                y_ = (i // num_x) * (dim_y + margin_y) + dim_y
                if _options['is_caption']:
                    l_ = ''.join([CHAR_MAP[j] for j in label])
                    if len(l_) > 20:
                        l_ = '\n'.join(
                            [l_[x:x+20] for x in range(0, len(l_), 20)])
                elif _options['is_attribute']:
                    attribs = [j for j, a in enumerate(label) if a == 1]
                    l_ = '\n'.join(_options['label_names'][a] for a in attribs)
                elif _options['label_names'] is not None:
                    l_ = _options['label_names'][label]
                    l_ = l_.replace('_', '\n')
                else:
                    l_ = str(label)
                idr.text((x_, y_), l_, fill=fill, font=font)
        arr = np.array(im)
        if arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)
        visualizer.image(arr, opts=dict(title=title, caption=caption),
                         win='image_{}'.format(image_id))
        im.save(out_file)


def save_movie(images, num_x, num_y, env='main', out_file=None, movie_id=0):
    if out_file is None:
        pass
    else:
        images_ = []
        for i, image in enumerate(images):
            if _options['quantized']:
                image = dequantize(image)
            dim_c, dim_x, dim_y = image.shape[-3:]
            image = image.reshape((num_x, num_y, dim_c, dim_x, dim_y))
            image = image.transpose(0, 3, 1, 4, 2)
            image = image.reshape(num_x * dim_x, num_y * dim_y, dim_c)
            if _options['use_tanh']:
                image = 0.5 * (image + 1.)
            images_.append(image)
        imageio.mimsave(out_file, images_)
    visualizer.video(videofile=out_file, env=env,
                     win='movie_{}'.format(movie_id))


def save_hist(fake_scores, real_scores, out_file, env='main'):
    bins = np.linspace(np.min(np.array([fake_scores, real_scores])),
                       np.max(np.array([fake_scores, real_scores])), 100)
    plt.clf()
    plt.hist(fake_scores, bins, alpha=0.5, label='generated samples')
    plt.hist(real_scores, bins, alpha=0.5, label='real samples')
    plt.legend(loc='upper right')
    plt.savefig(out_file)
    hist_real = np.histogram(real_scores, bins=bins)[0]
    hist_fake = np.histogram(fake_scores, bins=bins)[0]
    X = np.column_stack((hist_real, hist_fake))
    visualizer.stem(
        X=X, Y=np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(99)]),
        opts=dict(legend=['Real', 'Fake']), win='hist', env=env)


def save_tsne(X, labels, colormap, out_file, id=0, labels_name=None, title=''):
    plt.clf()
    if len(labels):
        color = [colormap[label] for label in labels]
        plt.scatter(*X, c=color)
        patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colormap, range(10))]
        plt.legend(handles=patches)
        visualizer.scatter(
            X.transpose(1,0), win='scatter%s' % id, Y=np.array(labels)+1,
            opts={'legend': labels_name, 'title': title})
    else:
        plt.scatter(*X)
        visualizer.scatter(
            X.transpose(1,0), win='scatter%s' % id, opts={'title': title})
    plt.savefig(out_file)


def save_train_error(errors, out_file, env='main', id=0, title=''):
    plt.clf()
    plt.plot(errors)
    plt.title(title)
    plt.savefig(out_file)
    visualizer.line(errors, win='line%s' % id, env=env, opts={'title': title})
