import torch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def create_figure(data, figsize):
    f = plt.figure(figsize=figsize)
    fig_len = len(data)
    for i, d in enumerate(data, 1):
        f.add_subplot(1, fig_len, i)
        plt.imshow(d, cmap=plt.cm.gray)
    f.tight_layout()
    return f


def log_images(image_batches, path, run_id, step, context, figsize):
    step = str(step)
    curr_dir = os.path.join(path, run_id)
    if not os.path.isdir(curr_dir):
        os.mkdir(curr_dir)
    if not os.path.isdir(os.path.join(curr_dir, context)):
        os.mkdir(os.path.join(curr_dir, context))
    if not os.path.isdir(os.path.join(curr_dir, context, step)):
        os.mkdir(os.path.join(curr_dir, context, step))

    image_list = []
    for images in image_batches:
        image_list.append(images)

    for i in range(image_batches[0].shape[0]):
        for channel in range(image_batches[0].shape[1]):
            create_figure([image[i, channel, :, :] for image in image_list], figsize)
            plt.savefig(os.path.join(curr_dir, context, step, str(i) + " " + str(channel)), format="png")


def log_heatmap(image_batches_A, image_batches_B, path, run_id, step, context, figsize):
    step = str(step)
    curr_dir = os.path.join(path, run_id)
    if not os.path.isdir(curr_dir):
        os.mkdir(curr_dir)
    if not os.path.isdir(os.path.join(curr_dir, context)):
        os.mkdir(os.path.join(curr_dir, context))
    if not os.path.isdir(os.path.join(curr_dir, context, step)):
        os.mkdir(os.path.join(curr_dir, context, step))

    for i, (image_A, image_B) in enumerate(zip(image_batches_A, image_batches_B)):
        f = plt.figure(figsize=figsize)
        f.add_subplot(1, 3, 1)
        plt.imshow(image_A[0, :, :], cmap=plt.cm.gray)
        f.add_subplot(1, 3, 2)
        plt.imshow(image_B[0, :, :], cmap=plt.cm.gray)
        ax = f.add_subplot(1, 3, 3)
        abs_img = np.abs(image_A - image_B)
        abs_img = abs_img[0, :, :]
        im = plt.imshow(abs_img, cmap='bwr')
        cax = f.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        plt.colorbar(im, cax=cax)
        plt.savefig(os.path.join(curr_dir, context, step, str(i)), format="png")


def scale(image, interval_a, interval_b, mask, mask_val):
    return ((interval_b - interval_a) * ((image - np.amin(a=image, where=mask!=mask_val, initial=10000)) / (np.amax(a=image, where=mask!=mask_val, initial=-10000) - np.amin(a=image, where=mask!=mask_val, initial=10000)))) + interval_a
#toto treba prerobit, pretoze, teraz najvyssiu hodnotu da na 100(napr 0.95), ale ja  chcem aby to hodnotu 1 dalo na 100

def mae(image_a, image_b, mask, mask_val):
    diff = np.abs(image_a - image_b)
    _sum = np.sum(diff[mask != mask_val])
    n = len(mask[mask != mask_val])
    return _sum / n


def log_data(data, path, run_id, step, context):
    step = str(step)
    curr_dir = os.path.join(path, run_id)
    if not os.path.isdir(curr_dir):
        os.mkdir(curr_dir)
    if not os.path.isdir(os.path.join(curr_dir, context)):
        os.mkdir(os.path.join(curr_dir, context))
    if not os.path.isdir(os.path.join(curr_dir, context, step)):
        os.mkdir(os.path.join(curr_dir, context, step))

    with open(os.path.join(curr_dir, context, step, "slices.pkl"), "wb") as handle:
        pickle.dump(data, handle)


class Buffer():
    def __init__(self, max_len):
        assert max_len > 0
        self.max_len = max_len
        self.data = []

    def push_and_pop(self, data):
        if len(self.data) < self.max_len:
            self.data.append(data)
            return data
        else:
            to_return = self.data.pop(0)
            self.data.append(data)
            return to_return
