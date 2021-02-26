import torch
import matplotlib.pyplot as plt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def denormalize(image):
    return ((image + 1) / 2) * 255


def create_figure(data, figsize):
    f = plt.figure(figsize=figsize)
    fig_len = len(data)
    for i, d in enumerate(data, 1):
        f.add_subplot(1, fig_len, i)
        plt.imshow(d, cmap=plt.cm.gray)
    f.tight_layout()
    return f


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
