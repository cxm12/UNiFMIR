import numpy as np


class DataLoader():
    def __init__(self):
        a = None

    def load_data(self, imgs_lr, imgs_hr, start, len, batch_size=1):
        start = np.random.randint(start*batch_size, len-batch_size-1)
        return imgs_hr[start:start+batch_size], imgs_lr[start:start+batch_size]
