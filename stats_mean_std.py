import numpy as np
import os
import glob
import utils


if __name__ == '__main__':
    dataset_root = '../dataset/EuroSAT'
    dataset_cls = os.listdir(dataset_root)  # ['cls1', 'cls2', ...]

    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_imgs = 0, 0, 0
    img_paths = []

    # get all the tif path
    for cls in dataset_cls:
        cls_img_paths = sorted(glob.glob(os.path.join(dataset_root, cls, '*.tif')),
                               key=lambda x: int(os.path.basename(x).split('.')[0].rsplit('_', 1)[-1]))
        img_paths += cls_img_paths

    # statistic the mean and std
    for i, img_path in enumerate(img_paths):
        img = utils.get_rawData(img_path)  # [13, 64, 64] float32 ~ [0, 10000]
        img /= 10000  # [0, 10000] -> [0, 1]
        channels_sum += np.mean(img, axis=(1, 2))  # [13,]
        channels_squared_sum += np.mean(img ** 2, axis=(1, 2))  # [13,]
        num_imgs += 1
        # print
        if (i + 1) % 100 == 0:
            print('[PRINT] %s' % img_path)

    mean = channels_sum / num_imgs
    std = (channels_squared_sum / num_imgs - mean ** 2) ** 0.5

    print('[MEAN]\n', mean, '\n[STD]\n', std)