import os
import rasterio
import numpy as np
import torch
from shutil import copyfile
from torchvision import transforms


def make_paths(dataroot):
    paths = []

    for root, dirs, files in os.walk(dataroot):
        if all(file.endswith(".tif") for file in files) and len(files) != 0:
            # sort the files
            files = sorted(files, key=lambda x: int(os.path.splitext(x)[0].rsplit('_', 1)[-1]))

            for file in files:
                path = os.path.join(root, file)
                paths.append(path)

    return paths


def get_pipeline(RandomResizedCrop_p, RandomResizedCrop_scale, is_train):
    pipeline = [transforms.ToTensor()]  # [13, 64, 64] float32 [0, 10000]

    if is_train:
        pipeline.append(
            transforms.RandomApply([
                transforms.RandomResizedCrop((64, 64), scale=RandomResizedCrop_scale, ratio=(1, 1))
            ], p=RandomResizedCrop_p)
        )  # [13, 64, 64] float32 [0, 10000]

    pipeline += [
        transforms.Lambda(lambda x: x.clamp_(0, 10000) / 10000),
        transforms.Normalize(
            mean=(0.13537274, 0.11172014, 0.10418849, 0.0946558, 0.11991832, 0.20030074, 0.23740004, 0.23012331,
                  0.07321849, 0.00120995, 0.18206967, 0.11182084, 0.2599792),
            std=(0.02452919, 0.03334453, 0.03952134, 0.05944701, 0.05670202, 0.08610252, 0.10869984, 0.1118287,
                 0.04038343, 0.00047294, 0.10025451, 0.07606022, 0.12317288)
        )
    ]
    pipeline = transforms.Compose(pipeline)
    return pipeline


def get_rawData(data_path):
    src = rasterio.open(data_path, 'r', driver='GTiff')
    image = src.read()
    src.close()
    image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts

    return image.astype('float32')


def split_train_test(source, target, test_ratio=10):
    source_clses = os.listdir(source)  # ['cls1', 'cls2', ...]

    for cls in source_clses:
        source_cls = os.path.join(source, cls)  # source/AnnualCrop
        target_train_cls = os.path.join(target, 'train', cls)  # target/train/AnnualCrop
        target_test_cls = os.path.join(target, 'test', cls)  # target/test/AnnualCrop
        os.makedirs(target_train_cls)
        os.makedirs(target_test_cls)

        # copy the tif
        img_names = sorted(os.listdir(source_cls),
                           key= lambda x: int(os.path.splitext(x)[0].rsplit('_', 1)[-1]))  # ['AnnualCrop_1.tif', ...]
        for i, img_name in enumerate(img_names):
            source_path = os.path.join(source_cls, img_name)  # source/AnnualCrop/AnnualCrop_1.tif
            # train: test = 9: 1
            if (i + 1) % test_ratio == 0:  # test
                target_path = os.path.join(target_test_cls, img_name)  # target/test/AnnualCrop/AnnualCrop_1.tif
            else:  # train
                target_path = os.path.join(target_train_cls, img_name)  # target/train/AnnualCrop/AnnualCrop_1.tif
            copyfile(source_path, target_path)

        # print the info
        print('[TRAIN / TEST] %s done!' % cls)


def get_acc(net, test_loader, is_train):
    n_total = 0
    n_correct = 0
    net.eval()

    with torch.no_grad():
        for img, label in test_loader:
            img = img.cuda()
            label = label.cuda()

            cls = net(img)
            _, cls_idx = torch.max(cls, dim=1)

            n_total += img.size()[0]
            n_correct += (label == cls_idx).sum().item()
        acc = n_correct / n_total

    if is_train:
        net.train()

    return acc


def save(net, checkpoint_dir):
    save_filename = os.path.join(checkpoint_dir, 'latest_net_VGG16.pth')
    torch.save(net.state_dict(), save_filename)
    print('Save', save_filename, 'done!')


def load_network(net, checkpoint_dir):
    device = next(net.parameters()).device
    filename = os.path.join(checkpoint_dir, 'latest_net_VGG16.pth')
    weights = torch.load(filename, map_location=device)
    net.load_state_dict(weights)
    print('Load network from', filename)