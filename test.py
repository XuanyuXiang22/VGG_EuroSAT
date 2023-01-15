import torch
from vgg import VGG16
from dataset import get_dataloader
from utils import load_network, get_acc


if __name__ == "__main__":
    train_dataroot = '../data/data_xuanyu/EuroSAT_train_test/train'
    test_dataroot = '../data/data_xuanyu/EuroSAT_train_test/test'
    checkpoint_dir = '.'
    torch.cuda.set_device('cuda:0')
    batch_size = 32

    train_loader = get_dataloader(train_dataroot, batch_size, isTrain=False)
    test_loader = get_dataloader(test_dataroot, batch_size, isTrain=False)
    net = VGG16().cuda()
    load_network(net, checkpoint_dir)

    train_acc = get_acc(net, train_loader, False)
    test_acc = get_acc(net, test_loader, False)

    print('train acc:', train_acc)
    print('test acc:', test_acc)