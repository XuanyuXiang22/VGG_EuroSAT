import torch
import wandb
import os
import utils
from dataset import get_dataloader
from vgg import VGG16


if __name__ == "__main__":
    ############ hyperparameters ############
    train_dataroot = '../data/data_xuanyu/EuroSAT_train_test/train'
    test_dataroot = '../data/data_xuanyu/EuroSAT_train_test/test'
    checkpoint_dir = '../model/model_xuanyu/VGG_EuroSAT'

    # dataset
    torch.cuda.set_device('cuda:0')
    RandomResizedCrop_p = 0.5  # the probability of applying scale&crop data augmentation
    RandomResizedCrop_scale = (0.5, 1.0)  # scale&crop data augmentation param

    batch_size = 32
    lr = 1e-2
    epochs = 50

    wandb.init(
        project='VGG_EuroSAT',
        name='v01',
        config={
            'RandomResizedCrop_p': 0.5,
            'RandomResizedCrop_scale': (0.5, 1.0),
            'batch_size': 32,
            'lr': 1e-2,
            'epochs': 50
        },
        dir=os.path.join(checkpoint_dir, 'wandb')
    )
    ########################################

    train_loader = get_dataloader(train_dataroot, batch_size, True, RandomResizedCrop_p, RandomResizedCrop_scale)
    test_loader = get_dataloader(test_dataroot, batch_size, False, RandomResizedCrop_p, RandomResizedCrop_scale)
    net = VGG16().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    total_steps = 0
    best_acc = 0
    for epoch in range(1, epochs + 1):
        epoch_step = 0

        for x, y in train_loader:
            total_steps += 1
            epoch_step += 1
            x = x.cuda()
            y = y.cuda()
            # x: [N, 3, 64, 64] float32 [-1, 1]
            # y: [N, 10] int64
            y_hat = net(x)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_steps % 100 == 0:
                wandb.log({'loss': loss})
                print('[LOSS] epoch: %d step: %d - loss: %.3f' % (epoch, epoch_step, loss))

        # eval the acc
        acc = utils.get_acc(net, test_loader, True)
        if acc > best_acc:
            best_acc = acc
            utils.save(net, checkpoint_dir)
        wandb.log({'acc': acc})
        print('[ACC] epoch: %d - acc: %.3f, best acc: %.3f' % (epoch, acc, best_acc))

        # update the lr
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(acc)
        new_lr = optimizer.param_groups[0]['lr']
        print('[LR] optimizer lr: %.6f -> %.6f' % (old_lr, new_lr))

    wandb.finish()