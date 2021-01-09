import torch
import numpy as np


def raw_collate(batch):
    i1, i2, i3, labels = [], [], [], []

    for im, label in batch:
        img1, img2, img3 = im
        i1.append(torch.from_numpy(img1))
        i2.append(torch.from_numpy(img2))
        i3.append(torch.from_numpy(img3))
        labels.append(label)

    ti1 = torch.stack(i1, dim=0).float()
    ti2 = torch.stack(i2, dim=0).float()
    ti3 = torch.stack(i3, dim=0).float()
    tl = torch.stack(labels, dim=0)

    return (ti1, ti2, ti3), tl


def raw_multi_collate(batch):
    # Only accept single batch
    ims, labels = batch[0]
    im1s, im2s, im3s = ims

    tis = []
    tls = []
    for i1, i2, i3, label in zip(im1s, im2s, im3s, labels):
        ti1 = torch.unsqueeze(torch.from_numpy(i1).float(), dim=0)
        ti2 = torch.unsqueeze(torch.from_numpy(i2).float(), dim=0)
        ti3 = torch.unsqueeze(torch.from_numpy(i3).float(), dim=0)
        tis.append([ti1, ti2, ti3])
        tls.append(torch.from_numpy(np.asarray([label])))

    return tis, tls
def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res 


if __name__ == '__main__':
    _im = (np.random.randn(3, 96, 96), np.random.randn(3, 96, 96), np.random.randn(3, 96, 96))
    _label = np.random.randint(0, 5, size=(1,))
    o = raw_collate([(_im, _label)])
