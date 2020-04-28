import os
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


def make_one_hot(input, num_classes=None):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Shapes:
        predict: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with predict
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    if num_classes is None:
        num_classes = input.max() + 1
    shape = np.array(input.shape)
    shape[-1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(-1, input.cpu().long(), 1)
    return result


class FocalLoss(nn.Module):
    def __init__(self, device, alpha=None, gamma=2, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        if alpha is None:
            self.alpha = [0.10, 0.90] #?
        self.gamma = gamma #2?
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.device = device

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2))
            assert self.alpha.shape[0] == 2, \
                'the `alpha` shape is not match the number of class'
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)
        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward1(self, output, y):
        alpha = 1.0
        # When γ = 0 we obtain BCE.
        if self.ignore_index is not None:
            mask = y != self.ignore_index
            mask.to(self.device)
            y = y * (y != self.ignore_index).long()
            output = output.mul(mask)
        target = make_one_hot(y, num_classes=2).to(self.device)

        output = F.softmax(output, dim=-1) + self.smooth
        # output = output.mul(mask)
        # print(output)
        # first compute binary cross-entropy
        bce = F.binary_cross_entropy(output, target, reduction='mean')
        # print(bce)
        bce_exp = torch.exp(-bce)
        loss = alpha * torch.mul((1.0 - bce_exp).pow(self.gamma), bce)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

    def forward(self, output, y):
        print('Focal Alpha', self.alpha)
        if self.ignore_index is not None:
            mask = y != self.ignore_index
            mask.to(self.device)
            y = y * (y != self.ignore_index).long()
            output = output.mul(mask)
        # print(y.shape)
        target = make_one_hot(y, num_classes=2).to(self.device)
        # print(target.shape)
        probs = F.softmax(output, dim=-1) + self.smooth
        # probs = torch.clamp(probs, self.smooth, 1.0 - self.smooth)

        y0 = target[:,:,0].view(y.shape) # take only one column
        y1 = target[:,:,-1].view(y.shape) # take only one column

        p0 = probs[:,:,0].view(y.shape)
        p1 = probs[:,:,-1].view(y.shape)

        n1 = self.alpha[0] * torch.mul(p1.pow(self.gamma), torch.mul(y0, torch.log(p0)))
        # print(n1)
        # n1 = torch.mul(y0, torch.log(p0))
        # print(n1)
        n2 = self.alpha[1] * torch.mul(p0.pow(self.gamma), torch.mul(y1, torch.log(p1)))
        # n2 = torch.mul(y1, torch.log(p1))
        loss = -(n1 + n2)
        # print(loss)
        # loss = torch.sum(loss, dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class TverskyLoss(nn.Module):
    def __init__(self, device, ignore_index=-100, reduction='mean'):
        super(TverskyLoss, self).__init__()
        self.reduction = reduction
        self.alpha = 0.4
        self.beta = 0.6
        self.ignore_index = ignore_index
        self.device = device
        # if a = b = 0.5, Tversky loss reduce to Dice loss
        # if remove all additional feature from Dice loss -- smoothing, decaying factor, pow**2

    def forward(self, output, y):
        if self.ignore_index is not None:
            mask = y != self.ignore_index
            mask.to(self.device)
            y = y * (y != self.ignore_index).long()
            output = output.mul(mask)
        target = make_one_hot(y, num_classes=2).to(self.device)
        assert output.shape == target.shape
        y0 = target[:,:,0].view(y.shape) # take only one column
        y1 = target[:,:,-1].view(y.shape) # take only one column

        probs = F.softmax(output, dim=-1)
        probs = probs.mul(mask)
        p0 = probs[:,:,0].view(y.shape)
        p1 = probs[:,:,-1].view(y.shape)
        tp = torch.mul(p1, y1)

        fp = self.alpha * torch.mul(p1, y0)
        fn = self.beta * torch.mul(p0, y1)

        num = torch.sum(tp, dim=1)
        d1 = torch.sum(fp, dim=1)
        d2 = torch.sum(fn, dim=1)

        den = num + d1 + d2
        loss = 1 - num/den
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = 1
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, y):
        if self.ignore_index is not None:
            mask = y != self.ignore_index
            #mask.cuda()
            # print(y.shape)
            y = y * (y != self.ignore_index).long()
            # print('output shape', output.shape)
            # print('mask shape', mask.shape)
            output = output.mul(mask)
        target = make_one_hot(y, num_classes=2)#.cuda()
        assert output.shape == target.shape

        target = target[:,:,-1].view(y.shape) # take only one column
        probs = F.softmax(output, dim=-1)
        probs = probs.mul(mask)
        probs = probs[:,:,-1].view(y.shape)

        negp = 1 - probs
        n1 = torch.mul(negp, probs)
        # print(n1)
        # n1 = probs
        n2 = torch.mul(n1, target)
        # print(n2)
        num = 2 * torch.sum(n2, dim=1) + self.smooth
        # print(num)
        # den = torch.sum(output.pow(2) + target.pow(2), dim=1) + self.smooth
        d0 = torch.mul(probs.pow(2), negp.pow(2))
        # d0 = probs
        d1 = torch.sum(d0, dim=1)
        d2 = torch.sum(target.pow(2), dim=1)
        # d2 = torch.sum(target, dim=1)
        den = d1 + d2 + self.smooth
        # print(den)
        loss = 1 - (num/den)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
