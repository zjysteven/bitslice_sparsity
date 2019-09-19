# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import yaml
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import nics_fix_pt as nfp
import nics_fix_pt.nn_fix as nnf

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    metavar="N",
    help="number of epochs to train (default: 1)",
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--float",
    action="store_true",
    default=False,
    help="use float point training/testing",
)
parser.add_argument("--save", default=None, help="save fixed-point paramters to file")
parser.add_argument(
    "--alpha", type=float, default=10, help="relative importance for sparsification gradients (default: 0.5)"
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.test_batch_size,
    shuffle=True,
    **kwargs
)


def _generate_default_fix_cfg(names, scale=0, bitwidth=8, method=0):
    return {
        n: {
            "method": torch.autograd.Variable(
                torch.IntTensor(np.array([method])), requires_grad=False
            ),
            "scale": torch.autograd.Variable(
                torch.IntTensor(np.array([scale])), requires_grad=False
            ),
            "bitwidth": torch.autograd.Variable(
                torch.IntTensor(np.array([bitwidth])), requires_grad=False
            ),
        }
        for n in names
    }


BITWIDTH = 8


class Net(nnf.FixTopModule):
    def __init__(self):
        super(Net, self).__init__()
        # initialize some fix configurations
        self.fc1_fix_params = _generate_default_fix_cfg(
            ["weight", "bias"], method=1, bitwidth=BITWIDTH
        )
        self.fc2_fix_params = _generate_default_fix_cfg(
            ["weight", "bias"], method=1, bitwidth=BITWIDTH
        )
        self.fix_params = [
            _generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)
            for _ in range(4)
        ]
        # initialize modules
        self.fc1 = nnf.Linear_fix(784, 100, nf_fix_params=self.fc1_fix_params)
        # self.bn_fc1 = nnf.BatchNorm1d_fix(100, nf_fix_params=self.bn_fc1_params)
        self.fc2 = nnf.Linear_fix(100, 10, nf_fix_params=self.fc2_fix_params)
        self.fix0 = nnf.Activation_fix(nf_fix_params=self.fix_params[0])
        # self.fix0_bn = nnf.Activation_fix(nf_fix_params=self.fix_params[1])
        self.fix1 = nnf.Activation_fix(nf_fix_params=self.fix_params[2])
        self.fix2 = nnf.Activation_fix(nf_fix_params=self.fix_params[3])

    def forward(self, x):
        x = self.fix0(x.view(-1, 784))
        x = F.relu(self.fix1(self.fc1(x)))
        # x = F.relu(self.fix0_bn(self.bn_fc1(self.fix1(self.fc1(x)))))
        self.logits = x = self.fix2(self.fc2(x))
        return F.log_softmax(x, dim=-1)


model = Net()
if args.cuda:
    model.cuda()
#model.print_fix_configs()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# https://discuss.pytorch.org/t/how-to-override-the-gradients-for-parameters/3417/6
class Floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.floor()

    @staticmethod
    def backward(ctx, g):
        return g


# https://discuss.pytorch.org/t/how-to-override-the-gradients-for-parameters/3417/6
class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


def calc_zero_ratio(weights, scale):
    step = 2 ** (scale - 8)
    #x = Round.apply(weights / step) * step

    y = Round.apply(weights.abs() / step).data.cpu().numpy()
    b1 = np.floor(y/64)
    b2 = np.floor((y-b1*64)/16)
    b3 = np.floor((y-b1*64-b2*16)/4)
    b4 = y-b1*64-b2*16-b3*4

    zero_cnt = np.array([np.count_nonzero(b1==0), np.count_nonzero(b2==0), np.count_nonzero(b3==0), np.count_nonzero(b4==0)])
    total_param_cnt = np.array([np.size(b1), np.size(b2), np.size(b3), np.size(b4)])
    return zero_cnt, total_param_cnt

def train(epoch, fix_method=nfp.FIX_AUTO):
    model.set_fix_method(fix_method)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        output = model(data)
        loss_ = F.nll_loss(output, target)

        l1_reg = 0
        zero_cnt = np.zeros((4, ))
        total_param_cnt = np.zeros((4, ))
        cur_fix_cfg = model.get_fix_configs(data_only=True)
        for name, param in model.named_parameters():
            layer, p_name = name.split('.')
            scale = cur_fix_cfg[layer][p_name]['scale']
            zero_cnt_, total_param_cnt_ = calc_zero_ratio(param, scale)
            l1_reg += torch.norm(param, p=1)
            zero_cnt += zero_cnt_
            total_param_cnt += total_param_cnt_

        loss = loss_ + args.alpha * l1_reg / (np.sum(total_param_cnt) / 4)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.0f}%)] NLL_loss: {:.4f}, L1_loss: {:.4f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss_.data.item(),
                    l1_reg.data.item() / (np.sum(total_param_cnt) / 4),
                ),
                end="",
            )
    print("")

def test(fix_method=nfp.FIX_FIXED):
    model.set_fix_method(fix_method)
    model.eval()
    test_nll_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_nll_loss += F.nll_loss(
                output, target, size_average=False
            ).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        l1_reg = 0
        zero_cnt = np.zeros((4, ))
        total_param_cnt = np.zeros((4, ))
        cur_fix_cfg = model.get_fix_configs(data_only=True)
        for name, param in model.named_parameters():
            #print(name.split('.'))
            layer, p_name = name.split('.')
            scale = cur_fix_cfg[layer][p_name]['scale']
            zero_cnt_, total_param_cnt_ = calc_zero_ratio(param, scale)
            l1_reg += torch.norm(param, p=1)
            zero_cnt += zero_cnt_
            total_param_cnt += total_param_cnt_

    test_nll_loss /= len(test_loader.dataset)
    acc = 100 * correct / len(test_loader.dataset)
    print(
        "Test set: Average NLL_loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), L1_loss: {:.4f}, Zero_ratio: {:.2f}%, {:.2f}%, {:.2f}%, {:.2f}%".format(
            test_nll_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
            l1_reg / (np.sum(total_param_cnt) / 4),
            zero_cnt[0] / total_param_cnt[0] * 100.0,
            zero_cnt[1] / total_param_cnt[1] * 100.0,
            zero_cnt[2] / total_param_cnt[2] * 100.0,
            zero_cnt[3] / total_param_cnt[3] * 100.0,
        )
    )
    
    return acc, zero_cnt, total_param_cnt


weight_dir = "pretrain_weights/{}_{}".format(args.alpha, args.lr)
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

print("======================")
print("Start training now....")

#best_acc_and_zero = -233
best_acc = -233
best_zero = -233 * np.ones((4, ))
best_epoch = -233
for epoch in range(1, args.epochs + 1):
    train(epoch, nfp.FIX_NONE if args.float else nfp.FIX_AUTO)
    acc, zero_cnt, total_param_cnt = test(nfp.FIX_NONE if args.float else nfp.FIX_FIXED)

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        best_zero[:] = np.array([zero_cnt[0] / total_param_cnt[0] * 100.0,
                                 zero_cnt[1] / total_param_cnt[1] * 100.0,
                                 zero_cnt[2] / total_param_cnt[2] * 100.0,
                                 zero_cnt[3] / total_param_cnt[3] * 100.0])

        #print("Saving best model...\n")
        for name, module in model._modules.items():
            if hasattr(module, 'weight'):
                temp_w = module.weight.data.cpu().numpy()
                np.save(weight_dir+"/{}_weight.npy".format(name), temp_w)
            if hasattr(module, 'bias'):
                temp_b = module.bias.data.cpu().numpy()
                np.save(weight_dir+"/{}_bias.npy".format(name), temp_b)
        fix_cfg = {
            "data": model.get_fix_configs(data_only=True),
            "grad": model.get_fix_configs(grad=True, data_only=True),
        }
        with open(weight_dir+"/fix_config.yaml", "w") as wf:
            yaml.dump(fix_cfg, wf, default_flow_style=False)
    
    print("Best Accuracy: {:.2f}%, Best Zero Ratio: {:.2f}%, {:.2f}%, {:.2f}%, {:.2f}% @ Epoch [{:d}]\n".format(best_acc, best_zero[0], best_zero[1], best_zero[2], best_zero[3], best_epoch))
        

print("Finish training.......")
print("======================\n")