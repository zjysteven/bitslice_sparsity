from __future__ import print_function
import argparse
import os
import shutil
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import ResNet
import vgg
from util_func import Floor, Round

import nics_fix_pt as nfp
import nics_fix_pt.nn_fix as nnf

model_names = sorted(
    name
    for name in vgg.__dict__
    if name.islower()
    and not name.startswith("__")
    and name.startswith("vgg")
    and callable(vgg.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Example")
parser.add_argument(
    "--arch",
    "-a",
    metavar="ARCH",
    default="vgg11_ugly",
    #choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: vgg19)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epoches",
    default=150,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 128)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.05,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "--momentum", 
    default=0.9, 
    type=float, 
    metavar="M", 
    help="momentum"
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=40,
    type=int,
    metavar="N",
    help="print frequency (default: 40)",
)
parser.add_argument(
    "--prefix",
    default=None,
    type=str,
    metavar="PREFIX",
    help="checkpoint prefix (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    default=True,
    help="evaluate model on validation set",
)
parser.add_argument(
    "--half", dest="half", action="store_true", help="use half-precision(16-bit) "
)
parser.add_argument(
    "--save-dir",
    dest="save_dir",
    help="The directory used to save the trained models",
    default="save",
    type=str,
)
parser.add_argument(
    "--alpha",
    type=float,
    default=20,
)
parser.add_argument(
    "--th", 
    type=float, 
    default=0.01, 
    help="threshold for prunning"
)
parser.add_argument(
    "--seed", 
    type=int, 
    default=1, 
    metavar="S", 
    help="random seed (default: 1)"
)

best_all = -233
best_prec1 = -233
best_zero_ratio = -233 * np.ones((4,))
best_epoch = -233
start = time.time()

def main():
    global args, best_prec1, best_epoch, best_all, best_zero_ratio
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.arch == "resnet":
        model = ResNet.resnet(depth=20)
        args.prefix = "resnet"
    elif "vgg" in args.arch:
        model = vgg.__dict__[args.arch]()
        args.prefix = "vgg"
    #for module in model.modules():
    #    print(module)
    #model.print_fix_configs()

    filename = os.path.join(args.save_dir, 
        "checkpoint_{}.tar".format(args.prefix))
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    # model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    fix_cfg = checkpoint["fix_cfg"]
    model.load_fix_configs(fix_cfg["data"])
    model.load_fix_configs(fix_cfg["grad"], grad=True)

    # Prune
    masks = {}
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        threshold = args.th #np.std(tensor) * args.sensitivity
        #print(f'Pruning with threshold : {threshold} for layer {name}')
        new_mask = np.where(abs(tensor) < threshold, 0, tensor)
        p.data = torch.from_numpy(new_mask).cuda()
        mask = np.where(abs(tensor) < threshold, 0., 1.)
        masks[name] = torch.from_numpy(mask).float().cuda()
    print("--- Finished Pruning ---")

    cudnn.benchmark = True

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="../data/cifar10",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="../data/cifar10",
            train=False,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        ),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
    )

    if args.evaluate:
        prec_prune, z1_prune, z2_prune, z3_prune, z4_prune = validate(val_loader, model, criterion)
        zero_prune = np.array([z1_prune, z2_prune, z3_prune, z4_prune])
        #validate(val_loader, model, criterion)
        print("--- Initial Validation After Pruning ---")
        #return

    train_curve = np.zeros((args.epoches, 5))
    for epoch in range(args.start_epoch, args.epoches):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, masks)

        # evaluate on validation set
        prec1, z1, z2, z3, z4 = validate(val_loader, model, criterion)
        train_curve[epoch, :] = np.array([prec1, z1, z2, z3, z4])

        # remember best prec@1 & average zero ratio and save checkpoint
        is_best = prec1 + (z1+z2+z3+z4)/4 > best_all
        if is_best:
            best_all = prec1 + (z1+z2+z3+z4)/4
            best_zero_ratio[:] = np.array([z1, z2, z3, z4])
            best_prec1 = prec1
            best_epoch = epoch
            
            fix_cfg = {
                "data": model.get_fix_configs(data_only=True),
                "grad": model.get_fix_configs(grad=True, data_only=True),
            }
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                    "best_zero_ratio": best_zero_ratio,
                    "prune_prec1": prec_prune,
                    "prune_zero_ratio": zero_prune,
                    "fix_cfg": fix_cfg
                },
                is_best,
                filename=os.path.join(
                    args.save_dir,
                    "finetune_l1_{}_{}.tar".format(args.arch, args.th),
                ),
            )

        print(
            " * Best Prec: {:.2f}%, Zero Ratio: {:.2f}%, {:.2f}%, {:.2f}%, {:.2f}% @ Epoch [{:d}]\n".format(
                best_prec1, best_zero_ratio[0], best_zero_ratio[1], best_zero_ratio[2], best_zero_ratio[3], best_epoch
            )
        )

        #np.save("l1_curve_{}_{}.npy".format(args.arch, args.th), train_curve)


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


def train(train_loader, model, criterion, optimizer, epoch, masks):
    """
        Run one train epoch
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.set_fix_method(nfp.FIX_AUTO)
    model.train()

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss_ = criterion(output, target_var)

        l1_reg = 0
        total_param_cnt = 0
        #cur_fix_cfg = model.get_fix_configs(data_only=True)
        for _, param in model.named_parameters():
            l1_reg += torch.norm(param, p=1)
            total_param_cnt += np.size(param.data.cpu().numpy())
            
        loss = loss_ + args.alpha * l1_reg / total_param_cnt

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # set gradients of pruned weights to 0
        for name, p in model.named_parameters():
            p.grad.data = p.grad.data*masks[name]

        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss_.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print(
                "\rEpoch: [{0}]\tLoss: {loss.avg:.4f}\tL1: {l1:.4f}\tPrec: {top1.avg:.2f}%".format(
                    epoch,
                    loss=losses,
                    l1=l1_reg.item()/total_param_cnt,
                    top1=top1,
                ),
                end="",
            )


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.set_fix_method(nfp.FIX_FIXED)
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target)

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
        
        l1_reg = 0
        zero_cnt = np.zeros((4, ))
        total_param_cnt = np.zeros((4, ))
        cur_fix_cfg = model.get_fix_configs(data_only=True)
        for name, param in model.named_parameters():
            # Using quantization module for batch norm will yield weird behavior, 
            # which we suspect originates from the fixed-point training codes
            # So we don't quantize the bn module
            if "bn" in name or "downsample.1" in name:
                continue

            length = len(name.split('.'))
            temp = cur_fix_cfg
            for i in range(length):
                temp = temp[name.split('.')[i]]
            scale = temp['scale']

            zero_cnt_, total_param_cnt_ = calc_zero_ratio(param, scale)
            l1_reg += torch.norm(param, p=1)
            zero_cnt += zero_cnt_
            total_param_cnt += total_param_cnt_

        z1 = zero_cnt[0] / total_param_cnt[0] * 100.0
        z2 = zero_cnt[1] / total_param_cnt[1] * 100.0
        z3 = zero_cnt[2] / total_param_cnt[2] * 100.0
        z4 = zero_cnt[3] / total_param_cnt[3] * 100.0

        print(
            "\nTest: [{0}/{1}]\t"
            "Loss: {loss.avg:.4f}\t"
            "L1: {l1:.4f}\t"
            "Prec: {top1.avg:.2f}%\t"
            "Zero_ratio: {a:.2f}%, {b:.2f}%, {c:.2f}%, {d:.2f}%".format(
                i, len(val_loader), 
                loss=losses, 
                l1=l1_reg.item()/(np.sum(total_param_cnt)/4),
                top1=top1,
                a=z1,
                b=z2,
                c=z3,
                d=z4,
            )
        )

    return top1.avg, z1, z2, z3, z4


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    main()
