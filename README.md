# Bitslice_Sparsity
This repo holds the codes for our preliminary study https://arxiv.org/abs/1909.08496v1 which aims at improving bit-slice sparsity for efficient ReRAM deployment of DNN.

The codes for MNIST and CIFAR-10 are within `mnist/` and `cifar/` respectively. The training routine mainly consists of three parts: pre-training, pruning, and fine-tuning.

First, pre-train a fixed-point model:
```
python3 pretrain.py
```
Then, load and prune the pre-trained model, and fine-tune with either normal l1 regularization, or bit-slice l1 regularization.
```
python3 finetune_l1.py or python3 finetune_bitslice.py
```
There are some arguments within the codes which we have set up default values, but you may want to check it yourself and make some adjustments.

# Acknowledgement
The codes are adapted from [nics_fix_pytorch](https://github.com/walkerning/nics_fix_pytorch).
