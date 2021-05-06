# -*- coding:utf-8 -*-
# Created Time: Thu 05 Jul 2018 10:00:41 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>
from config import cfg

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
import numpy as np
import argparse
import time
from functools import reduce
from operator import mul
from vgg import VGG

device = 'cuda' if torch.cuda.is_available() else 'cpu'
layer_counter = 0
layer_activations = {}
total_values_dict = {}
zero_values_dict = {}
batch_size = 32
input_shape = (3, 32, 32)


def activations(self, input, output):
    global layer_activations, layer_counter, total_values_dict, zero_values_dict
    layer_counter += 1
    layer_activations[layer_counter] = output

    if 'ReLU' in self.__class__.__name__:
        o_shape = output.shape
        total_values = reduce(mul, o_shape[1:], 1)
        zero_values = torch.sum(output == 0, dim=[i for i in range(1, len(o_shape))]).to(dtype=torch.float)
        total_values_dict[layer_counter] = total_values
        zero_values_dict[layer_counter] = zero_values


class Program(nn.Module):
    def __init__(self, model_path, program_path):
        super(Program, self).__init__()
        self.init_net(model_path)
        self.W = Parameter(torch.zeros(input_shape, device=device), requires_grad=False)
        if program_path is not None:
            self.W.data = torch.load(program_path, map_location=torch.device(device))['W']
        hooks = {}
        for name, module in self.net.named_modules():
            module.module_name = name
            hooks[name] = module.register_forward_hook(activations)

    # load pre-trained model from checkpoint
    def init_net(self, model_path):
        self.net = VGG('VGG16')

        self.net = self.net.to(device)

        # load pre-trained weights
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        self.net.load_state_dict(checkpoint['net'])
        # mean and std for input
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        mean = mean[..., np.newaxis, np.newaxis]
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
        std = std[..., np.newaxis, np.newaxis]
        self.mean = Parameter(torch.from_numpy(mean).float().to(device), requires_grad=False)
        self.std = Parameter(torch.from_numpy(std).float().to(device), requires_grad=False)

        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

    def imagenet_label2_mnist_label(self, imagenet_label):
        return imagenet_label[:, :10]

    # forward pass of image perturbed with the program
    # returns the prediction along with (all intermediate activations, density of post RELU activations)
    def forward(self, image):
        global layer_counter, layer_activations, total_values_dict, zero_values_dict
        X = image.data.clone()
        X = Variable(X, requires_grad=False)
        P = self.W
        X_adv = 2 * X - 1
        X_adv = torch.tanh(0.5 * (torch.log(1 + X_adv + 1e-15) - torch.log(1 - X_adv + 1e-15)) + P)
        X_adv = 0.5 * X_adv + 0.5
        X_adv = (X_adv - self.mean) / self.std

        layer_counter = 0
        layer_activations = {}
        Y_adv = self.net(X_adv)
        total_values_sum = sum(total_values_dict.values())
        zero_values_sum = sum(zero_values_dict.values())
        density = 1 - zero_values_sum / total_values_sum

        return self.imagenet_label2_mnist_label(Y_adv), (layer_activations, density)

class Adversarial_Reprogramming(object):
    def __init__(self, args):
        self.gpu = device == 'cuda'
        self.init_dataset()
        self.Program = Program(args.model_path, args.program_path)

    def init_dataset(self):
        test_set = torchvision.datasets.CIFAR10('.', train=False, download=True, transform=transforms.ToTensor())
        kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': True}
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, **kwargs)

    @property
    def get_W(self):
        for p in self.Program.parameters():
            if p.requires_grad:
                return p

    def imagenet_label2_mnist_label(self, imagenet_label):
        return imagenet_label[:, :10]

    def tensor2var(self, tensor, requires_grad=False, volatile=False):
        if self.gpu:
            with torch.cuda.device(0):
                tensor = tensor.cuda()
        return Variable(tensor, requires_grad=requires_grad, volatile=volatile)

    def test(self):
        start_time = time.time()
        acc = 0.0
        average_density = 0.0
        for k, (image, label) in enumerate(self.test_loader): 
            image = self.tensor2var(image)
            out, (intermediate_activations, density) = self.Program(image)
            pred = out.data.cpu().numpy().argmax(1)
            average_density += sum(density.cpu().numpy()) / float(len(label) * len(self.test_loader))
            acc += sum(label.numpy() == pred) / float(len(label) * len(self.test_loader))

        print('Accuracy: %.6f' % acc)
        print('Average density: %6f' % average_density, flush=True)
        end_time = time.time()
        print("INFERENCE TIME: %s" % (end_time - start_time))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', '--model_path', type=str, help='path to pretrained model')
    parser.add_argument('-program_path', '--program_path', type=str, default=None, help='path to trained program weights')
    args = parser.parse_args()

    AR = Adversarial_Reprogramming(args)
    AR.test()

    
if __name__ == "__main__":
    main()
