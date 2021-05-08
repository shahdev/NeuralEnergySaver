# -*- coding:utf-8 -*-
# Created Time: Thu 05 Jul 2018 10:00:41 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>
from config import cfg

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os
import argparse
from tqdm import tqdm, trange
from functools import reduce
from operator import mul
from vgg import VGG
from mobilenetv2 import MobileNetV2
from alexnet import AlexNet
from resnet import ResNet18
from vgg9 import VGG9
from mnist_model import Net as MNIST_M
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from codecarbon import EmissionsTracker
from codecarbon import OfflineEmissionsTracker
import torchvision.models as models
import time
import resnet56
layer_loss = {}
layer_counter = 0
total_values_dict = {}
zero_values_dict = {}


def loss_fn(self, input, output):
	global layer_loss, layer_counter, total_values_dict, zero_values_dict
	layer_counter += 1
	#if 'ReLU' in self.__class__.__name__ or 'MaxPool2d' in self.__class__.__name__:
	if self.is_conv_input:
		o_shape = output.shape
		total_values = reduce(mul, o_shape[1:], 1)
		zero_values = torch.sum(output == 0, dim=[i for i in range(1, len(o_shape))]).to(dtype=torch.float)
		total_values_dict[layer_counter] = total_values
		zero_values_dict[layer_counter] = zero_values
		layer_loss[layer_counter] = output


#def loss_fn(self, input, output):

class Program(nn.Module):
	def __init__(self, cfg, gpu, checkpoint_path):
		super(Program, self).__init__()
		self.cfg = cfg
		self.gpu = gpu
		self.num_classes = 10
		self.init_net(checkpoint_path)
		self.init_mask()
		self.W = Parameter((torch.randn(self.M.shape) * 2 - 1) * 0.0001, requires_grad=True)
		# self.W = Parameter(torch.zeros(self.M.shape), requires_grad=True)
		# self.W.data = torch.load('train_log_resnet18/lb_2/W_030.pt', map_location=torch.device(device))['W']
		#self.W = Parameter(torch.load('train_log_resnet18/lb_5e-1/W_030.pt', map_location=torch.device(device)), requires_grad=True)
		
		self.beta = 22
		self.temperature = self.cfg.temperature
		self.activation_ = torch.nn.Tanh()
		hooks = {}
		module_names = []
		for name, module in self.net.named_modules():
			module.module_name = name
			module.is_conv_input = False
			module_names.append(module.__class__.__name__)
			hooks[name] = module.register_forward_hook(loss_fn)
		print(module_names)
		module_idx = 0
		for name, module in self.net.named_modules():
			if module_idx>=2 and module_idx < len(module_names)-1:
				module.is_conv_input = 'Conv2d' in module_names[module_idx+1] or 'Linear' in module_names[module_idx+1]
			module_idx += 1

	def init_net(self, checkpoint_path):
		if self.cfg.net == 'vgg16':
			self.net = VGG('VGG16')
		elif self.cfg.net == 'mobilenet':
			self.net = MobileNetV2()
		elif self.cfg.net == 'alexnet':
			self.net = AlexNet()
		elif self.cfg.net == 'resnet18':
			self.net = ResNet18()
		elif self.cfg.net == 'vgg9':
			self.net = VGG9()
		elif self.cfg.net == 'vgg16_pretrained':
			self.net = models.vgg16(pretrained=True)
		elif self.cfg.net == 'resnet56':
			self.net = models.resnet18(pretrained=True)
			self.num_classes = 1000
			print(self.num_classes)
			#self.net = resnet56.cifar_resnet56(pretrained='cifar10')
		elif self.cfg.net == 'mnist_m':
			self.net = MNIST_M()
			self.net = self.net.to(device)
			checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
			self.net.module.load_state_dict(checkpoint)

			mean = 0.1307 #np.array([0.1307], dtype=np.float32)
			std = 0.3081 #np.array([0.3081], dtype=np.float32)
			self.mean = mean #arameter(mean, requires_grad=False)
			self.std = std #Parameter(std, requires_grad=False)
			self.net.eval()
			for param in self.net.parameters():
				param.requires_grad = False
			return 
		else:
			raise NotImplementationError()

		if self.cfg.net == 'vgg16_pretrained' or self.cfg.net == 'resnet56':						
			self.net = self.net.to(device)
			# mean and std for input
			mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
			mean = mean[..., np.newaxis, np.newaxis]
			std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
			std = std[..., np.newaxis, np.newaxis]
			self.mean = Parameter(torch.from_numpy(mean), requires_grad=False)
			self.std = Parameter(torch.from_numpy(std), requires_grad=False)
			self.net.eval()
			for param in self.net.parameters():
				param.requires_grad = False

		else:
			self.net = self.net.to(device)
			if self.cfg.net != 'resnet56':
				checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
				self.net.load_state_dict(checkpoint['net'])

			# mean and std for input
			mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
			mean = mean[..., np.newaxis, np.newaxis]
			std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
			std = std[..., np.newaxis, np.newaxis]
			self.mean = Parameter(torch.from_numpy(mean), requires_grad=False)
			self.std = Parameter(torch.from_numpy(std), requires_grad=False)

			self.net.eval()
			for param in self.net.parameters():
				param.requires_grad = False

	# Initialize mask to all 1's
	def init_mask(self):
		M = torch.ones(self.cfg.channels, self.cfg.h1, self.cfg.w1)
		# c_w, c_h = int(np.ceil(self.cfg.w1/2.)), int(np.ceil(self.cfg.h1/2.))
		# M[:,c_h-self.cfg.h2//2:c_h+self.cfg.h2//2, c_w-self.cfg.w2//2:c_w+self.cfg.w2//2] = 0
		self.M = Parameter(M, requires_grad=False)

	def imagenet_label2_mnist_label(self, imagenet_label):
		return imagenet_label[:, :self.num_classes]/self.temperature

	def forward(self, image):
		global layer_loss, layer_counter, total_values_dict, zero_values_dict
		X = image.data.new(self.cfg.batch_size_per_gpu, self.cfg.channels, self.cfg.h1, self.cfg.w1)
		X[:] = 0
		X[:, :, int((self.cfg.h1 - self.cfg.h2) // 2):int((self.cfg.h1 + self.cfg.h2) // 2),
		int((self.cfg.w1 - self.cfg.w2) // 2):int((self.cfg.w1 + self.cfg.w2) // 2)] = image.data.clone()
		X = image.data.clone()
		X = Variable(X, requires_grad=True)
		P = self.W #self.dropout(self.W)
		X_adv = 2 * X - 1
		X_adv = torch.tanh(0.5 * (torch.log(1 + X_adv + 1e-15) - torch.log(1 - X_adv + 1e-15)) + P)
		X_adv = 0.5 * X_adv + 0.5
		# X_adv = torch.clamp(X+P, 0.0, 1.0)
		X_adv = (X_adv - self.mean) / self.std

		layer_counter = 0
		layer_loss = {}
		total_values_dict = {}
		zero_values_dict = {}
		Y_adv = self.net(X_adv)
		#Y_adv = F.softmax(Y_adv, 1)
		total_values_sum = sum(total_values_dict.values())
		self.total_values_sum = total_values_sum
		zero_values_sum = sum(zero_values_dict.values())
		density = 1 - zero_values_sum / total_values_sum
		layer_wise_density = [zero_values_dict[x]/total_values_dict[x] for x in zero_values_dict]
		# l_sparsity = sum([torch.norm(x, p=2) for x in layer_loss.values()]) / total_values_sum
		l_sparsity = sum([torch.sum(self.activation_(self.beta * x)) for x in layer_loss.values()]) / total_values_sum
		return self.imagenet_label2_mnist_label(Y_adv), (l_sparsity, density, layer_wise_density)


class Adversarial_Reprogramming(object):
	def __init__(self, args, cfg=cfg):
		self.num_classes = 10
		self.mode = args.mode
		self.gpu = args.gpu
		self.restore = args.restore
		self.cfg = cfg
		self.init_dataset()
		self.Program = Program(self.cfg, self.gpu, args.checkpoint_path)
		self.restore_from_file()
		self.set_mode_and_gpu()
		self.lb = args.lb
		self.save_dir = args.save_dir #'%s/lb_%.1f/' % (self.cfg.train_dir, self.lb)
		if not os.path.isdir(self.save_dir):
			os.makedirs(self.save_dir)

	def init_dataset(self):
		if self.cfg.dataset == 'mnist':
			train_set = torchvision.datasets.MNIST(os.path.join(self.cfg.data_dir, 'mnist'), train=True,
												   transform=transforms.ToTensor(), download=True)
			test_set = torchvision.datasets.MNIST(os.path.join(self.cfg.data_dir, 'mnist'), train=False,
												  transform=transforms.ToTensor(), download=True)
			kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': True}
			if self.gpu:
				self.train_loader = torch.utils.data.DataLoader(train_set,
																batch_size=self.cfg.batch_size_per_gpu * len(self.gpu),
																shuffle=True, **kwargs)
				self.test_loader = torch.utils.data.DataLoader(test_set,
															   batch_size=self.cfg.batch_size_per_gpu * len(self.gpu),
															   shuffle=True, **kwargs)
			else:
				self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.cfg.batch_size_per_gpu,
																shuffle=True, **kwargs)
				self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.cfg.batch_size_per_gpu,
															   shuffle=True, **kwargs)
		elif self.cfg.dataset == 'cifar10':
			train_set = torchvision.datasets.CIFAR10(os.path.join(self.cfg.data_dir, 'cifar10'), train=True,
													 download=True, 
													 transform=torchvision.transforms.Compose([
															torchvision.transforms.ToTensor(),]))
			test_set = torchvision.datasets.CIFAR10(os.path.join(self.cfg.data_dir, 'cifar10'), train=False,
													download=True, transform=transforms.ToTensor())
			kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': True}
			if self.gpu:
				self.train_loader = torch.utils.data.DataLoader(train_set,
																batch_size=self.cfg.batch_size_per_gpu * len(self.gpu),
																shuffle=True, **kwargs)
				self.test_loader = torch.utils.data.DataLoader(test_set,
															   batch_size=self.cfg.batch_size_per_gpu * len(self.gpu),
															   shuffle=True, **kwargs)
			else:
				self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.cfg.batch_size_per_gpu,
																shuffle=True, **kwargs)
				self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.cfg.batch_size_per_gpu,
															   shuffle=True, **kwargs)
		elif self.cfg.dataset == 'imagenet':
			transform = transforms.Compose([transforms.Resize(255),
								transforms.CenterCrop(224),
								transforms.ToTensor()])
			path = 'val'
			self.num_classes = 1000
			train_set = datasets.ImageFolder('/home/shared/imagenet/raw/train', transform=transform)
			test_set = datasets.ImageFolder('val', transform=transform)
			num_data_samples = len(train_set)
			all_labels = {}
			for i in range(1000): all_labels[i] = []
			for i in range(len(train_set)): all_labels[train_set.targets[i]].append(i)

			samples_per_class = 10
			train_indices = []
			for i in range(1000):
				train_indices += all_labels[i][:samples_per_class] 
			train_set.imgs = list(np.array(train_set.imgs)[train_indices])
			train_set.targets = list(np.array(train_set.targets)[train_indices])
			kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': True}
			if self.gpu:
				self.train_loader = torch.utils.data.DataLoader(train_set,
																batch_size=self.cfg.batch_size_per_gpu * len(self.gpu),
																shuffle=True, **kwargs)
				self.test_loader = torch.utils.data.DataLoader(test_set,
															   batch_size=self.cfg.batch_size_per_gpu * len(self.gpu),
															   shuffle=True, **kwargs)
			else:
				self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.cfg.batch_size_per_gpu,
																shuffle=True, **kwargs)
				self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.cfg.batch_size_per_gpu,
															   shuffle=True, **kwargs)

		else:
			raise NotImplementationError()

	def restore_from_file(self):
		if self.restore is not None:
			ckpt = os.path.join(self.cfg.train_dir, 'W_%03d.pt' % self.restore)
			assert os.path.exists(ckpt)
			if self.gpu:
				self.Program.load_state_dict(torch.load(ckpt), strict=False)
			else:
				self.Program.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
			self.start_epoch = self.restore + 1
		else:
			self.start_epoch = 1

	def set_mode_and_gpu(self):
		if self.mode == 'train':
			# optimizer
			self.criterion = nn.CrossEntropyLoss()
			#self.BCE = torch.nn.BCELoss()
			self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.Program.parameters()),
											  lr=self.cfg.lr, betas=(0.5, 0.999), weight_decay=5e-4)
			self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=self.cfg.decay)

			if self.restore is not None:
				for i in range(self.restore):
					self.lr_scheduler.step()
			if self.gpu:
				with torch.cuda.device(0):
					self.criterion.cuda()
					#self.BCE.cuda()
					self.Program.cuda()

			if len(self.gpu) > 1:
				self.Program = torch.nn.DataParallel(self.Program, device_ids=list(range(len(self.gpu))))

		elif self.mode == 'validate' or self.mode == 'test':
			if self.gpu:
				with torch.cuda.device(0):
					self.Program.cuda()

			if len(self.gpu) > 1:
				self.Program = torch.nn.DataParallel(self.Program, device_ids=list(range(len(self.gpu))))

		else:
			raise NotImplementationError()

	@property
	def get_W(self):
		for p in self.Program.parameters():
			if p.requires_grad:
				return p

	def imagenet_label2_mnist_label(self, imagenet_label):
		return imagenet_label[:, :self.num_classes]/self.temperature

	def tensor2var(self, tensor, requires_grad=False, volatile=False):
		if self.gpu:
			with torch.cuda.device(0):
				tensor = tensor.cuda()
		return Variable(tensor, requires_grad=requires_grad, volatile=volatile)

	def compute_loss(self, out, label):
		label = self.tensor2var(label)
		return self.criterion(out, label) #+ self.cfg.lmd * torch.norm(self.get_W) ** 2

	def validate(self):
		#tracker = EmissionsTracker()
		tracker = OfflineEmissionsTracker(country_iso_code="USA")
		tracker.start()
		start_time = time.time()
		acc = 0.0
		average_density = 0.0
		average_layer_wise_density = []
		for k, (image, label) in enumerate(self.train_loader):
			image = self.tensor2var(image)
			out, (_, density, layer_wise_density) = self.Program(image)
			pred = out.data.cpu().numpy().argmax(1)
			average_density += sum(density.cpu().numpy()) / float(len(label) * len(self.train_loader))
			if average_layer_wise_density == []:
				average_layer_wise_density = [0.0 for x in layer_wise_density]
			for layer_num in range(len(average_layer_wise_density)):
				average_layer_wise_density[layer_num] += torch.sum(layer_wise_density[layer_num]).cpu().numpy()/float(len(label) * len(self.train_loader))
			acc += sum(label.numpy() == pred) / float(len(label) * len(self.train_loader))
		print('train accuracy: %.6f' % acc, flush=True)
		print('train average density: %6f' % average_density, flush=True)
		print('Total activation size: %d' % self.Program.total_values_sum)
		print('train layer wise density', average_layer_wise_density)
		acc = 0.0
		average_density = 0.0
		average_layer_wise_density = []
		for k, (image, label) in enumerate(self.test_loader):
			image = self.tensor2var(image)
			out, (_, density, layer_wise_density) = self.Program(image)
			pred = out.data.cpu().numpy().argmax(1)
			average_density += sum(density.cpu().numpy()) / float(len(label) * len(self.test_loader))
			if average_layer_wise_density == []:
				average_layer_wise_density = [0.0 for x in layer_wise_density]
			for layer_num in range(len(average_layer_wise_density)):
				average_layer_wise_density[layer_num] += torch.sum(layer_wise_density[layer_num]).cpu().numpy()/float(len(label) * len(self.test_loader))
			acc += sum(label.numpy() == pred) / float(len(label) * len(self.test_loader))
		print('test accuracy: %.6f' % acc)
		print('test average density: %6f' % average_density, flush=True)
		print('test layer wise density', average_layer_wise_density)
		emissions: float = tracker.stop()
		print(f"Emissions: {emissions} kg")
		end_time = time.time()
		print("INFERENCE TIME: %s"%(end_time-start_time))

	def train(self):
		self.validate()
		for self.epoch in range(self.start_epoch, self.cfg.max_epoch + 1):
			
			self.lr_scheduler.step()
			for j, (image, label) in tqdm(enumerate(self.train_loader)):
				image = self.tensor2var(image)
				self.out, (l_sparsity, density, _) = self.Program(image)
				self.loss = self.compute_loss(self.out, label) + self.lb * l_sparsity
				self.optimizer.zero_grad()
				self.loss.backward()
				self.optimizer.step()
				# print(self.loss.data.cpu().numpy(), l_sparsity.data.cpu().numpy())
			print('epoch: %03d/%03d, loss: %.6f, l_sparsity: %.6f' % (
			self.epoch, self.cfg.max_epoch, self.loss.data.cpu().numpy(), l_sparsity.data.cpu().numpy()))
			torch.save({'W': self.get_W}, '%s/W_%03d.pt' % (self.save_dir, self.epoch))
			self.validate()
			if self.epoch%20==19:
				self.lb *= 0.5
				print("Lambda value: ", self.lb)

	def test(self):
		pass


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'validate', 'test'])
	parser.add_argument('-r', '--restore', default=None, action='store', type=int,
						help='Specify checkpoint id to restore.')
	parser.add_argument('-g', '--gpu', default=[], nargs='+', type=str, help='Specify GPU ids.')
	parser.add_argument('-lb', '--lb', type=float, help='proportion of sparsity term')
	parser.add_argument('-checkpoint_path', '--checkpoint_path', type=str, help='path to pretrained model')
	parser.add_argument('-save_dir', '--save_dir', type=str, help='path to save program weights')
	# test params

	args = parser.parse_args()
	# print(args)
	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
	AR = Adversarial_Reprogramming(args)
	if args.mode == 'train':
		AR.train()
	elif args.mode == 'validate':
		AR.validate()
	elif args.mode == 'test':
		AR.test()
	else:
		raise NotImplementationError()


if __name__ == "__main__":
	main()

