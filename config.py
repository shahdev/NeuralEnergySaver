import os
from easydict import EasyDict

cfg = EasyDict()

#cfg.net = 'alexnet'
#cfg.net = 'mobilenet'
#cfg.net = 'resnet18'
#cfg.net = 'vgg9'
cfg.net = 'vgg16'
#cfg.net = 'mnist_m'
#cfg.net = 'vgg16_pretrained'
#cfg.net = 'resnet56'
cfg.dataset = 'cifar10'
#cfg.dataset = 'imagenet'
#cfg.dataset = 'mnist'
cfg.train_dir = 'train_log_%s'%cfg.net 
cfg.models_dir = 'models'
cfg.data_dir = 'datasets'

cfg.temperature = 1

cfg.batch_size_per_gpu = 30
#cfg.batch_size_per_gpu = 1
cfg.channels = 3
cfg.w1 = 32 #28 #224
cfg.h1 = 32 #28 #224
cfg.w2 = 32 #32 28 #224
cfg.h2 = 32 #32 28 #224
cfg.lmd = 5e-7
cfg.lr = 1e-3
cfg.flow_lr = 2e-3
cfg.decay = 0.96
cfg.max_epoch = 40
#cfg.lb = 1000
if not os.path.exists(cfg.train_dir):
    os.makedirs(cfg.train_dir)

if not os.path.exists(cfg.models_dir):
    os.makedirs(cfg.models_dir)

if not os.path.exists(cfg.data_dir):
    os.makedirs(cfg.data_dir)

