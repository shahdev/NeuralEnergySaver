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
batch_size = 1
input_shape = (3, 32, 32)


reprogram = True
class EnergyEstimation():

    def __init__(self):
        self.xvf32ger_energy = 1 # dynamic energy per instruction = 1 unit as of now since I need to check if this value can be shared.
        self.xvf32ger_multiplies = 16
        self.accumulators = 8 # 64B each
        self.xvf32ger_n = 4 # 4 instructions to create a 4x4 outer product of two 4-element vectors
        self.bblock_r = 8
        self.bblock_c = 16
        self.xvf32ger_energy_density_scale = [164/1701.6, 811/1701.6, 1023.3/1701.6, 1191.7/1701.6, 1364.1/1701.6, 1472.5/1701.6, 1549.2/1701.6, 1639.7/1701.6, 1684.9/1701.6, 1699.7/1701.6, 1701.6/1701.6]

    def no_sparsity_energy(self, weight_size, input_size, output_size):

        bblock8x16_xvf32ger_n = self.accumulators * self.xvf32ger_n # number of instructions to compute 8x8 block with 8x4 and 4x16 inputs 
        bblock8x16_energy = bblock8x16_xvf32ger_n * self.xvf32ger_energy
        """
         Matrix multiply compute MxN = MxK * KxN
         Each MxN output block is divided into computation blocks of size 8x16. 
         Outer product matrix multiply is computed for inputs of size 8xK, Kx16

        """
    
        M = output_size[0]
        K = input_size[0] # or weight_size[1]
        N = output_size[1]
        block8x16_xvf32ger_n = bblock8x16_xvf32ger_n * K/4
        block8x16_energy = K/4 * bblock8x16_energy

        block8x16_n = np.ceil(M/self.bblock_r) * np.ceil(N/self.bblock_c)
        energy = block8x16_n * block8x16_energy
        print('random init', energy)
        return energy

    
    def uniform_sparsity_energy_ds(self, weight_size, input_size, output_size, density):
        """
        This estimates energy for a given layer, if there is no explicit support for sparsity. 0-valued computations are not eliminated. 
        The objective is to include the effect of data switching. This will be the baseline case.  
        """
        bblock8x16_xvf32ger_n = self.accumulators * self.xvf32ger_n # number of instructions to compute 8x8 block with 8x4 and 4x16 inputs 
        sparse_index = np.array(np.round(density*10), dtype=np.uint8)
        bblock8x16_energy = bblock8x16_xvf32ger_n * self.xvf32ger_energy * self.xvf32ger_energy_density_scale[sparse_index]
        """
         Matrix multiply compute MxN = MxK * KxN
         Each MxN output block is divided into computation blocks of size 8x16. 
         Outer product matrix multiply is computed for inputs of size 8xK, Kx16
        """
        M = output_size[0]
        K = input_size[0] # or weight_size[1]
        N = output_size[1]
        block8x16_xvf32ger_n = bblock8x16_xvf32ger_n * K/4
        block8x16_energy = K/4 * bblock8x16_energy
        
        block8x16_n = np.ceil(M/self.bblock_r) * np.ceil(N/self.bblock_c)
        energy = block8x16_n * block8x16_energy
        print('uniform sparse with data switching energy reduction', energy)
        return energy

    def uniform_sparsity_energy(self, weight_size, input_size, output_size, density):
        """
        This estimates energy for a given layer, if every basic block in that layer had the same activation density. 
        We assume that there is explicit support for fine-grained sparsity exploitation, by way of perfect run-time prediction of 0-valued inputs and fine-grained clock-gating.
        We assume that since the prediction capability exists, 0-valued computations can be eliminated from each of the 4x4 basic blocks, resulting in dynamic energy reduction .         
        """
        bblock8x16_xvf32ger_n = self.accumulators * self.xvf32ger_n 
        # number of instructions to compute 8x8 block with 8x4 and 4x16 inputs 
        computations_reduced = np.floor((1-density) * 4) * 4
        # Each 0-valued activation in 4-element vector results in elimination of 4 computations
        energy_reduction = (self.xvf32ger_energy/self.xvf32ger_multiplies)* computations_reduced 
        sparse_energy = self.xvf32ger_energy - energy_reduction
        bblock8x16_energy = bblock8x16_xvf32ger_n * sparse_energy 

        M = output_size[0]
        K = input_size[0] # or weight_size[1]
        N = output_size[1]
        block8x16_xvf32ger_n = bblock8x16_xvf32ger_n * K/4
        block8x16_energy = K/4 * bblock8x16_energy
        block8x16_n = np.ceil(M/self.bblock_r) * np.ceil(N/self.bblock_c)
        energy = block8x16_n * block8x16_energy
        print('uniform sparse with 0-valued computes eliminated', energy, computations_reduced)
        return energy

    """TO BE COMPLETED 
    #def instruction_level_sparsity_energy():
    """



def mma_instructions_estimate(conv2d_i, conv2d_w, conv2d_o):

    with torch.no_grad():
        conv2d_input = conv2d_i.clone().detach()
        conv2d_weight = conv2d_w.clone().detach()
        conv2d_output = conv2d_o.clone().detach()

        ee = EnergyEstimation()

        i_shape = list(conv2d_input.shape)
        o_shape = list(conv2d_output.shape)
        w_shape = list(conv2d_weight.shape)
        #print('input', i_shape, 'output', o_shape, 'weight', w_shape)

        # deriving matrix shapes for MMA instructions

        gemm_weight_shape = [w_shape[0], w_shape[1]*w_shape[2]*w_shape[3]]
        gemm_input_shape = [i_shape[1]*w_shape[2]*w_shape[3], o_shape[2]*o_shape[3]]
        gemm_output_shape = [o_shape[1], o_shape[2]*o_shape[3]] 
        
        #print(gemm_weight_shape, gemm_input_shape, gemm_output_shape)

        # 0 sparsity in data (just for verification)
        #ee.no_sparsity_energy(gemm_weight_shape, gemm_input_shape, gemm_output_shape)

        # per-layer sparsity, where the computations that can be skipped are randomly distributed, and the % density of activations is the same for each block
        #density = torch.count_nonzero(conv2d_input)/(i_shape[0]*i_shape[1]*i_shape[2]*i_shape[3])
        density = torch.sum(conv2d_input != 0)/(i_shape[0]*i_shape[1]*i_shape[2]*i_shape[3])
        density = density.cpu().numpy()
        ee.uniform_sparsity_energy(gemm_weight_shape, gemm_input_shape, gemm_output_shape, density)

        """ Code below to be completed 
        """
        # Reshaping activation matrix to compute instruction level sparsity
        zpad = nn.ZeroPad2d(w_shape[2]-2)
        conv2d_input_z = zpad(conv2d_input)
        iz_shape = list(conv2d_input_z.shape)
        conv2d_input_r = torch.reshape(conv2d_input_z, (iz_shape[0], iz_shape[1]*iz_shape[2]*iz_shape[3]))

        #print(conv2d_input_z.shape, conv2d_input_r.shape)
        ir_shape = list(conv2d_input_r.shape)

        #for i in range(0,w_shape[2]):
        #    conv2d_input_r[i,0:ir_shape[1]-w_shape[2]+1)]

def activations(self, input, output):
    global layer_counter, total_values_dict, zero_values_dict

    if 'Conv2d' in self.__class__.__name__:
        i_shape = input[0].shape
        total_values = reduce(mul, i_shape[1:], 1)
        #zero_values = total_values - torch.count_nonzero(input[0])  
        zero_values = torch.sum(input[0] == 0, dim=[i for i in range(1, len(i_shape))]).to(dtype=torch.float)
        total_values_dict[layer_counter] = total_values
        zero_values_dict[layer_counter] = zero_values
        mma_instructions_estimate(input[0], self.weight, output)

    layer_counter += 1

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
        print(self.net)
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
        if (reprogram):
            X_adv = 2 * X - 1
            P = self.W
            X_adv = torch.tanh(0.5 * (torch.log(1 + X_adv + 1e-15) - torch.log(1 - X_adv + 1e-15)) + P)
            X_adv = 0.5 * X_adv + 0.5
        else:
            X_adv = X
        X_adv = (X_adv - self.mean) / self.std

        layer_counter = 0
        layer_activations = {}
        Y_adv = self.net(X_adv)
        for q in zero_values_dict.keys():
            print('density in layer %d : %.6f'%(q,1-zero_values_dict[q]/total_values_dict[q]))
        total_values_sum = sum(total_values_dict.values())
        zero_values_sum = sum(zero_values_dict.values())
        density = 1 - zero_values_sum / total_values_sum

        return self.imagenet_label2_mnist_label(Y_adv), (layer_activations, density)

class Adversarial_Reprogramming(object):
    def __init__(self, args):
        self.gpu = device == 'cuda'
        self.Program = Program(args.model_path, args.program_path)
    
    # def init_dataset(self):
    #     test_set = torchvision.datasets.CIFAR10('.', train=False, download=True, transform=transforms.ToTensor())
    #     kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': True}
    #     self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, **kwargs)

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

    def forward(self, image_path):
        image = torch.tensor(np.expand_dims(np.load(image_path), axis=0))
        image = self.tensor2var(image)
        start_time = time.time()
        out, (intermediate_activations, density) = self.Program(image)
        end_time = time.time()
        average_density = sum(density.cpu().numpy()) 
        print('Average density: %6f' % average_density, flush=True)
        print("INFERENCE TIME: %s" % (end_time - start_time))
        torch.save(intermediate_activations, 'activations.pt')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', '--model_path', type=str, default='ckpt.pth', help='path to pretrained model')
    parser.add_argument('-program_path', '--program_path', type=str, default='W.pt', help='path to trained program weights')
    parser.add_argument('-image_path', '--image_path', type=str, default='image.npy', help='path to image')
    args = parser.parse_args()

    AR = Adversarial_Reprogramming(args)
    AR.forward(args.image_path)

    
if __name__ == "__main__":
    main()
