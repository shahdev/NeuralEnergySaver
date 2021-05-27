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

import pdb
reprogram = True
class EnergyEstimation():

    def __init__(self):
        self.xvf32ger_energy = 395.2*0.505  # dynamic energy per instruction = 1 unit as of now since I need to check if this value can be shared.
        self.xvf32ger_multiplies = 16
        self.accumulators = 8 # 64B each
        self.xvf32ger_n = 4 # 4 instructions to create a 4x4 outer product of two 4-element vectors
        self.bblock_r = 8
        self.bblock_c = 16
        self.xvf32ger_energy_density_scale = [0.203665988, 0.476610249, 0.601375176, 0.700340856, 0.801657264, 0.865362012, 0.910437236, 0.963622473, 0.990185708, 0.998883404, 1]


 

    
    def baseline_energy_dataswitching(self, weight_size, input_size, output_size, density):
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
        print('uniform sparse with 0-valued computes eliminated', energy)
        return energy

    def sparse_energy(self, activations, weights, input_size, weight_size, output_size):
        # Here activations is the 0-padded matrix
        ishape = list(activations.shape)
        wshape = list(weights.shape)

        xvf32ger_instructions_eliminated = 0
        xvf32ger_instructions = 0
        xvf32ger_energy_total = 0.0
        xvf32ger_energy_skip_inst = 0.0
        xvf32ger_energy_skip_comp = 0.0
        for b in range(0,ishape[0]): # batches
            for c in range(0,wshape[1]): # input channels
    
                #print(activations[b,c,:,:])
                for i in range(0,wshape[2]): # row stride      
                    for j in range(0,wshape[3]): # column stride

                        # assemble elements of one row of MMA activation matrix
                        rowCount = 0 
                        for k in range(i,ishape[2]-wshape[2]+i+1): # rows of activations 
                            #print(b,c,k,j, ishape[3]-wshape[3]+j+1) 
                            if rowCount==0: 
                                mma_act_row = activations[b,c,k,j:ishape[3]-wshape[3]+j+1] # choose elements
                            else: 
                                mma_act_row = torch.cat((mma_act_row,activations[b,c,k,j:ishape[3]-wshape[3]+j+1]),0) 
                            rowCount += 1
                        #print(mma_act_row) 
                        #pdb.set_trace()

                        # check for density in each 4-element sequence
                        mshape=list(mma_act_row.shape)
                        for m in range(0,np.ceil(mshape[0]/4).astype(int)):
                            xvf32ger_instructions += 1

                            density = 1 - (torch.sum(mma_act_row.view(-1)[:4:] == 0))/4
                            if (density == 0):
                                xvf32ger_instructions_eliminated += 1
                            else:
                                sparse_index = (density*10).to(torch.uint8)
                                xvf32ger_energy_skip_inst += self.xvf32ger_energy_density_scale[sparse_index] * self.xvf32ger_energy

                            computes_skipped = torch.sum(mma_act_row.view(-1)[:4:] == 0)
                            energy_reduction = (self.xvf32ger_energy/self.xvf32ger_multiplies)* computes_skipped * 4 # each zero-valued activation results in 4 0-valued partial products among 16 
                            sparse_energy = self.xvf32ger_energy - energy_reduction
                            xvf32ger_energy_skip_comp += sparse_energy

                            # baseline energy without skipping sparse computations or instructions 
                            sparse_index = (density*10).to(torch.uint8)
                            xvf32ger_energy_total += self.xvf32ger_energy_density_scale[sparse_index] * self.xvf32ger_energy

                            mma_act_row = mma_act_row.view(-1)[4::]
 
        M = weight_size[0]
        K = input_size[0]
        N = input_size[1]
        print(M,K,N)
        xvf32ger_instructions_eliminated = xvf32ger_instructions_eliminated * (M/4)                 
        #print('instructions eliminated', xvf32ger_instructions_eliminated)
        xvf32ger_instructions = xvf32ger_instructions * (M/4)
        print('percentage instructions eliminated %.4f'%(xvf32ger_instructions_eliminated/xvf32ger_instructions))
        xvf32ger_energy_total = xvf32ger_energy_total * (M/4)
        print('Total energy baseline %.4f'%(xvf32ger_energy_total))
        xvf32ger_energy_skip_inst = xvf32ger_energy_skip_inst * (M/4)
        print('Total energy with skipped instructions %.4f'%(xvf32ger_energy_skip_inst))
        xvf32ger_energy_skip_comp = xvf32ger_energy_skip_comp * (M/4)
        print('Total energy with skipped computations per instruction %.4f'%(xvf32ger_energy_skip_comp))
        #print('Energy reduction in percentage %.4f'%((xvf32ger_energy_total - xvf32ger_energy_sparse)/xvf32ger_energy_total*100))
        ## verify code to compute # instructions
        #bblock8x16_xvf32ger_n = self.accumulators * self.xvf32ger_n  
        #block8x16_xvf32ger_n = bblock8x16_xvf32ger_n * K/4
        #block8x16_n = np.ceil(M/self.bblock_r) * np.ceil(N/self.bblock_c)
        #print('total number of instructions', block8x16_n*block8x16_xvf32ger_n)                            
        
        # verify energy reduction    
        # print('Energy reduced %.4f'%(xvf32ger_instructions_eliminated*self.xvf32ger_energy_density_scale[0]*self.xvf32ger_energy))



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

        mma_weight_shape = [w_shape[0], w_shape[1]*w_shape[2]*w_shape[3]]
        mma_input_shape = [w_shape[1]*w_shape[2]*w_shape[3], o_shape[2]*o_shape[3]]
        mma_output_shape = [o_shape[1], o_shape[2]*o_shape[3]] 
        
        #print(mma_weight_shape, mma_input_shape, mma_output_shape)

        # 0 sparsity in data (just for verification)
        #ee.no_sparsity_energy(mma_weight_shape, mma_input_shape, mma_output_shape)

        # per-layer sparsity, where the computations that can be skipped are randomly distributed, and the % density of activations is the same for each block
        #density = torch.count_nonzero(conv2d_input)/(i_shape[0]*i_shape[1]*i_shape[2]*i_shape[3])
        density = 1 - (torch.sum(conv2d_input==0).to(torch.float32)/(i_shape[0]*i_shape[1]*i_shape[2]*i_shape[3]))
        print('layer density %.4f'%(density))
        #ee.uniform_sparsity_energy(mma_weight_shape, mma_input_shape, mma_output_shape, density.cpu().numpy())

        # zeropadded activation matrix to compute instruction level sparsity
        zpad = nn.ZeroPad2d(w_shape[2]-2)
        conv2d_input_z = zpad(conv2d_input)
        iz_shape = list(conv2d_input_z.shape)
        ee.sparse_energy(conv2d_input_z, conv2d_weight, mma_input_shape, mma_weight_shape, mma_output_shape)

                



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
    parser.add_argument('-program_path', '--program_path', type=str, default='W_conv_input.pt', help='path to trained program weights')
    parser.add_argument('-image_path', '--image_path', type=str, default='image.npy', help='path to image')
    parser.add_argument('-reprogram', '--reprogram', type=bool, default=False, help='if reprogrammed image is used')
    args = parser.parse_args()

    AR = Adversarial_Reprogramming(args)
    AR.forward(args.image_path)

    
if __name__ == "__main__":
    main()
