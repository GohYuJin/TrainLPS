

import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from APS.circular_pad_layer import circular_pad



class ApsPool(nn.Module):
    def __init__(self, channels, pad_type='circular', filt_size=3, stride=2, apspool_criterion = 'l2', 
                return_poly_indices = True, circular_flag = True, N = None):
        super(ApsPool, self).__init__()
        
        self.filt_size = filt_size
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        self.N = N
        self.return_poly_indices = return_poly_indices
        
        self.apspool_criterion = apspool_criterion
        
        if self.filt_size>1:
            a = construct_1d_array(self.filt_size)

            filt = torch.Tensor(a[:,None]*a[None,:])
            filt = filt/torch.sum(filt)
            self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)
        
        if self.N is not None:
            self.permute_indices = permute_polyphase(N, stride = 2)
            
        else:
            self.permute_indices = None
            
            
        
    def forward(self, input_to_pool):
        
        if isinstance(input_to_pool, dict):
            inp, polyphase_indices = input_to_pool['inp'], input_to_pool['polyphase_indices']
    
        else:
#             this is the case when polyphase indices are not pre-defined
            inp = input_to_pool
            polyphase_indices = None
        if(self.filt_size==1):
            return aps_downsample_v2(aps_pad(inp), self.stride, polyphase_indices, return_poly_indices = self.return_poly_indices, permute_indices = self.permute_indices, apspool_criterion = self.apspool_criterion)
            
        else:
            
            blurred_inp = F.conv2d(self.pad(inp), self.filt, stride = 1, groups=inp.shape[1])
            return aps_downsample_v2(aps_pad(blurred_inp), self.stride, polyphase_indices, return_poly_indices = self.return_poly_indices, permute_indices = self.permute_indices, apspool_criterion = self.apspool_criterion)
        

def aps_downsample_v2(x, stride, polyphase_indices = None, return_poly_indices = True, permute_indices = None, apspool_criterion = 'l2'):
    if stride==1:
        return x
    
    elif stride>2:
        raise Exception('Stride>2 currently not supported in this implementation')
    
    
    else:
        B, C, N, _ = x.shape
        N_poly = int(N**2/4)
        Nb2 = int(N/2)
        
        if permute_indices is None:
            permute_indices = permute_polyphase(N, device=x.device).long()
            
        xpoly_0 = x[:, :, ::stride, ::stride]
        xpoly_1 = x[:, :, 1::stride, ::stride]
        xpoly_2 = x[:, :, ::stride, 1::stride]
        xpoly_3 = x[:, :, 1::stride, 1::stride]
        xpoly_combined = torch.stack([xpoly_0, xpoly_1, xpoly_2, xpoly_3], dim=1)
        
        if polyphase_indices is None:
            polyphase_indices = get_polyphase_indices_v2(xpoly_combined, apspool_criterion)
            
        batch_indices = torch.arange(B, device=x.device)
        output = xpoly_combined[batch_indices, polyphase_indices, :, :, :]
        
        if return_poly_indices:
            return output, polyphase_indices
        
        else:
            return output
        
        
def get_polyphase_indices_v2(x, apspool_criterion):
#     x has the form (B, 4, C, N_poly) where N_poly corresponds to the reduced version of the 2d feature maps

    if apspool_criterion == 'l2':
        norms = torch.norm(x, dim = (2, 3, 4), p = 2)
        polyphase_indices = torch.argmax(norms, dim = 1)
        
    elif apspool_criterion == 'l1':
        norms = torch.norm(x, dim = (2, 3, 4), p = 1)
        polyphase_indices = torch.argmax(norms, dim = 1)
        
    elif apspool_criterion == 'l_infty':
        B = x.shape[0]
        max_vals = torch.max(x.reshape(B, 4, -1).abs(), dim = 2).values
        polyphase_indices = torch.argmax(max_vals, dim = 1)
        
    
    elif apspool_criterion == 'non_abs_max':
        B = x.shape[0]
        max_vals = torch.max(x.reshape(B, 4, -1), dim = 2).values
        polyphase_indices = torch.argmax(max_vals, dim = 1)
        
        
    elif apspool_criterion == 'l2_min':
        norms = torch.norm(x, dim = (2, 3, 4), p = 2)
        polyphase_indices = torch.argmin(norms, dim = 1)
        
    elif apspool_criterion == 'l1_min':
        norms = torch.norm(x, dim = (2, 3, 4), p = 1)
        polyphase_indices = torch.argmin(norms, dim = 1)
        
    else:
        raise Exception('Unknown APS criterion')
        
        
        
    return polyphase_indices
        
        
def construct_1d_array(filt_size):
    
    if(filt_size==1):
        a = np.array([1.,])
    elif(filt_size==2):
        a = np.array([1., 1.])
    elif(filt_size==3):
        a = np.array([1., 2., 1.])
    elif(filt_size==4):    
        a = np.array([1., 3., 3., 1.])
    elif(filt_size==5):    
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size==6):    
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size==7):    
        a = np.array([1., 6., 15., 20., 15., 6., 1.])
        
    return a
    

def aps_pad(x):
    
    N1, N2 = x.shape[2:4]
    
    if N1%2==0 and N2%2==0:
        return x
    
    if N1%2!=0:
        x = F.pad(x, (0, 0, 0, 1), mode = 'circular')
    
    if N2%2!=0:
        x = F.pad(x, (0, 1, 0, 0), mode = 'circular')
    
    return x
        
    
    
def permute_polyphase(N, stride = 2, device='cpu'):
    
    base_even_ind = 2*torch.arange(int(N/2), device=device)[None, :]
    base_odd_ind = 1 + 2*torch.arange(int(N/2), device=device)[None, :]
    
    even_increment = 2*N*torch.arange(int(N/2), device=device)[:,None]
    odd_increment = N + 2*N*torch.arange(int(N/2), device=device)[:,None]
    
    p0_indices = (base_even_ind + even_increment).view(-1)
    p1_indices = (base_even_ind + odd_increment).view(-1)
    
    p2_indices = (base_odd_ind + even_increment).view(-1)
    p3_indices = (base_odd_ind + odd_increment).view(-1)
    
    permute_indices = torch.cat([p0_indices, p1_indices, p2_indices, p3_indices], dim = 0)
    
    return permute_indices





def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    elif(pad_type == 'circular'):
        PadLayer = circular_pad
        
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

# %%
