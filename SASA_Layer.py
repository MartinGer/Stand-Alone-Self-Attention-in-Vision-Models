import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
    

class SASA_Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=7, num_heads=8, image_size=224, inference=False):
        super(SASA_Layer, self).__init__()
        self.kernel_size = min(kernel_size, image_size) # receptive field shouldn't be larger than input H/W         
        self.num_heads = num_heads
        self.dk = self.dv = in_channels
        self.dkh = self.dk // self.num_heads
        self.dvh = self.dv // self.num_heads

        assert self.dk % self.num_heads == 0, "dk should be divided by num_heads. (example: dk: 32, num_heads: 8)"
        assert self.dk % self.num_heads == 0, "dv should be divided by num_heads. (example: dv: 32, num_heads: 8)"  
        
        self.k_conv = nn.Conv2d(self.dk, self.dk, kernel_size=1).to(device)
        self.q_conv = nn.Conv2d(self.dk, self.dk, kernel_size=1).to(device)
        self.v_conv = nn.Conv2d(self.dv, self.dv, kernel_size=1).to(device)
        
        # Positional encodings
        self.rel_encoding_h = nn.Parameter(torch.randn(self.dk // 2, self.kernel_size, 1), requires_grad=True)
        self.rel_encoding_w = nn.Parameter(torch.randn(self.dk // 2, 1, self.kernel_size), requires_grad=True)
        
        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)
            
    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Compute k, q, v
        padded_x = F.pad(x, [(self.kernel_size-1)//2, (self.kernel_size-1)-((self.kernel_size-1)//2), (self.kernel_size-1)//2, (self.kernel_size-1)-((self.kernel_size-1)//2)])
        k = self.k_conv(padded_x)
        q = self.q_conv(x)
        v = self.v_conv(padded_x)
        
        # Unfold patches into [BS, num_heads*depth, horizontal_patches, vertical_patches, kernel_size, kernel_size]
        k = k.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        v = v.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)

        # Reshape into [BS, num_heads, horizontal_patches, vertical_patches, depth_per_head, kernel_size*kernel_size]
        k = k.reshape(batch_size, self.num_heads, height, width, self.dkh, -1)
        v = v.reshape(batch_size, self.num_heads, height, width, self.dvh, -1)
        
        # Reshape into [BS, num_heads, height, width, depth_per_head, 1]
        q = q.reshape(batch_size, self.num_heads, height, width, self.dkh, 1)

        qk = torch.matmul(q.transpose(4, 5), k)    
        qk = qk.reshape(batch_size, self.num_heads, height, width, self.kernel_size, self.kernel_size)
        
        # Add positional encoding
        qr_h = torch.einsum('bhxydz,cij->bhxyij', q, self.rel_encoding_h)
        qr_w = torch.einsum('bhxydz,cij->bhxyij', q, self.rel_encoding_w)
        qk += qr_h
        qk += qr_w
        
        qk = qk.reshape(batch_size, self.num_heads, height, width, 1, self.kernel_size*self.kernel_size)
        weights = F.softmax(qk, dim=-1)    
        
        if self.inference:
            self.weights = nn.Parameter(weights)
        
        attn_out = torch.matmul(weights, v.transpose(4, 5)) 
        attn_out = attn_out.reshape(batch_size, -1, height, width)
        return attn_out
    