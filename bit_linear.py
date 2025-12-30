import torch
import torch.nn as nn
import warp_bitnet_cuda
from packer import pack_ternary_weights

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # We don't store the full FP16 weight. We store the packed version.
        # Shape: (out_features, in_features // 16)
        # We use register_buffer so it's part of the state_dict but not optimized by an optimizer
        self.register_buffer('packed_weight', torch.zeros(out_features, in_features // 16, dtype=torch.int32))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
            
    def load_from_dense_weights(self, dense_weight, bias=None):
        """
        Load from a standard {-1, 0, 1} float tensor.
        dense_weight: (out_features, in_features)
        """
        if dense_weight.shape != (self.out_features, self.in_features):
            raise ValueError(f"Shape mismatch: expected {self.out_features}x{self.in_features}, got {dense_weight.shape}")
            
        # Pack and store
        self.packed_weight.copy_(pack_ternary_weights(dense_weight))
        
        if bias is not None:
            if self.bias is not None:
                self.bias.data.copy_(bias)
                
    def forward(self, x):
        # x shape: (batch_size, in_features) or (in_features)
        # Our kernel currently supports (in_features) -> (out_features) i.e. GEMV
        
        if x.dim() == 2:
            # Batch mode - Naive loop for now (Kernel is GEMV)
            # Ideally we would update kernel to support GEMM for high batch size
            batch_size, dim = x.shape
            out = torch.empty(batch_size, self.out_features, dtype=torch.float16, device=x.device)
            
            for i in range(batch_size):
                warp_bitnet_cuda.gemv(self.packed_weight, x[i], out[i])
                
            if self.bias is not None:
                out += self.bias
            return out
            
        elif x.dim() == 1:
            out = torch.empty(self.out_features, dtype=torch.float16, device=x.device)
            warp_bitnet_cuda.gemv(self.packed_weight, x, out)
            if self.bias is not None:
                out += self.bias
            return out

class BitNetMLP(nn.Module):
    """
    A simple MLP block using BitLinear layers.
    Simulates the FeedForward network in a Transformer.
    """
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
