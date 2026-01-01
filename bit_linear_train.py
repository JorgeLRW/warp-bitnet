"""
BitNet Training Module
======================

Drop-in replacement for nn.Linear that uses quantized weights for forward pass
but maintains full-precision gradients via Straight-Through Estimator (STE).

Usage:
    from bit_linear_train import BitLinearTrain
    
    # Replace: layer = nn.Linear(4096, 14336)
    # With:    layer = BitLinearTrain(4096, 14336)
    
    # Forward uses quantized 1.58-bit weights
    # Backward computes full-precision gradients for master weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# Import CUDA kernels
try:
    import bitnet_train_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: bitnet_train_cuda not found. Run: python setup_train.py build_ext --inplace")


def pack_ternary(W_ternary):
    """Pack ternary weights {-1, 0, +1} into int32 (16 weights per int)"""
    M, N = W_ternary.shape
    assert N % 16 == 0, "N must be divisible by 16"
    
    packed = torch.zeros(M, N // 16, dtype=torch.int32, device=W_ternary.device)
    for i in range(16):
        col = W_ternary[:, i::16]
        # Encoding: 0 -> 00, +1 -> 01, -1 -> 10
        bits = torch.where(col == 0, 0, torch.where(col == 1, 1, 2))
        packed |= bits.int() << (i * 2)
    return packed


def quantize_to_ternary(W, scale=None):
    """
    Quantize FP16/FP32 weights to ternary {-1, 0, +1}.
    
    Uses AbsMax quantization:
        scale = max(|W|)
        W_q = round(W / scale) clamped to {-1, 0, +1}
    """
    if scale is None:
        scale = W.abs().max()
    
    W_normalized = W / (scale + 1e-8)
    W_ternary = torch.round(W_normalized).clamp(-1, 1)
    
    return W_ternary, scale


class BitLinearFunction(Function):
    """
    Autograd function for BitNet linear layer.
    
    Forward: Y = quantize(W) @ X  (uses CUDA kernel)
    Backward: Uses STE - gradient flows through as if quantization didn't happen
    """
    
    @staticmethod
    def forward(ctx, X, W_master, bias=None):
        """
        X: (B, N) input
        W_master: (M, N) full-precision master weights
        bias: (M,) optional bias
        
        Returns: (B, M) output
        """
        B, N = X.shape
        M = W_master.shape[0]
        
        # Quantize weights to ternary
        W_ternary, scale = quantize_to_ternary(W_master)
        W_packed = pack_ternary(W_ternary)
        
        # Save for backward
        ctx.save_for_backward(X, W_packed, W_ternary)
        ctx.scale = scale
        ctx.has_bias = bias is not None
        
        # Allocate output
        Y = torch.zeros(B, M, dtype=X.dtype, device=X.device)
        
        # Use PyTorch for correctness (CUDA GEMM needs debugging)
        # The forward pass: Y = X @ W^T where W is ternary
        Y = F.linear(X, W_ternary * scale, None)
        
        if bias is not None:
            Y = Y + bias
        
        return Y
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (B, M) gradient from upstream
        
        Returns:
            grad_X: (B, N) gradient w.r.t input
            grad_W: (M, N) gradient w.r.t master weights (STE)
            grad_bias: (M,) gradient w.r.t bias
        """
        X, W_packed, W_ternary = ctx.saved_tensors
        scale = ctx.scale
        
        B, M = grad_output.shape
        N = X.shape[1]
        
        grad_X = None
        grad_W = None
        grad_bias = None
        
        # Gradient w.r.t. input: dX = dY @ W (W is MxN, dY is BxM, dX is BxN)
        if ctx.needs_input_grad[0]:
            # Use PyTorch for correctness
            grad_X = F.linear(grad_output, W_ternary.t().contiguous())
        
        # Gradient w.r.t. weights: dW = dY^T @ X (STE: gradient passes through)
        if ctx.needs_input_grad[1]:
            # Standard gradient - STE means we pretend quantization didn't happen
            grad_W = grad_output.t() @ X
        
        # Gradient w.r.t. bias
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)
        
        return grad_X, grad_W, grad_bias


class BitLinearTrain(nn.Module):
    """
    Drop-in replacement for nn.Linear with 1.58-bit quantized forward pass.
    
    During training:
        - Forward uses quantized weights (ternary)
        - Backward uses full-precision gradients (STE)
        - Optimizer updates full-precision master weights
    
    During inference:
        - Call .quantize() to freeze weights and use fast inference kernel
    """
    
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Master weights (full precision)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype or torch.float16)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype or torch.float16)
            )
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        
        # Cached quantized weights for inference
        self._quantized = False
        self._W_packed = None
        self._scale = None
    
    def forward(self, x):
        if self._quantized:
            # Fast inference path
            return self._inference_forward(x)
        else:
            # Training path with STE
            return BitLinearFunction.apply(x, self.weight, self.bias)
    
    def _inference_forward(self, x):
        """Fast inference using pre-quantized weights"""
        B = x.shape[0] if x.dim() > 1 else 1
        x_flat = x.view(-1, self.in_features)
        
        Y = torch.zeros(B, self.out_features, dtype=x.dtype, device=x.device)
        
        if CUDA_AVAILABLE and x.is_cuda:
            bitnet_train_cuda.gemm_forward(self._W_packed, x_flat.contiguous(), Y)
            Y = Y * self._scale
        else:
            W_ternary = self._unpack_weights()
            Y = F.linear(x_flat, W_ternary * self._scale, None)
        
        if self.bias is not None:
            Y = Y + self.bias
        
        return Y.view(*x.shape[:-1], self.out_features)
    
    def quantize(self):
        """Freeze weights for inference"""
        with torch.no_grad():
            W_ternary, self._scale = quantize_to_ternary(self.weight)
            self._W_packed = pack_ternary(W_ternary)
            self._quantized = True
        return self
    
    def unquantize(self):
        """Return to training mode"""
        self._quantized = False
        self._W_packed = None
        self._scale = None
        return self
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, quantized={self._quantized}'


# Utility to convert existing model
def convert_to_bitlinear(model, inplace=True):
    """
    Convert all nn.Linear layers to BitLinearTrain.
    
    Usage:
        model = AutoModelForCausalLM.from_pretrained(...)
        model = convert_to_bitlinear(model)
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = BitLinearTrain(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype
            )
            new_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_layer.bias.data = module.bias.data.clone()
            setattr(model, name, new_layer)
        else:
            convert_to_bitlinear(module, inplace=True)
    
    return model
