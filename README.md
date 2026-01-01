# WARP BITNET

**Ultra-fast 1.58-bit (ternary) GEMV kernels for BitNet model inference.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-blue.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

## 🚀 Performance

### Open Source (Lite Kernel)

| Size | cuBLAS FP16 | Lite | Speedup | Memory |
|------|-------------|------|---------|--------|
| 4096×4096 | 0.035ms | 0.014ms | **2.5x** | **8x smaller** |
| 8192×8192 | 0.300ms | 0.120ms | **2.5x** | **8x smaller** |
| 14336×4096 | 0.259ms | 0.100ms | **2.6x** | **8x smaller** |

### Commercial (Pro Kernel)

| Size | cuBLAS FP16 | Pro | Speedup | Bandwidth |
|------|-------------|-----|---------|----------|
| 4096×4096 | 0.035ms | 0.018ms | **1.96x** | 119 GB/s |
| 8192×8192 | 0.300ms | 0.058ms | **5.14x** | 144 GB/s |
| 14336×4096 | 0.259ms | 0.042ms | **6.13x** | 175 GB/s |
| 28672×8192 | 1.036ms | 0.182ms | **5.70x** | 162 GB/s |

**Summary:**
- Peak Speedup: **6.13x** over cuBLAS FP16
- Average Speedup: **4.75x**
- Peak Bandwidth: **175 GB/s** (30% of theoretical peak)
- Memory Compression: **8x** (2-bit vs 16-bit)

**Real-world impact:** 140 GB → 14 GB for Llama-3-70B weights!

## 📦 Installation

```bash
git clone https://github.com/yourname/warp-bitnet.git
cd warp-bitnet
pip install -e .
```

**Requirements:**
- CUDA 12.0+
- PyTorch 2.0+
- GPU with compute capability 8.0+ (Ampere, Ada Lovelace, Hopper)

## 🔧 Usage

```python
import torch
import warp_bitnet_cuda

# Pack ternary weights {-1, 0, +1} into int32
def pack_ternary(W):
    """Pack 16 ternary weights per int32"""
    M, N = W.shape
    W_packed = torch.zeros(M, N // 16, dtype=torch.int32, device=W.device)
    for k in range(16):
        bits = torch.where(W[:, k::16] == 1, 1,
               torch.where(W[:, k::16] == -1, 2, 0))
        W_packed |= bits.int() << (k * 2)
    return W_packed

# Create ternary weights
W = torch.randint(-1, 2, (4096, 4096), device='cuda', dtype=torch.float32)
W_packed = pack_ternary(W)

# Input and output tensors
X = torch.randn(4096, dtype=torch.float16, device='cuda')
Y = torch.zeros(4096, dtype=torch.float16, device='cuda')

# GEMV: Y = W @ X
warp_bitnet_cuda.gemv(W_packed, X, Y)
```

## 🧮 How It Works

BitNet uses ternary weights {-1, 0, +1} which we pack efficiently:

```
Encoding: -1 → 0b10, 0 → 0b00, +1 → 0b01
16 weights → 32 bits → 1 int32
Memory: 2 bits per weight = 8x compression vs FP16
```

Our CUDA kernel uses the bit-trick for ultra-fast decoding:
```cuda
// Decode in just 3 ops!
int coeff = (bits & 1) - (bits >> 1);  
// 0b00 → 0, 0b01 → +1, 0b10 → -1
```

## 📊 Benchmarks

Run the comprehensive benchmark:

```bash
python prove_benchmark.py
```

This tests against real model architectures:
- Llama-3 (8B, 70B)
- Mistral-7B
- Qwen2.5-72B
- DeepSeek-V2 MoE

## 🏢 Commercial License

**Need more speed?** Our Pro kernel delivers up to **6.13x speedup** at **175 GB/s** bandwidth:

- Optimized thread block configuration
- Vectorized half2 memory loads
- L2-friendly tiling strategy
- Near-zero dequantization overhead

**Numerical Accuracy:**

| Size | Max Error | Status |
|------|-----------|--------|
| 4096×4096 | 0.0000 | ✓ |
| 14336×4096 | 0.0039 | ✓ |
| 28672×8192 | 0.0625 | ✓ |

All tests pass with < 0.001% relative error.

Contact: jorgeruizwilliams@gmail.com for commercial licensing.

## 🎓 Training Your Own Models

Training is **fully open source**! Use our training module to create your own 1.58-bit models:

```python
from bit_linear_train import BitLinearTrain, convert_to_bitlinear

# Convert any model to trainable BitNet
model = YourModel()
model = convert_to_bitlinear(model)

# Train with standard PyTorch
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()  # STE: gradients flow through quantization
    optimizer.step()

# Deploy with fast inference
for layer in model.modules():
    if hasattr(layer, 'quantize'):
        layer.quantize()
```

**What's included (open source):**
- `bit_linear_train.py` - Drop-in replacement for `nn.Linear`
- `cuda/bitnet_train.cu` - GEMM kernels for batched training
- `cuda/bitnet_lite.cu` - Inference kernel (2.5x speedup)
- Full STE (Straight-Through Estimator) backward pass

**What requires license (proprietary):**
- Pro inference kernel (6.13x speedup)
- INT2 inference kernel (4-level quantization, same optimizations)
- Optimized deployment configurations

## 📚 Citation

```bibtex
@software{warp_bitnet,
  title = {WARP BITNET: Ultra-fast 1.58-bit GEMV Kernels},
  author = {Jorge L. Ruiz Williams},
  year = {2025},
  url = {https://github.com/NaNZeta/warp-bitnet}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔢 INT2 Support (Experimental)

We also provide an INT2 kernel for true 4-level quantization:

```
Ternary (1.58-bit): 3 levels {-1, 0, +1}
INT2 (2-bit):       4 levels {-1.5, -0.5, +0.5, +1.5}
```

Same memory footprint, but 4 levels can provide slightly better model quality for some architectures.

```python
import bitnet_int2_cuda
bitnet_int2_cuda.gemv_int2(W_packed, X, Y)  # Without scale
bitnet_int2_cuda.gemv_int2_scaled(W_packed, X, scales, Y)  # With per-row scale
```

---

**WARP BITNET** - Making 1.58-bit models fly! 🚀
