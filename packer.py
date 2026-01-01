import torch
import numpy as np

def pack_ternary_weights(weights):
    """
    Packs a ternary weight matrix ({-1, 0, 1}) into a 4-bit packed int32 tensor.
    
    Args:
        weights: Float tensor of shape (rows, cols) containing values {-1, 0, 1}.
                 Cols must be divisible by 8.
    
    Returns:
        packed_weights: Int32 tensor of shape (rows, cols // 8).
    """
    rows, cols = weights.shape
    assert cols % 8 == 0, "Columns must be divisible by 8 for packing"
    
    # Map {-1, 0, 1} to {2, 0, 1} (4-bit representation, only using 2 bits)
    # 0 -> 0
    # 1 -> 1
    # -1 -> 2
    
    w_int = weights.to(torch.int8)
    
    # Create the mapping
    w_mapped = torch.zeros_like(w_int, dtype=torch.int32)
    w_mapped[w_int == 1] = 1
    w_mapped[w_int == -1] = 2
    
    # Reshape to (rows, cols // 8, 8) to process chunks
    w_reshaped = w_mapped.view(rows, cols // 8, 8)
    
    packed = torch.zeros((rows, cols // 8), dtype=torch.int32, device=weights.device)
    
    # Pack 8 weights into one int32 (4 bits each)
    for i in range(8):
        packed |= (w_reshaped[:, :, i] << (4 * i))
        
    return packed

def unpack_ternary_weights(packed, original_shape):
    """
    Unpacks int32 tensor back to float ternary weights for verification.
    """
    rows, cols = original_shape
    packed_view = packed.view(rows, cols // 16, 1)
    
    unpacked = torch.zeros(original_shape, dtype=torch.float32, device=packed.device)
    
    # Unpack 8 weights from each int32 (4 bits each)
    unpacked = torch.zeros(original_shape, dtype=torch.float32, device=packed.device)
    
    for i in range(8):
        # Extract 4 bits
        mask = 0xF  # binary 1111
        val = (packed_view >> (4 * i)) & mask
        
        # Map back: 0->0, 1->1, 2->-1
        mapped_val = torch.zeros_like(val, dtype=torch.float32)
        mapped_val[val == 1] = 1.0
        mapped_val[val == 2] = -1.0
        
        # Place into output
        unpacked[:, i::8] = mapped_val.squeeze(-1)
        
    return unpacked

if __name__ == "__main__":
    # Test the packer
    print("Testing Packer...")
    rows, cols = 4, 32
    w = torch.randint(-1, 2, (rows, cols)).float()
    # Ensure strictly -1, 0, 1
    w[w == 2] = 0 # randint is exclusive on high end? no, low, high. 
    
    print(f"Original shape: {w.shape}")
    print(f"Original memory (FP32): {w.numel() * 4} bytes")
    
    packed = pack_ternary_weights(w)
    print(f"Packed shape: {packed.shape}")
    print(f"Packed memory (INT32): {packed.numel() * 4} bytes")
    print(f"Compression Ratio: {w.numel() * 4 / (packed.numel() * 4)}x")
    
    # Verify first element
    first_16 = w[0, :16]
    packed_val = packed[0, 0].item()
    print("\nFirst 16 weights:", first_16.tolist())
    print(f"Packed Int32: {packed_val:032b}")
    
    # Manual check of first weight
    w0 = first_16[0].item()
    extracted_w0_bits = packed_val & 3
    print(f"Weight 0: {w0}, Extracted Bits: {extracted_w0_bits:02b}")
    
    expected_bits = 0
    if w0 == 1: expected_bits = 1
    elif w0 == -1: expected_bits = 2
    
    if extracted_w0_bits == expected_bits:
        print("SUCCESS: First weight matches!")
    else:
        print("FAILURE: Packing logic error.")
