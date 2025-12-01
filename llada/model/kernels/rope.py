import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Tuple, Optional
from torch import einsum


@triton.jit
def fused_rope_kernel(
    # Input/Output pointers
    input_ptr,
    output_ptr,
    # Sin/Cos embeddings
    sin_ptr, cos_ptr,
    # Strides for input/output
    stride_batch, stride_head, stride_seq, stride_dim,
    # Strides for sin/cos (shape: [1, 1, seq_len, head_dim])
    stride_sin_seq, stride_sin_dim,
    # Sequence offset
    seq_offset,  # Starting position in the sin/cos cache
    # Dimensions
    head_dim,
    seq_len,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that applies rotary embeddings to Q or K tensor.
    
    For each position, applies:
    - Rotate half: swap and negate pairs
    - Apply RoPE: out = x * cos + rotate_half(x) * sin
    """
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_seq = tl.program_id(2)
    
    # Check bounds
    if pid_seq >= seq_len:
        return
    
    # Base offset for this batch, head, sequence position
    base_offset = (
        pid_batch * stride_batch +
        pid_head * stride_head +
        pid_seq * stride_seq
    )
    
    # Sin/cos offset (accounting for sequence offset in cache)
    sincos_seq_idx = seq_offset + pid_seq
    sincos_offset = sincos_seq_idx * stride_sin_seq
    
    # Process head_dim in blocks
    for block_start in range(0, head_dim, BLOCK_SIZE):
        # Calculate dimension indices (process pairs)
        # RoPE works on pairs: (d_0, d_1), (d_2, d_3), etc.
        pair_idx = block_start + tl.arange(0, BLOCK_SIZE)
        mask = pair_idx < head_dim
        
        # Calculate the paired indices for rotate_half
        # For each pair (i, i+head_dim/2), we swap and negate
        half_dim = head_dim // 2
        is_first_half = pair_idx < half_dim
        
        # Get the paired index for rotation
        # First half: pair with second half (add half_dim)
        # Second half: pair with first half (subtract half_dim)
        pair_partner = tl.where(
            is_first_half,
            pair_idx + half_dim,
            pair_idx - half_dim
        )
        
        # Load input values
        x_ptrs = input_ptr + base_offset + pair_idx * stride_dim
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Load paired values for rotate_half
        x_pair_ptrs = input_ptr + base_offset + pair_partner * stride_dim
        x_pair = tl.load(x_pair_ptrs, mask=mask, other=0.0)
        
        # Compute rotate_half: swap pairs and negate second half
        # rotate_half swaps (x1, x2) -> (-x2, x1)
        rotated = tl.where(is_first_half, -x_pair, x_pair)
        
        # Load sin and cos values
        sin_ptrs = sin_ptr + sincos_offset + pair_idx * stride_sin_dim
        cos_ptrs = cos_ptr + sincos_offset + pair_idx * stride_sin_dim
        sin_val = tl.load(sin_ptrs, mask=mask, other=0.0)
        cos_val = tl.load(cos_ptrs, mask=mask, other=0.0)
        
        # Apply RoPE: out = x * cos + rotate_half(x) * sin
        output = x * cos_val + rotated * sin_val
        
        # Store output
        out_ptrs = output_ptr + base_offset + pair_idx * stride_dim
        tl.store(out_ptrs, output, mask=mask)


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    Optimized with fused Triton kernels.
    """

    def __init__(self, config, cache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.rope_theta = config.rope_theta
        
        # Determine init device
        if config.init_device is not None and config.init_device != "meta":
            init_device = torch.device(config.init_device)
        else:
            init_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.get_rotary_embedding(config.max_sequence_length, init_device)

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback rotate_half for compatibility (not used in fused kernel)"""
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Fallback apply_rotary_pos_emb for compatibility (not used in fused kernel)"""
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                block_end_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Handle precision conversion
        original_q_dtype = q.dtype
        original_k_dtype = k.dtype
        
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)

            # Calculate sequence offsets
            if block_end_index is None:
                q_seq_offset = key_len - query_len
                k_seq_offset = 0
            else:
                # Handle tensor-based block_end_index
                if isinstance(block_end_index, torch.Tensor):
                    block_end_index = block_end_index.item()
                q_seq_offset = block_end_index - query_len
                k_seq_offset = 0

            # Use fused Triton kernel
            q_out, k_out = self._apply_rope_fused(
                q_, k_, pos_sin, pos_cos, 
                q_seq_offset, k_seq_offset
            )

        return q_out.type_as(q), k_out.type_as(k)

    def _apply_rope_fused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_sin: torch.Tensor,
        pos_cos: torch.Tensor,
        q_seq_offset: int,
        k_seq_offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE using fused Triton kernel.
        
        Args:
            q: Query tensor [batch, n_heads, q_seq_len, head_dim]
            k: Key tensor [batch, n_heads, k_seq_len, head_dim]
            pos_sin: Sin embeddings [1, 1, max_seq_len, head_dim]
            pos_cos: Cos embeddings [1, 1, max_seq_len, head_dim]
            q_seq_offset: Offset into sin/cos cache for query
            k_seq_offset: Offset into sin/cos cache for key
        """
        batch_size, n_heads, q_seq_len, head_dim = q.shape
        _, _, k_seq_len, _ = k.shape
        
        # Allocate output tensors
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        
        # Determine block size
        BLOCK_SIZE = triton.next_power_of_2(min(head_dim, 128))
        
        # Launch kernel for Q
        grid_q = (batch_size, n_heads, q_seq_len)
        fused_rope_kernel[grid_q](
            # Input/Output
            q, q_out,
            # Sin/Cos
            pos_sin, pos_cos,
            # Strides
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            # Sin/Cos strides (shape is [1, 1, seq, dim])
            pos_sin.stride(2), pos_sin.stride(3),
            # Sequence offset
            q_seq_offset,
            # Dimensions
            head_dim, q_seq_len,
            # Block size
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Launch kernel for K
        grid_k = (batch_size, n_heads, k_seq_len)
        fused_rope_kernel[grid_k](
            # Input/Output
            k, k_out,
            # Sin/Cos
            pos_sin, pos_cos,
            # Strides
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            # Sin/Cos strides
            pos_sin.stride(2), pos_sin.stride(3),
            # Sequence offset
            k_seq_offset,
            # Dimensions
            head_dim, k_seq_len,
            # Block size
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return q_out, k_out


# Benchmark and testing code
if __name__ == "__main__":
    import time
    
    # Mock config for testing
    class MockConfig:
        def __init__(self):
            self.d_model = 4096
            self.n_heads = 32
            self.rope_theta = 10000.0
            self.max_sequence_length = 2048
            self.init_device = "cuda"
            self.rope_full_precision = True
    
    config = MockConfig()
    cache = {}
    
    # Create RoPE module
    rope = RotaryEmbedding(config, cache)
    
    # Test tensors
    batch_size = 2
    n_heads = 32
    q_seq_len = 128
    k_seq_len = 128
    head_dim = config.d_model // config.n_heads
    
    q = torch.randn(batch_size, n_heads, q_seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, n_heads, k_seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    # Warm up
    for _ in range(10):
        q_out, k_out = rope(q, k)
    torch.cuda.synchronize()
    
    # Benchmark
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        q_out, k_out = rope(q, k)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"Fused RoPE: {(end - start) / iterations * 1000:.3f} ms per iteration")
    print(f"Output shapes - Q: {q_out.shape}, K: {k_out.shape}")
    print(f"Output dtypes - Q: {q_out.dtype}, K: {k_out.dtype}")