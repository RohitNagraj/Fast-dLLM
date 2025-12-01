import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def rms_norm_forward_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    stride_x_row,
    stride_y_row,
    n_cols,
    eps,
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # Compute row pointers
    x_row_ptr = x_ptr + row_idx * stride_x_row
    y_row_ptr = y_ptr + row_idx * stride_y_row
    
    # Load data in blocks
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Accumulate variance in float32 for numerical stability
    variance = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        variance += tl.where(mask, x * x, 0.0)
    
    # Compute mean variance across the row
    var = tl.sum(variance, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply affine transformation
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x_normed = x * rstd
        
        if has_weight:
            weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
            x_normed = x_normed * weight
        
        if has_bias:
            bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
            x_normed = x_normed + bias
        
        tl.store(y_row_ptr + offsets, x_normed, mask=mask)


@triton.jit
def rms_norm_backward_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    dx_ptr,
    dweight_ptr,
    dbias_ptr,
    stride_row,
    n_cols,
    eps,
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    dy_row_ptr = dy_ptr + row_idx * stride_row
    x_row_ptr = x_ptr + row_idx * stride_row
    dx_row_ptr = dx_ptr + row_idx * stride_row
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # First pass: compute variance and rstd
    variance = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        variance += tl.where(mask, x * x, 0.0)
    
    var = tl.sum(variance, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Second pass: compute gradients
    dvar_sum = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        if has_weight:
            weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
            dy = dy * weight
        
        x_normed = x * rstd
        dvar_sum += tl.sum(tl.where(mask, dy * x_normed, 0.0), axis=0)
    
    # Third pass: compute dx
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        if has_weight:
            weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
            dy_weighted = dy * weight
        else:
            dy_weighted = dy
        
        x_normed = x * rstd
        dx = (dy_weighted - x_normed * dvar_sum / n_cols) * rstd
        
        tl.store(dx_row_ptr + offsets, dx, mask=mask)


class RMSLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        # Ensure contiguous
        x = x.contiguous()
        
        # Get shape info
        *batch_dims, n_cols = x.shape
        n_rows = x.numel() // n_cols
        
        # Flatten to 2D
        x_2d = x.view(n_rows, n_cols)
        
        # Allocate output
        y = torch.empty_like(x_2d)
        
        # Choose block size
        BLOCK_SIZE = triton.next_power_of_2(min(n_cols, 4096))
        
        # Launch kernel
        grid = (n_rows,)
        rms_norm_forward_kernel[grid](
            x_2d,
            y,
            weight if weight is not None else x_2d,  # dummy pointer if None
            bias if bias is not None else x_2d,      # dummy pointer if None
            x_2d.stride(0),
            y.stride(0),
            n_cols,
            eps,
            has_weight=weight is not None,
            has_bias=bias is not None,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Reshape back
        y = y.view_as(x)
        
        # Save for backward
        ctx.save_for_backward(x, weight, bias)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        
        return y
    
    @staticmethod
    def backward(ctx, dy):
        x, weight, bias = ctx.saved_tensors
        eps = ctx.eps
        BLOCK_SIZE = ctx.BLOCK_SIZE
        
        # Flatten
        *batch_dims, n_cols = x.shape
        n_rows = x.numel() // n_cols
        
        x_2d = x.contiguous().view(n_rows, n_cols)
        dy_2d = dy.contiguous().view(n_rows, n_cols)
        
        # Allocate gradient
        dx = torch.empty_like(x_2d)
        
        # Launch backward kernel
        grid = (n_rows,)
        rms_norm_backward_kernel[grid](
            dy_2d,
            x_2d,
            weight if weight is not None else x_2d,
            dx,
            x_2d,  # dummy for dweight
            x_2d,  # dummy for dbias
            x_2d.stride(0),
            n_cols,
            eps,
            has_weight=weight is not None,
            has_bias=bias is not None,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        dx = dx.view_as(x)
        
        # Compute weight and bias gradients if needed
        dweight = None
        dbias = None
        
        if weight is not None:
            # Compute dweight
            x_2d_f32 = x_2d.to(torch.float32)
            dy_2d_f32 = dy_2d.to(torch.float32)
            
            variance = (x_2d_f32 * x_2d_f32).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(variance + eps)
            x_normed = x_2d_f32 * rstd
            
            dweight = (dy_2d_f32 * x_normed).sum(dim=0).to(weight.dtype)
        
        if bias is not None:
            dbias = dy_2d.sum(dim=0).to(bias.dtype)
        
        return dx, dweight, dbias, None


class RMSLayerNorm(torch.nn.Module):
    """
    RMS layer norm, a simplified LayerNorm implementation with Triton kernel acceleration
    """

    def __init__(
        self,
        config,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        
        # Get epsilon from config if available
        self.eps = getattr(config, 'rms_norm_eps', eps)
        
        # Determine size
        if size is None:
            if hasattr(config, 'd_model'):
                size = config.d_model
            elif hasattr(config, 'hidden_size'):
                size = config.hidden_size
            else:
                raise ValueError("size must be specified or config must have d_model/hidden_size")
        
        self.size = size
        
        # Determine if we use affine transform
        if elementwise_affine is None:
            elementwise_affine = True
        
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(size))
            self.bias = torch.nn.Parameter(torch.zeros(size))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel if available and input is on CUDA
        if x.is_cuda and triton is not None:
            return RMSLayerNormFunction.apply(x, self.weight, self.bias, self.eps)
        
        # Fallback to PyTorch implementation
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


if __name__ == "__main__":
    import time
    
    print("Testing RMSLayerNorm Triton Implementation")
    print("=" * 60)
    
    # Create a simple config object
    class SimpleConfig:
        def __init__(self):
            self.rms_norm_eps = 1e-5
            self.d_model = 512
    
    config = SimpleConfig()
    
    # Test 1: Correctness test
    print("\n1. Correctness Test")
    print("-" * 60)
    
    batch_size, seq_len, hidden_size = 4, 128, 512
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
    
    # PyTorch reference implementation
    def reference_rms_norm(x, weight, bias, eps):
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            x = x.to(og_dtype)
        if weight is not None:
            x = weight * x
            if bias is not None:
                x = x + bias
        return x
    
    # Test with weight and bias
    layer = RMSLayerNorm(config, size=hidden_size, elementwise_affine=True).cuda().half()
    
    # Forward pass
    output_triton = layer(x)
    output_reference = reference_rms_norm(x, layer.weight, layer.bias, layer.eps)
    
    # Check correctness
    max_diff = (output_triton - output_reference).abs().max().item()
    mean_diff = (output_triton - output_reference).abs().mean().item()
    
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    print(f"Outputs match: {torch.allclose(output_triton, output_reference, rtol=1e-3, atol=1e-5)}")
    
    # Test 2: Backward pass test
    print("\n2. Backward Pass Test")
    print("-" * 60)
    
    x_grad = x.clone().requires_grad_(True)
    layer_grad = RMSLayerNorm(config, size=hidden_size, elementwise_affine=True).cuda().half()
    
    output = layer_grad(x_grad)
    loss = output.sum()
    loss.backward()
    
    print(f"Input gradient shape: {x_grad.grad.shape}")
    print(f"Weight gradient shape: {layer_grad.weight.grad.shape}")
    print(f"Bias gradient shape: {layer_grad.bias.grad.shape}")
    print(f"Gradients computed successfully: {x_grad.grad is not None}")
    
    # Test 3: Without affine transform
    print("\n3. Test Without Affine Transform")
    print("-" * 60)
    
    layer_no_affine = RMSLayerNorm(config, size=hidden_size, elementwise_affine=False).cuda().half()
    output_no_affine = layer_no_affine(x)
    output_ref_no_affine = reference_rms_norm(x, None, None, layer.eps)
    
    max_diff_no_affine = (output_no_affine - output_ref_no_affine).abs().max().item()
    print(f"Max difference (no affine): {max_diff_no_affine:.6e}")
    print(f"Outputs match: {torch.allclose(output_no_affine, output_ref_no_affine, rtol=1e-3, atol=1e-5)}")
    
    # Test 4: Performance benchmark
    print("\n4. Performance Benchmark")
    print("-" * 60)
    
    # Larger batch for benchmarking
    batch_size, seq_len, hidden_size = 32, 512, 1024
    x_large = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
    
    layer_bench = RMSLayerNorm(config, size=hidden_size, elementwise_affine=True).cuda().half()
    
    # Warmup
    for _ in range(10):
        _ = layer_bench(x_large)
    
    torch.cuda.synchronize()
    
    # Benchmark Triton
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        output = layer_bench(x_large)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_iters
    
    # Benchmark PyTorch reference
    start = time.time()
    for _ in range(num_iters):
        output_ref = reference_rms_norm(x_large, layer_bench.weight, layer_bench.bias, layer_bench.eps)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iters
    
    print(f"Input shape: {x_large.shape}")
    print(f"Triton time: {triton_time*1000:.4f} ms")
    print(f"PyTorch time: {pytorch_time*1000:.4f} ms")
    print(f"Speedup: {pytorch_time/triton_time:.2f}x")
    
    # Test 5: Different shapes
    print("\n5. Testing Different Shapes")
    print("-" * 60)
    
    test_shapes = [
        (2, 64, 256),
        (8, 256, 512),
        (16, 1024, 768),
        (1, 2048, 2048),
    ]
    
    for shape in test_shapes:
        x_test = torch.randn(*shape, device='cuda', dtype=torch.float16)
        hidden_size = shape[-1]
        layer_test = RMSLayerNorm(config, size=hidden_size, elementwise_affine=True).cuda().half()
        
        try:
            output = layer_test(x_test)
            output_ref = reference_rms_norm(x_test, layer_test.weight, layer_test.bias, layer_test.eps)
            matches = torch.allclose(output, output_ref, rtol=1e-3, atol=1e-5)
            print(f"Shape {shape}: {'✓ PASS' if matches else '✗ FAIL'}")
        except Exception as e:
            print(f"Shape {shape}: ✗ ERROR - {str(e)}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")