# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import List, Optional, Tuple, Union
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTN_AVAILABLE = True
except:
    FLEX_ATTN_AVAILABLE = False
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel
from transformers.cache_utils import Cache

from sgl_kernel import (
    rmsnorm as rmsnorm_sgl_impl,
    gemma_rmsnorm as gemma_rmsnorm_sgl_impl,
    rotary_embedding as rotary_embedding_sgl_impl,
    silu_and_mul as silu_and_mul_sgl_impl,
)

@torch._dynamo.allow_in_graph
def rmsnorm_sgl(*args, **kwargs):
    return rmsnorm_sgl_impl(*args, **kwargs)
@torch._dynamo.allow_in_graph
def gemma_rmsnorm_sgl(*args, **kwargs):
    return gemma_rmsnorm_sgl_impl(*args, **kwargs)
@torch._dynamo.allow_in_graph
def rotary_embedding_sgl(*args, **kwargs):
    return rotary_embedding_sgl_impl(*args, **kwargs)
@torch._dynamo.allow_in_graph
def silu_and_mul_sgl(*args, **kwargs):
    return silu_and_mul_sgl_impl(*args, **kwargs)
KNOB = {
    "gemma_rmsnorm": False,
    "rmsnorm": False,
    "rotary_embedding": False,
    "llama_block": False,
    "swiglu": False,
    "layer_fusion": False,
    "cudagraph": False,
}


from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "GemmaRMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "LLaDABlock",
    "LLaDASequentialBlock",
    "LLaDAModel",
    "LLaDAOutput",
    "LLaDAGenerateOutput",
]

log = logging.getLogger(__name__)

@torch.compile()
def scaled_dot_product_attention(q, k, v, mask=None, attn_mask=None, dropout_p=0.0, is_causal=False):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: ModelConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param d: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            # for att_proj (same as QKV), ff_proj
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            # for attn_out, ff_out
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            # positional embeddings (wpe)
            # token embeddings (wte)
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            # final output (ff_out)
            std = config.d_model**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.gemma_rms:
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNormOri(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

class RMSLayerNormSgl(LayerNormBase):
    """
    RMS layer norm using SGL kernel implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None or self.bias is not None:
            raise RuntimeError("SGL kernel expects weight parameter and no bias")

        # SGL kernel expects 2D input: 
        #   (batch_size, seq_len, hidden_dim) -> (batch_size * seq_len, hidden_dim)
        original_shape = x.shape
        x = x.view(-1, x.shape[-1]).contiguous()

        output = rmsnorm_sgl(x, self.weight, self.eps)

        output = output.view(original_shape)
        return output

RMSLayerNorm = RMSLayerNormSgl if KNOB["rmsnorm"] else RMSLayerNormOri


class GemmaRMSLayerNormOri(LayerNormBase):
    """
    Gemma RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return x * (1 + self.weight) + self.bias
            else:
                return x * (1 + self.weight)
        else:
            return x

class GemmaRMSLayerNormSgl(LayerNormBase):
    """
    Gemma RMS layer norm using SGL kernel implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None or self.bias is not None:
            raise RuntimeError("SGL kernel expects weight parameter and no bias")
        
        # SGL kernel expects 2D input: 
        #   (batch_size, seq_len, hidden_dim) -> (batch_size * seq_len, hidden_dim)
        original_shape = x.shape
        x = x.view(-1, x.shape[-1]).contiguous()

        output = gemma_rmsnorm_sgl(x, self.weight, self.eps)

        output = output.view(original_shape)
        return output

GemmaRMSLayerNorm = GemmaRMSLayerNormSgl if KNOB["gemma_rmsnorm"] else GemmaRMSLayerNormOri


class RotaryEmbeddingOri(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.rope_theta = config.rope_theta
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))

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
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                block_end_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)

            # Build tensor indices instead of using .item()
            if block_end_index is None:
                start = key_len - query_len
                end = key_len
            else:
                # block_end_index is a tensor; keep ops tensor-based
                start = (block_end_index - query_len)
                end = block_end_index

            # Make an index tensor [start, ..., end-1] on the right device/dtype
            idx = torch.arange(start, end, device=q_.device, dtype=torch.long)

            # Use index_select on the sequence dimension (dim=2)
            pos_sin_slice = pos_sin.index_select(2, idx)
            pos_cos_slice = pos_cos.index_select(2, idx)

            q_ = self.apply_rotary_pos_emb(pos_sin_slice, pos_cos_slice, q_)
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)

            # print(idx.shape, q_.shape, k_.shape, idx.dtype, q_.dtype, k_.dtype)

        return q_.type_as(q), k_.type_as(k)

class RotaryEmbeddingSgl(nn.Module):
    """
    Rotary positional embeddings (RoPE) using SGL kernel implementation.
    CUDA graph compatible version.
    """
    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.head_size = config.d_model // config.n_heads
        self.rotary_dim = self.head_size
        self.max_position_embeddings = config.max_sequence_length
        self.base = config.rope_theta
        self.max_batch_size = getattr(config, 'max_batch_size', 1)  # Default max batch

        # SGL init of cache
        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        # Pre-allocate buffers for CUDA graph compatibility
        # These will be resized dynamically but avoid allocation during forward
        self.register_buffer("q_buffer", torch.zeros(
            self.max_batch_size * config.n_heads,
            config.max_sequence_length,
            self.head_size,
            dtype=torch.float16
        ), persistent=False)
        self.register_buffer("k_buffer", torch.zeros(
            self.max_batch_size * config.effective_n_kv_heads,
            config.max_sequence_length,
            self.head_size,
            dtype=torch.float16
        ), persistent=False)
        self.register_buffer("idx_q_buffer", torch.zeros(
            self.max_batch_size * config.n_heads,
            config.max_sequence_length,
            dtype=torch.long
        ), persistent=False)
        self.register_buffer("idx_k_buffer", torch.zeros(
            self.max_batch_size * config.effective_n_kv_heads,
            config.max_sequence_length,
            dtype=torch.long
        ), persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache
    
    def forward(
        self,
        q: torch.Tensor,  # [B, n_heads, q_len, head_size]
        k: torch.Tensor,  # [B, n_kv_heads, k_len, head_size]
        block_end_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cos_sin_cache.dtype != q.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(q.dtype)

        query_len, key_len = q.shape[-2], k.shape[-2]

        # Calculate start/end positions (these are always ints)
        if block_end_index is None:
            start = key_len - query_len
            end = key_len
        else:
            # Note: block_end_index should be passed as Python int, not tensor
            start = block_end_index - query_len
            end = block_end_index

        original_shape_q, original_shape_k = q.shape, k.shape
        batch_size_q = q.shape[0] * q.shape[1]  # B * n_heads
        batch_size_k = k.shape[0] * k.shape[1]  # B * n_kv_heads

        q_reshaped = q.view(-1, q.shape[-2], q.shape[-1])
        k_reshaped = k.view(-1, k.shape[-2], k.shape[-1])

        # Use pre-allocated buffers - slice to current sizes
        q_zeros = self.q_buffer[:batch_size_q, :query_len, :self.head_size].contiguous()
        k_zeros = self.k_buffer[:batch_size_k, :key_len, :self.head_size].contiguous()

        # Fill buffers with zeros (in-place to avoid allocation)
        q_zeros.zero_()
        k_zeros.zero_()

        # Create position indices using pre-allocated buffers
        idx_q_slice = self.idx_q_buffer[:batch_size_q, :query_len]
        idx_k_slice = self.idx_k_buffer[:batch_size_k, :key_len]

        # Fill index buffers - vectorized operation using broadcasting
        idx_q_slice[:] = torch.arange(start, end, device=q.device, dtype=torch.long).unsqueeze(0)
        idx_k_slice[:] = torch.arange(0, key_len, device=k.device, dtype=torch.long).unsqueeze(0)

        rotary_embedding_sgl(
            idx_q_slice.contiguous(),
            q_reshaped.contiguous(),
            q_zeros,
            self.head_size,
            self.cos_sin_cache,
            True,  # Llama uses this
        )
        rotary_embedding_sgl(
            idx_k_slice.contiguous(),
            k_zeros,
            k_reshaped.contiguous(),
            self.head_size,
            self.cos_sin_cache,
            True,  # Llama uses this
        )

        q = q_reshaped.view(original_shape_q)
        k = k_reshaped.view(original_shape_k)

        return q, k

RotaryEmbedding = RotaryEmbeddingSgl if KNOB["rotary_embedding"] else RotaryEmbeddingOri


class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.silu:
            return cast(Activation, SiLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0

class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0

class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5

class GeluAndMul(nn.Module):
    def __init__(self):
        """placeholder"""
        raise NotImplementedError

def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(seq_len: int, config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)

    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)

    # shape: (n_heads,)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore


class LLaDABlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)

        self.flash_attn_func = None
        if config.flash_attention:
            log.info(f"Flash attention enabled: {config.flash_attention}")
            try:
                from flash_attn import flash_attn_func  # type: ignore

                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass

        # CUDA graph compatibility: pre-allocate KV cache buffers
        # These avoid torch.cat() allocations during forward pass
        max_batch_size = getattr(config, 'max_batch_size', 1)
        self.register_buffer("kv_cache_buffer_k", torch.zeros(
            max_batch_size,
            config.effective_n_kv_heads,
            config.max_sequence_length,
            config.d_model // config.n_heads,
            dtype=torch.bfloat16
        ), persistent=False)
        self.register_buffer("kv_cache_buffer_v", torch.zeros(
            max_batch_size,
            config.effective_n_kv_heads,
            config.max_sequence_length,
            config.d_model // config.n_heads,
            dtype=torch.bfloat16
        ), persistent=False)

        self.layer_past_empty = True  # For CUDA graph compatibility

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=False
            )
            return r.transpose(1, 2)
        else:
            # torch's sdpa doesn't support GQA, so we're doing this
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            # Modify: MDM set causal to False, and with no attn_mask.
            return scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None: #self.q_norm: None, self.k_norm: None
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)


        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        # self.config.n_heads: 32
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is None:
            past_len = 0
            self.layer_past_empty = True
        else:
            past_len = layer_past[0].shape[-2]
            if self.layer_past_empty:
                self.kv_cache_buffer_k[:B, :, :past_len, :] = layer_past[0]
                self.kv_cache_buffer_v[:B, :, :past_len, :] = layer_past[1]
                self.layer_past_empty = False

        new_len = k.shape[-2]
        total_len = past_len + new_len
        self.kv_cache_buffer_k[:B, :, past_len:total_len, :] = k
        self.kv_cache_buffer_v[:B, :, past_len:total_len, :] = v
        k, v = self.kv_cache_buffer_k[:B, :, :total_len, :], self.kv_cache_buffer_v[:B, :, :total_len, :]

        # if layer_past is not None:
        #     past_key, past_value = layer_past
        #     if replace_position is None:
        #         # CUDA graph compatible: use pre-allocated buffer instead of torch.cat
        #         # torch.cat allocates new memory which breaks CUDA graphs
        #         past_len = past_key.shape[-2]
        #         new_len = k.shape[-2]
        #         total_len = past_len + new_len
        #         # print(past_len, new_len, total_len)

        #         # Use pre-allocated buffer and copy data in-place
        #         k_combined = self.kv_cache_buffer_k[:B, :, :total_len, :]
        #         v_combined = self.kv_cache_buffer_v[:B, :, :total_len, :]

        #         # Copy past and new KV into buffer
        #         if self.layer_past_empty:
        #             k_combined[:, :, :past_len, :] = past_key
        #             v_combined[:, :, :past_len, :] = past_value
        #             self.layer_past_empty = False

        #         k_combined[:, :, past_len:total_len, :] = k
        #         v_combined[:, :, past_len:total_len, :] = v

        #         k = k_combined
        #         v = v_combined
        #     else:
        #         raise RuntimeError("Batched replace_position is not supported without CUDA graph compatibility")
                
        #         # TODO: CUDA graph incompatible - requires batch_size=1 and vectorization
        #         # This path uses .nonzero() which causes CPU sync
        #         # For now, only batch_size=1 is supported with replace_position
        #         # k shape is [B, n_kv_h, selected_length, hs]
        #         # replace_position shape is [B, L], where L contains 0s and 1s, 0 means no replacement, 1 means replace, with selected_length number of 1s
        #         # past_key shape is [B, n_kv_h, L, hs]
        #         # Replace selected_length number of 1s in past_key with k

        #         # Handle batched replace_position correctly
        #         B = replace_position.shape[0]
        #         for batch_idx in range(B):
        #             # Get indices for this batch
        #             batch_replace_indices = replace_position[batch_idx].nonzero(as_tuple=True)[0]
        #             if len(batch_replace_indices) > 0:
        #                 # Replace positions in past_key and past_value for this batch
        #                 past_key[batch_idx, :, batch_replace_indices] = k[batch_idx, :, :len(batch_replace_indices)]
        #                 past_value[batch_idx, :, batch_replace_indices] = v[batch_idx, :, :len(batch_replace_indices)]

        #         k = past_key
        #         v = past_value
        # else:
        #     self.layer_past_empty = True

        present = (k, v) if use_cache else None #present: None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            # Apply rotary embeddings.
            if replace_position is None:
                q, k = self.rotary_emb(q, k)
            else:
                # For batched replace_position, use the maximum position across all batches
                max_replace_pos = replace_position.nonzero(as_tuple=True)[1].max() + 1 if replace_position.any() else key_len
                q, k = self.rotary_emb(q, k, max_replace_pos)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
        )
        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # end.record()
        # torch.cuda.synchronize()
        # elapsed_time = start.elapsed_time(end)
        # print(f"Layer {self.layer_id} Attention time: {elapsed_time} ms")
        # Apply output projection.
        return self.attn_out(att), present


    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> LLaDABlock:
        if config.block_type == BlockType.sequential:
            return LLaDASequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return LLaDALlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")

class LLaDALlamaBlockOri(LLaDABlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )
        # new add
        self.up_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)  # new add

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        x_normed = self.attn_norm(x) #x:torch.Size([2, 168, 4096])
        q = self.q_proj(x_normed) #q:torch.Size([2, 168, 4096])
        k = self.k_proj(x_normed) #k:torch.Size([2, 168, 4096])
        v = self.v_proj(x_normed) #v:torch.Size([2, 168, 4096])
        # attention_bias: None
        # layer_past: None
        # use_cache: False
        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache,replace_position=replace_position
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache,replace_position=replace_position)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x, x_up = self.ff_proj(x), self.up_proj(x) # new add
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = x * x_up # new add
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache

class LLaDALlamaBlockSgl(LLaDABlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )
        # new add
        self.up_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

        # CUDA graph compatibility: pre-fuse weights at initialization
        # Register as buffer so it's part of state_dict but not trainable
        self.fused_qkv: Optional[torch.nn.Parameter] = None  # Could be used for QKV fusion
        self.fused_ffn: Optional[torch.nn.Parameter] = None  # Will be initialized after weights are loaded

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)  # new add

    def fuse_weights_for_cudagraph(self):
        """
        Pre-fuse weights for CUDA graph compatibility.
        Call this after model initialization and before CUDA graph capture.
        """
        if self.fused_qkv is None:
            fused_weight = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
            self.fused_qkv = nn.Parameter(fused_weight)
        if self.fused_ffn is None:
            fused_weight = torch.cat([self.ff_proj.weight, self.up_proj.weight], dim=0)
            self.fused_ffn = nn.Parameter(fused_weight)


    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        x_normed = self.attn_norm(x) #x:torch.Size([2, 168, 4096])
        if KNOB["layer_fusion"] and self.fused_qkv is not None:
            qkv_fused = F.linear(x_normed, self.fused_qkv, bias=None)  # Single matmul
            q, k, v = qkv_fused.chunk(3, dim=-1)
        else:
            q = self.q_proj(x_normed) #q:torch.Size([2, 168, 4096])
            k = self.k_proj(x_normed) #k:torch.Size([2, 168, 4096])
            v = self.v_proj(x_normed) #v:torch.Size([2, 168, 4096])
        # attention_bias: None
        # layer_past: None
        # use_cache: False
        # Get attention scores.

        # end.record()
        # torch.cuda.synchronize()
        # elapsed_time = start.elapsed_time(end)
        # print(f"Layer {self.layer_id} LlamaBlock Projection time: {elapsed_time} ms")
        # start.record()

        # if self.fused_qkv is None:
        #     self.fused_qkv = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        # qkv_fused = F.linear(x_normed, self.fused_qkv, bias=None)  # Single matmul
        # q, k, v = qkv_fused.chunk(3, dim=-1)

        att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache,replace_position=replace_position)
        
        # end.record()
        # torch.cuda.synchronize()
        # elapsed_time = start.elapsed_time(end)
        # print(f"Layer {self.layer_id} LlamaBlock Attention time: {elapsed_time} ms")
        # start.record()

        x = x + self.dropout(att)
        og_x = x
        x = self.ff_norm(x)

        # CUDA graph compatible: use pre-fused weights (must call fuse_weights_for_cudagraph() first)
        if  KNOB["layer_fusion"] and self.fused_ffn is not None:
            # Use pre-fused weights - no dynamic allocation
            x_fused = F.linear(x, self.fused_ffn, bias=None)
            if KNOB["swiglu"]:
                x = silu_and_mul_sgl(x_fused)
            else:
                x, x_up = x_fused.chunk(2, dim=-1)
                x = F.silu(x) * x_up

        else:
            x_up = self.up_proj(x)
            x = self.ff_proj(x)
            x = F.silu(x) * x_up

        x = self.ff_out(x)
        # x = self.ff_out(out_x)

        x = self.dropout(x)
        x = og_x + x

        # end.record()
        # torch.cuda.synchronize()
        # elapsed_time = start.elapsed_time(end)
        # print(f"Layer {self.layer_id} LlamaBlock FFN time: {elapsed_time} ms")

        return x, cache

LLaDALlamaBlock = LLaDALlamaBlockSgl if KNOB["llama_block"] else LLaDALlamaBlockOri


class LLaDAOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor]]
    """
    Hidden states from each block.
    """


class LLaDAGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


class LLaDABlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if (
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                    and block_idx % 2 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                    and block_idx % 3 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                    and block_idx % 4 == 0
                )
            ):
                # shape: (batch_size, seq_len, d_model)
                x, cache = self._activation_checkpoint_fn(  # type: ignore
                    block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache
                )
            else:
                # shape: (batch_size, seq_len, d_model)
                x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return x, attn_key_values

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy)


class LLaDAModel(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.alibi and self.config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise Exception("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [LLaDABlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                LLaDABlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None
        # Warm up cache.
        if self.config.alibi:
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))

        # CUDA graph compatibility: pre-allocate output buffers
        max_batch_size = getattr(config, 'max_batch_size', 1)
        self.register_buffer("logits_buffer", torch.zeros(
            max_batch_size,
            config.max_sequence_length,
            config.embedding_size or config.vocab_size,
            dtype=torch.float32
        ), persistent=False)


    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        print("[DEBUG] Initializing model parameters...")  # Debug print
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.
        init_weights(
            self.config,
            self.transformer.wte,  # type: ignore
            std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)  # type: ignore

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)  # type: ignore

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def fuse_weights_for_cudagraph(self):
        """
        Pre-fuse all layer weights for CUDA graph compatibility.
        Call this after model initialization and before CUDA graph capture.
        """
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                if hasattr(block, 'fuse_weights_for_cudagraph'):
                    block.fuse_weights_for_cudagraph()
        else:
            for block_group in self.transformer.block_groups:
                for block in block_group:
                    if hasattr(block, 'fuse_weights_for_cudagraph'):
                        block.fuse_weights_for_cudagraph()

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (alibi_bias := self.__cache.get("alibi_attention_bias")) is not None and alibi_bias.shape[
            -1
        ] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    @torch.compile
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        replace_position: Optional[torch.Tensor] = None,
    ) -> LLaDAOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        # Add Basic MDM Model config check
        assert not self.config.alibi, "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        # assert (past_key_values is None and not use_cache), "The kvcache is not suppotred for MDM."

        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None and 0.0 in attention_mask:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if (
                    (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                        and block_idx % 2 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                        and block_idx % 3 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                        and block_idx % 4 == 0
                    )
                ):
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = self._activation_checkpoint_fn(
                        block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache,replace_position=replace_position
                    )
                else:
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache,replace_position=replace_position)
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(
                    x, attention_bias=attention_bias, layers_past=layers_past, use_cache=use_cache
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return LLaDAOutput(logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)  # type: ignore[arg-type]


def create_model_config_from_pretrained_config(config: LLaDAConfig):
    """
    Utility function
    """

    kwargs = {}
    for field in fields(ModelConfig):
        kwargs[field.name] = getattr(config, field.name)

    model_config = ModelConfig(**kwargs)
    return model_config


class LLaDAModelLM(PreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """

    config_class = LLaDAConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDABlock", "LLaDASequentialBlock", "LLaDALlamaBlock"]

    def __init__(self, config: LLaDAConfig, model: Optional[LLaDAModel] = None, init_params: bool = False):
        super().__init__(config)

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            self.model = LLaDAModel(model_config, init_params=init_params)
        else:
            self.model = model

        # CUDA graph support - multiple graphs for different seq lengths
        self.use_cudagraph = KNOB["cudagraph"]  # Set to True to enable automatic CUDA graph capture
        self._cuda_graphs = {}  # Dict mapping (batch, seq_len, past_len) -> (graph, static_inputs, static_outputs)
        self._prepared_for_cudagraph = False

    def prepare_for_cudagraph(self, batch_size: int, seq_len: int, device: torch.device = None):
        """
        Prepare model for CUDA graph capture by:
        1. Fusing weights
        2. Warming up all buffers
        3. Running a dummy forward pass

        Args:
            batch_size: The batch size that will be used for CUDA graph
            seq_len: The sequence length that will be used for CUDA graph
            device: The device to use (defaults to model's device)
        """
        if device is None:
            device = next(self.parameters()).device

        print(f"[CUDA Graph Prep] Fusing weights for all layers...")
        # Step 1: Fuse weights
        self.model.fuse_weights_for_cudagraph()

        print(f"[CUDA Graph Prep] Running warmup forward pass (batch={batch_size}, seq_len={seq_len})...")
        # Step 2: Warmup with dummy data
        dummy_input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=device)

        # Run forward pass to initialize all buffers
        with torch.no_grad():
            _ = self.model.forward(
                input_ids=dummy_input_ids,
                use_cache=False,
                output_hidden_states=False,
            )

        print(f"[CUDA Graph Prep] Model ready for CUDA graph capture!")
        return True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        replace_position: Optional[torch.Tensor] = None,  # This is a hack mitigation of an issue in transformers `4.39.x`
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in LLaDA")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # CUDA graph path - lazily capture separate graph for each (batch, seq_len, past_len)
        if self.use_cudagraph and not self.training:
            batch_size, seq_len = input_ids.shape
            past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
            graph_key = (batch_size, seq_len, past_len)

            if graph_key not in self._cuda_graphs:
                # First time seeing this configuration - prepare and capture graph
                if not self._prepared_for_cudagraph:
                    print("[CUDA Graph] Preparing model (first time only)...")
                    self.prepare_for_cudagraph(batch_size, seq_len, device=input_ids.device)
                    self._prepared_for_cudagraph = True

                print(f"[CUDA Graph] Capturing new graph for batch={batch_size}, seq={seq_len}, past={past_len}...")

                # Create static input tensors for this configuration
                static_input_ids = input_ids.clone()

                # Create static past_key_values (deep copy the structure)
                if past_key_values is not None:
                    static_past_key_values = []
                    for layer_past in past_key_values:
                        static_past_key_values.append(
                            (layer_past[0].clone(), layer_past[1].clone())
                        )
                else:
                    static_past_key_values = None

                cuda_graph = torch.cuda.CUDAGraph()

                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = self.model.forward(
                            input_ids=static_input_ids,
                            input_embeddings=inputs_embeds,
                            attention_mask=attention_mask,
                            attention_bias=attention_bias,
                            past_key_values=static_past_key_values,
                            use_cache=use_cache,
                            output_hidden_states=output_hidden_states,
                            replace_position=replace_position,
                        )

                # Capture
                with torch.cuda.graph(cuda_graph):
                    static_outputs = self.model.forward(
                        input_ids=static_input_ids,
                        input_embeddings=inputs_embeds,
                        attention_mask=attention_mask,
                        attention_bias=attention_bias,
                        past_key_values=static_past_key_values,
                        use_cache=use_cache,
                        output_hidden_states=output_hidden_states,
                        replace_position=replace_position,
                    )

                # Store graph and static tensors
                self._cuda_graphs[graph_key] = (cuda_graph, static_input_ids, static_past_key_values, static_outputs)
                print(f"[CUDA Graph] Graph captured! Total graphs: {len(self._cuda_graphs)}")

            # Get the cached graph for this configuration
            cuda_graph, static_input_ids, static_past_key_values, static_outputs = self._cuda_graphs[graph_key]

            # Replay graph - copy inputs to static tensors
            static_input_ids.copy_(input_ids)
            if past_key_values is not None and static_past_key_values is not None:
                for i, layer_past in enumerate(past_key_values):
                    static_past_key_values[i][0].copy_(layer_past[0])
                    static_past_key_values[i][1].copy_(layer_past[1])

            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            cuda_graph.replay()
            # end.record()
            # torch.cuda.synchronize()
            # elapsed_time = start.elapsed_time(end)
            # print(f"[CUDA Graph] Graph replay time: {elapsed_time} ms")
            # Use static outputs
            outputs = static_outputs
        else:
            # Regular forward pass (no CUDA graph)
            outputs = self.model.forward(
                input_ids=input_ids,
                input_embeddings=inputs_embeds,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                replace_position=replace_position,
            )

        # Extract outputs (works for both CUDA graph and regular path)
        logits = outputs.logits
        hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:
            import warnings
            warnings.warn("Note that for LLaDA, you cannot calculate the loss here.", UserWarning)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs


    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        if self.config.weight_tying:
            self.model.transformer.ff_out = self.model.transformer.wte

# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModel.register(LLaDAConfig, LLaDAModelLM)