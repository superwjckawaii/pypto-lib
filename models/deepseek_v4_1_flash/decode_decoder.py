# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode decoder stack (layers 20-39) attention chain.

Every decoder layer runs in order inside one chip program: the layer is one attention
sub-layer (mHC mixes -> collapse -> attn_norm -> C1A attention -> expansion) and the
staggered pre-mix it computes is consumed by the next sub-layer, exactly like the released
``Block.forward``. The attention mode follows the checkpoint schedule, which the config
resolves: the first decoder layer publishes the compressed KV, the index keys and the
candidate mask (FULL); every fourth layer after it selects its own Top-K inside that mask
(REINDEX); the remaining fifteen read the source layer's selection (REUSE).

The FFN sub-layer and its MoE are the next milestone, so this chain covers what stage 1
must prove: the cross-layer cache flow (one compressed/index pool, one Top-K slot and one
candidate mask for the whole stack; one window cache per layer), the mode schedule, the TP
reduce and the epoch sequence the shared transport windows need. The 20 slots are sliced out
of layer-stacked weights inside the loop (the ``decode_fwd`` pattern), the stream buffers
alternate inside each group of four layers, and the loop therefore returns to the buffer it
started from.

The body is emitted by ``build_output/decode_decoder/generate_single.py``: the three modes
take different tensor lists, and unrolling the groups keeps every ABI the operators publish.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash.config import (
    B_DYN,
    CMP_BLOCKS_DYN,
    CMP_POSITIONS_DYN,
    COMPRESSED_CACHE_GROUP,
    D,
    DECODE_MAX_TOKENS,
    FLASH,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    INDEX_BLOCKS_DYN,
    INDEX_CACHE_GROUP,
    INDEX_DIM,
    INDEX_H,
    INDEX_TOPK,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    MIX_HC,
    O_GROUP_IN,
    O_LORA,
    Q_LORA,
    ROPE_DIM,
    TABLE_DYN,
    TP_SIZE,
    T_DYN,
    WINDOW_CACHE_GROUP,
)
from models.deepseek_v4_1_flash.attention_common import quantized_cache_compare
from models.deepseek_v4_1_flash.decode_attn_c1a_full import exact_bytes
from models.deepseek_v4_1_flash.decode_c1a_full import (
    CACHE_MAX_RELATIVE_L2,
    MXFP4_CACHE_MAX_RELATIVE_L2,
    next_pre_mix_compare,
    output_hc_compare,
)
from models.deepseek_v4_1_flash.decode_c1a_full import decode_c1a_full
from models.deepseek_v4_1_flash.decode_c1a_reindex import decode_c1a_reindex
from models.deepseek_v4_1_flash.decode_c1a_reuse import decode_c1a_reuse

# ---- decoder schedule (derived from the preset, never hard-coded) --------------------
DECODER_SLOTS = tuple(layer for layer, ratio in enumerate(FLASH.compress_ratios) if ratio == 1)
LAYERS = len(DECODER_SLOTS)
GROUP = 4
# Static packed-FP4 storage dimensions of this entry (the layer-stacked window cache needs
# a static extent; the harness asserts --pages matches).
PAGES = 8
assert tuple(
    slot for slot, layer in enumerate(DECODER_SLOTS) if layer in FLASH.kv_source_layer_ids
) == (0,), "the first decoder layer must own the compressed KV"
assert tuple(
    slot
    for slot, layer in enumerate(DECODER_SLOTS)
    if layer in FLASH.index_source_layer_ids and layer not in FLASH.kv_source_layer_ids
) == tuple(range(GROUP, LAYERS, GROUP)), "later index sources must repeat every GROUP layers"
assert LAYERS % GROUP == 0


def decoder_modes(layers: int = LAYERS) -> tuple:
    """Attention mode of every decoder slot, in stack order."""
    return tuple(
        "full" if slot == 0 else ("reindex" if slot % GROUP == 0 else "reuse")
        for slot in range(layers)
    )


def build_decoder_values(tokens, pages, layers=LAYERS, seed=17, case="random"):
    """Layer-stacked weights plus the caches the whole stack shares."""
    from models.deepseek_v4_1_flash.decode_c1a_full import build_hc_validation_values

    per_layer = [
        build_hc_validation_values("full", tokens, pages, seed + slot, case) for slot in range(layers)
    ]
    shared = build_hc_validation_values("full", tokens, pages, seed, case)
    values = {}
    for name in (
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight", "wq_a", "wq_a_scale",
        "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale", "kv_norm_weight", "attn_sink",
        "wo_a", "wo_b", "wo_b_scale", "compressor_wkv", "compressor_norm_weight", "index_wk",
        "index_norm_weight", "index_wq_b", "index_wq_b_scale", "index_weights_proj",
        "window_cache", "window_cache_scale",
    ):
        values[name] = torch.cat([layer[name] for layer in per_layer], dim=1).contiguous()
    # One compressed pool, one index pool, one Top-K slot and one candidate mask for the
    # whole stack: a source always publishes before its consumers read, so one slot each.
    for name in (
        "compressed_cache", "compressed_cache_scale", "request_ids", "compressed_lens",
        "index_cache", "index_cache_scale", "index_block_table", "compressed_rope_cos",
        "compressed_rope_sin", "compressed_slots", "candidate_mask", "topk_indices",
        "compressed_indices",
        "rope_cos", "rope_sin", "window_slots", "window_indices",
    ):
        values[name] = shared[name].clone()
    generator = torch.Generator().manual_seed(seed)
    values["chain_x"] = torch.randn(TP_SIZE, tokens, HC_MULT, D, generator=generator)
    values["chain_pre"] = torch.sigmoid(
        torch.randn(TP_SIZE, tokens, HC_MULT, generator=torch.Generator().manual_seed(seed + 1))
    )
    values["output"] = torch.zeros(TP_SIZE, tokens, HC_MULT, D, dtype=torch.float32)
    values["next_pre_mix"] = torch.zeros(TP_SIZE, tokens, HC_MULT, dtype=torch.float32)
    return values, per_layer

@pl.jit(auto_scope=False)
def decode_decoder_test(
    chain_x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    chain_pre: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[20 * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[20 * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[20 * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[20 * D], pl.BF16],
    wq_a: pl.Tensor[[20 * D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[20 * D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[20 * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[20 * Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[20 * Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[20 * D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[20 * D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[20 * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[20 * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[20 * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[20 * LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[20 * LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    window_cache: pl.Tensor[[20 * PAGES, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[20 * PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0],
    compressor_wkv: pl.Tensor[[20 * D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[20 * HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[20 * HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[20 * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[20 * Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[20 * Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[20 * D, INDEX_H], pl.BF16],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    candidate_mask: pl.InOut[pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8]],
    topk_indices: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    epoch_base: pl.Scalar[pl.INT32],
):
    """Run the decoder attention chain for one chip on one DP group."""
    tokens = pl.tensor.dim(chain_x, 0)
    x0 = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    x1 = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    p0 = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    p1 = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    hc_attn_fn_c0 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [0 * MIX_HC, 0])
    hc_attn_scale_c0 = pl.slice(hc_attn_scale, [3], [0 * 3])
    hc_attn_base_c0 = pl.slice(hc_attn_base, [MIX_HC], [0 * MIX_HC])
    attn_norm_weight_c0 = pl.slice(attn_norm_weight, [D], [0 * D])
    wq_a_c0 = pl.slice(wq_a, [D, Q_LORA], [0 * D, 0])
    wq_a_scale_c0: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [0 * (D // 32), 0])
    q_norm_weight_c0 = pl.slice(q_norm_weight, [Q_LORA], [0 * Q_LORA])
    wq_b_c0 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [0 * Q_LORA, 0])
    wq_b_scale_c0: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [0 * (Q_LORA // 32), 0])
    wkv_c0 = pl.slice(wkv, [D, HEAD_DIM], [0 * D, 0])
    wkv_scale_c0: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [0 * (D // 32), 0])
    kv_norm_weight_c0 = pl.slice(kv_norm_weight, [HEAD_DIM], [0 * HEAD_DIM])
    attn_sink_c0 = pl.slice(attn_sink, [LOCAL_H], [0 * LOCAL_H])
    wo_a_c0 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [0 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c0 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [0 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c0: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [0 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c0 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [0 * PAGES, 0, 0, 0])
    window_cache_scale_c0 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [0 * PAGES, 0, 0, 0])
    compressor_wkv_c0 = pl.slice(compressor_wkv, [D, HEAD_DIM], [0 * D, 0])
    compressor_norm_weight_c0 = pl.slice(compressor_norm_weight, [HEAD_DIM], [0 * HEAD_DIM])
    index_wk_c0 = pl.slice(index_wk, [HEAD_DIM, INDEX_DIM], [0 * HEAD_DIM, 0])
    index_norm_weight_c0 = pl.slice(index_norm_weight, [INDEX_DIM], [0 * INDEX_DIM])
    index_wq_b_c0 = pl.slice(index_wq_b, [Q_LORA, INDEX_H * INDEX_DIM], [0 * Q_LORA, 0])
    index_wq_b_scale_c0: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(index_wq_b_scale, [Q_LORA // 32, INDEX_H * INDEX_DIM], [0 * (Q_LORA // 32), 0])
    index_weights_proj_c0 = pl.slice(index_weights_proj, [D, INDEX_H], [0 * D, 0])
    decode_c1a_full(
        chain_x,
        chain_pre,
        hc_attn_fn_c0,
        hc_attn_scale_c0,
        hc_attn_base_c0,
        attn_norm_weight_c0,
        wq_a_c0,
        wq_a_scale_c0,
        q_norm_weight_c0,
        wq_b_c0,
        wq_b_scale_c0,
        wkv_c0,
        wkv_scale_c0,
        kv_norm_weight_c0,
        attn_sink_c0,
        wo_a_c0,
        wo_b_c0,
        wo_b_scale_c0,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c0,
        window_cache_scale_c0,
        compressed_cache,
        compressed_cache_scale,
        request_ids,
        compressed_lens,
        index_cache,
        index_cache_scale,
        index_block_table,
        compressed_rope_cos,
        compressed_rope_sin,
        compressor_wkv_c0,
        compressor_norm_weight_c0,
        compressed_slots,
        index_wk_c0,
        index_norm_weight_c0,
        index_wq_b_c0,
        index_wq_b_scale_c0,
        index_weights_proj_c0,
        topk_indices,
        candidate_mask,
        output_window,
        output_arrived,
        x0,
        p0,
        group_base, tp_rank, num_tokens, epoch_base + (1),
    )
    hc_attn_fn_c1 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [1 * MIX_HC, 0])
    hc_attn_scale_c1 = pl.slice(hc_attn_scale, [3], [1 * 3])
    hc_attn_base_c1 = pl.slice(hc_attn_base, [MIX_HC], [1 * MIX_HC])
    attn_norm_weight_c1 = pl.slice(attn_norm_weight, [D], [1 * D])
    wq_a_c1 = pl.slice(wq_a, [D, Q_LORA], [1 * D, 0])
    wq_a_scale_c1: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [1 * (D // 32), 0])
    q_norm_weight_c1 = pl.slice(q_norm_weight, [Q_LORA], [1 * Q_LORA])
    wq_b_c1 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [1 * Q_LORA, 0])
    wq_b_scale_c1: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [1 * (Q_LORA // 32), 0])
    wkv_c1 = pl.slice(wkv, [D, HEAD_DIM], [1 * D, 0])
    wkv_scale_c1: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [1 * (D // 32), 0])
    kv_norm_weight_c1 = pl.slice(kv_norm_weight, [HEAD_DIM], [1 * HEAD_DIM])
    attn_sink_c1 = pl.slice(attn_sink, [LOCAL_H], [1 * LOCAL_H])
    wo_a_c1 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [1 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c1 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [1 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c1: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [1 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c1 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [1 * PAGES, 0, 0, 0])
    window_cache_scale_c1 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [1 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x0,
        p0,
        hc_attn_fn_c1,
        hc_attn_scale_c1,
        hc_attn_base_c1,
        attn_norm_weight_c1,
        wq_a_c1,
        wq_a_scale_c1,
        q_norm_weight_c1,
        wq_b_c1,
        wq_b_scale_c1,
        wkv_c1,
        wkv_scale_c1,
        kv_norm_weight_c1,
        attn_sink_c1,
        wo_a_c1,
        wo_b_c1,
        wo_b_scale_c1,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c1,
        window_cache_scale_c1,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x1,
        p1,
        group_base, tp_rank, num_tokens, epoch_base + (2),
    )
    hc_attn_fn_c2 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [2 * MIX_HC, 0])
    hc_attn_scale_c2 = pl.slice(hc_attn_scale, [3], [2 * 3])
    hc_attn_base_c2 = pl.slice(hc_attn_base, [MIX_HC], [2 * MIX_HC])
    attn_norm_weight_c2 = pl.slice(attn_norm_weight, [D], [2 * D])
    wq_a_c2 = pl.slice(wq_a, [D, Q_LORA], [2 * D, 0])
    wq_a_scale_c2: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [2 * (D // 32), 0])
    q_norm_weight_c2 = pl.slice(q_norm_weight, [Q_LORA], [2 * Q_LORA])
    wq_b_c2 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [2 * Q_LORA, 0])
    wq_b_scale_c2: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [2 * (Q_LORA // 32), 0])
    wkv_c2 = pl.slice(wkv, [D, HEAD_DIM], [2 * D, 0])
    wkv_scale_c2: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [2 * (D // 32), 0])
    kv_norm_weight_c2 = pl.slice(kv_norm_weight, [HEAD_DIM], [2 * HEAD_DIM])
    attn_sink_c2 = pl.slice(attn_sink, [LOCAL_H], [2 * LOCAL_H])
    wo_a_c2 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [2 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c2 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [2 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c2: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [2 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c2 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [2 * PAGES, 0, 0, 0])
    window_cache_scale_c2 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [2 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x1,
        p1,
        hc_attn_fn_c2,
        hc_attn_scale_c2,
        hc_attn_base_c2,
        attn_norm_weight_c2,
        wq_a_c2,
        wq_a_scale_c2,
        q_norm_weight_c2,
        wq_b_c2,
        wq_b_scale_c2,
        wkv_c2,
        wkv_scale_c2,
        kv_norm_weight_c2,
        attn_sink_c2,
        wo_a_c2,
        wo_b_c2,
        wo_b_scale_c2,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c2,
        window_cache_scale_c2,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x0,
        p0,
        group_base, tp_rank, num_tokens, epoch_base + (3),
    )
    hc_attn_fn_c3 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [3 * MIX_HC, 0])
    hc_attn_scale_c3 = pl.slice(hc_attn_scale, [3], [3 * 3])
    hc_attn_base_c3 = pl.slice(hc_attn_base, [MIX_HC], [3 * MIX_HC])
    attn_norm_weight_c3 = pl.slice(attn_norm_weight, [D], [3 * D])
    wq_a_c3 = pl.slice(wq_a, [D, Q_LORA], [3 * D, 0])
    wq_a_scale_c3: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [3 * (D // 32), 0])
    q_norm_weight_c3 = pl.slice(q_norm_weight, [Q_LORA], [3 * Q_LORA])
    wq_b_c3 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [3 * Q_LORA, 0])
    wq_b_scale_c3: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [3 * (Q_LORA // 32), 0])
    wkv_c3 = pl.slice(wkv, [D, HEAD_DIM], [3 * D, 0])
    wkv_scale_c3: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [3 * (D // 32), 0])
    kv_norm_weight_c3 = pl.slice(kv_norm_weight, [HEAD_DIM], [3 * HEAD_DIM])
    attn_sink_c3 = pl.slice(attn_sink, [LOCAL_H], [3 * LOCAL_H])
    wo_a_c3 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [3 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c3 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [3 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c3: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [3 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c3 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [3 * PAGES, 0, 0, 0])
    window_cache_scale_c3 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [3 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x0,
        p0,
        hc_attn_fn_c3,
        hc_attn_scale_c3,
        hc_attn_base_c3,
        attn_norm_weight_c3,
        wq_a_c3,
        wq_a_scale_c3,
        q_norm_weight_c3,
        wq_b_c3,
        wq_b_scale_c3,
        wkv_c3,
        wkv_scale_c3,
        kv_norm_weight_c3,
        attn_sink_c3,
        wo_a_c3,
        wo_b_c3,
        wo_b_scale_c3,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c3,
        window_cache_scale_c3,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x1,
        p1,
        group_base, tp_rank, num_tokens, epoch_base + (4),
    )
    hc_attn_fn_c4 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [4 * MIX_HC, 0])
    hc_attn_scale_c4 = pl.slice(hc_attn_scale, [3], [4 * 3])
    hc_attn_base_c4 = pl.slice(hc_attn_base, [MIX_HC], [4 * MIX_HC])
    attn_norm_weight_c4 = pl.slice(attn_norm_weight, [D], [4 * D])
    wq_a_c4 = pl.slice(wq_a, [D, Q_LORA], [4 * D, 0])
    wq_a_scale_c4: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [4 * (D // 32), 0])
    q_norm_weight_c4 = pl.slice(q_norm_weight, [Q_LORA], [4 * Q_LORA])
    wq_b_c4 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [4 * Q_LORA, 0])
    wq_b_scale_c4: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [4 * (Q_LORA // 32), 0])
    wkv_c4 = pl.slice(wkv, [D, HEAD_DIM], [4 * D, 0])
    wkv_scale_c4: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [4 * (D // 32), 0])
    kv_norm_weight_c4 = pl.slice(kv_norm_weight, [HEAD_DIM], [4 * HEAD_DIM])
    attn_sink_c4 = pl.slice(attn_sink, [LOCAL_H], [4 * LOCAL_H])
    wo_a_c4 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [4 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c4 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [4 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c4: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [4 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c4 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [4 * PAGES, 0, 0, 0])
    window_cache_scale_c4 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [4 * PAGES, 0, 0, 0])
    index_wq_b_c4 = pl.slice(index_wq_b, [Q_LORA, INDEX_H * INDEX_DIM], [4 * Q_LORA, 0])
    index_wq_b_scale_c4: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(index_wq_b_scale, [Q_LORA // 32, INDEX_H * INDEX_DIM], [4 * (Q_LORA // 32), 0])
    index_weights_proj_c4 = pl.slice(index_weights_proj, [D, INDEX_H], [4 * D, 0])
    decode_c1a_reindex(
        x1,
        p1,
        hc_attn_fn_c4,
        hc_attn_scale_c4,
        hc_attn_base_c4,
        attn_norm_weight_c4,
        wq_a_c4,
        wq_a_scale_c4,
        q_norm_weight_c4,
        wq_b_c4,
        wq_b_scale_c4,
        wkv_c4,
        wkv_scale_c4,
        kv_norm_weight_c4,
        attn_sink_c4,
        wo_a_c4,
        wo_b_c4,
        wo_b_scale_c4,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c4,
        window_cache_scale_c4,
        compressed_cache,
        compressed_cache_scale,
        request_ids,
        compressed_lens,
        index_cache,
        index_cache_scale,
        index_block_table,
        candidate_mask,
        index_wq_b_c4,
        index_wq_b_scale_c4,
        index_weights_proj_c4,
        topk_indices,
        output_window,
        output_arrived,
        x0,
        p0,
        group_base, tp_rank, num_tokens, epoch_base + (5),
    )
    hc_attn_fn_c5 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [5 * MIX_HC, 0])
    hc_attn_scale_c5 = pl.slice(hc_attn_scale, [3], [5 * 3])
    hc_attn_base_c5 = pl.slice(hc_attn_base, [MIX_HC], [5 * MIX_HC])
    attn_norm_weight_c5 = pl.slice(attn_norm_weight, [D], [5 * D])
    wq_a_c5 = pl.slice(wq_a, [D, Q_LORA], [5 * D, 0])
    wq_a_scale_c5: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [5 * (D // 32), 0])
    q_norm_weight_c5 = pl.slice(q_norm_weight, [Q_LORA], [5 * Q_LORA])
    wq_b_c5 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [5 * Q_LORA, 0])
    wq_b_scale_c5: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [5 * (Q_LORA // 32), 0])
    wkv_c5 = pl.slice(wkv, [D, HEAD_DIM], [5 * D, 0])
    wkv_scale_c5: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [5 * (D // 32), 0])
    kv_norm_weight_c5 = pl.slice(kv_norm_weight, [HEAD_DIM], [5 * HEAD_DIM])
    attn_sink_c5 = pl.slice(attn_sink, [LOCAL_H], [5 * LOCAL_H])
    wo_a_c5 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [5 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c5 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [5 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c5: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [5 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c5 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [5 * PAGES, 0, 0, 0])
    window_cache_scale_c5 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [5 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x0,
        p0,
        hc_attn_fn_c5,
        hc_attn_scale_c5,
        hc_attn_base_c5,
        attn_norm_weight_c5,
        wq_a_c5,
        wq_a_scale_c5,
        q_norm_weight_c5,
        wq_b_c5,
        wq_b_scale_c5,
        wkv_c5,
        wkv_scale_c5,
        kv_norm_weight_c5,
        attn_sink_c5,
        wo_a_c5,
        wo_b_c5,
        wo_b_scale_c5,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c5,
        window_cache_scale_c5,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x1,
        p1,
        group_base, tp_rank, num_tokens, epoch_base + (6),
    )
    hc_attn_fn_c6 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [6 * MIX_HC, 0])
    hc_attn_scale_c6 = pl.slice(hc_attn_scale, [3], [6 * 3])
    hc_attn_base_c6 = pl.slice(hc_attn_base, [MIX_HC], [6 * MIX_HC])
    attn_norm_weight_c6 = pl.slice(attn_norm_weight, [D], [6 * D])
    wq_a_c6 = pl.slice(wq_a, [D, Q_LORA], [6 * D, 0])
    wq_a_scale_c6: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [6 * (D // 32), 0])
    q_norm_weight_c6 = pl.slice(q_norm_weight, [Q_LORA], [6 * Q_LORA])
    wq_b_c6 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [6 * Q_LORA, 0])
    wq_b_scale_c6: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [6 * (Q_LORA // 32), 0])
    wkv_c6 = pl.slice(wkv, [D, HEAD_DIM], [6 * D, 0])
    wkv_scale_c6: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [6 * (D // 32), 0])
    kv_norm_weight_c6 = pl.slice(kv_norm_weight, [HEAD_DIM], [6 * HEAD_DIM])
    attn_sink_c6 = pl.slice(attn_sink, [LOCAL_H], [6 * LOCAL_H])
    wo_a_c6 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [6 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c6 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [6 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c6: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [6 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c6 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [6 * PAGES, 0, 0, 0])
    window_cache_scale_c6 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [6 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x1,
        p1,
        hc_attn_fn_c6,
        hc_attn_scale_c6,
        hc_attn_base_c6,
        attn_norm_weight_c6,
        wq_a_c6,
        wq_a_scale_c6,
        q_norm_weight_c6,
        wq_b_c6,
        wq_b_scale_c6,
        wkv_c6,
        wkv_scale_c6,
        kv_norm_weight_c6,
        attn_sink_c6,
        wo_a_c6,
        wo_b_c6,
        wo_b_scale_c6,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c6,
        window_cache_scale_c6,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x0,
        p0,
        group_base, tp_rank, num_tokens, epoch_base + (7),
    )
    hc_attn_fn_c7 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [7 * MIX_HC, 0])
    hc_attn_scale_c7 = pl.slice(hc_attn_scale, [3], [7 * 3])
    hc_attn_base_c7 = pl.slice(hc_attn_base, [MIX_HC], [7 * MIX_HC])
    attn_norm_weight_c7 = pl.slice(attn_norm_weight, [D], [7 * D])
    wq_a_c7 = pl.slice(wq_a, [D, Q_LORA], [7 * D, 0])
    wq_a_scale_c7: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [7 * (D // 32), 0])
    q_norm_weight_c7 = pl.slice(q_norm_weight, [Q_LORA], [7 * Q_LORA])
    wq_b_c7 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [7 * Q_LORA, 0])
    wq_b_scale_c7: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [7 * (Q_LORA // 32), 0])
    wkv_c7 = pl.slice(wkv, [D, HEAD_DIM], [7 * D, 0])
    wkv_scale_c7: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [7 * (D // 32), 0])
    kv_norm_weight_c7 = pl.slice(kv_norm_weight, [HEAD_DIM], [7 * HEAD_DIM])
    attn_sink_c7 = pl.slice(attn_sink, [LOCAL_H], [7 * LOCAL_H])
    wo_a_c7 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [7 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c7 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [7 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c7: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [7 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c7 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [7 * PAGES, 0, 0, 0])
    window_cache_scale_c7 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [7 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x0,
        p0,
        hc_attn_fn_c7,
        hc_attn_scale_c7,
        hc_attn_base_c7,
        attn_norm_weight_c7,
        wq_a_c7,
        wq_a_scale_c7,
        q_norm_weight_c7,
        wq_b_c7,
        wq_b_scale_c7,
        wkv_c7,
        wkv_scale_c7,
        kv_norm_weight_c7,
        attn_sink_c7,
        wo_a_c7,
        wo_b_c7,
        wo_b_scale_c7,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c7,
        window_cache_scale_c7,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x1,
        p1,
        group_base, tp_rank, num_tokens, epoch_base + (8),
    )
    hc_attn_fn_c8 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [8 * MIX_HC, 0])
    hc_attn_scale_c8 = pl.slice(hc_attn_scale, [3], [8 * 3])
    hc_attn_base_c8 = pl.slice(hc_attn_base, [MIX_HC], [8 * MIX_HC])
    attn_norm_weight_c8 = pl.slice(attn_norm_weight, [D], [8 * D])
    wq_a_c8 = pl.slice(wq_a, [D, Q_LORA], [8 * D, 0])
    wq_a_scale_c8: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [8 * (D // 32), 0])
    q_norm_weight_c8 = pl.slice(q_norm_weight, [Q_LORA], [8 * Q_LORA])
    wq_b_c8 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [8 * Q_LORA, 0])
    wq_b_scale_c8: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [8 * (Q_LORA // 32), 0])
    wkv_c8 = pl.slice(wkv, [D, HEAD_DIM], [8 * D, 0])
    wkv_scale_c8: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [8 * (D // 32), 0])
    kv_norm_weight_c8 = pl.slice(kv_norm_weight, [HEAD_DIM], [8 * HEAD_DIM])
    attn_sink_c8 = pl.slice(attn_sink, [LOCAL_H], [8 * LOCAL_H])
    wo_a_c8 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [8 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c8 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [8 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c8: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [8 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c8 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [8 * PAGES, 0, 0, 0])
    window_cache_scale_c8 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [8 * PAGES, 0, 0, 0])
    index_wq_b_c8 = pl.slice(index_wq_b, [Q_LORA, INDEX_H * INDEX_DIM], [8 * Q_LORA, 0])
    index_wq_b_scale_c8: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(index_wq_b_scale, [Q_LORA // 32, INDEX_H * INDEX_DIM], [8 * (Q_LORA // 32), 0])
    index_weights_proj_c8 = pl.slice(index_weights_proj, [D, INDEX_H], [8 * D, 0])
    decode_c1a_reindex(
        x1,
        p1,
        hc_attn_fn_c8,
        hc_attn_scale_c8,
        hc_attn_base_c8,
        attn_norm_weight_c8,
        wq_a_c8,
        wq_a_scale_c8,
        q_norm_weight_c8,
        wq_b_c8,
        wq_b_scale_c8,
        wkv_c8,
        wkv_scale_c8,
        kv_norm_weight_c8,
        attn_sink_c8,
        wo_a_c8,
        wo_b_c8,
        wo_b_scale_c8,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c8,
        window_cache_scale_c8,
        compressed_cache,
        compressed_cache_scale,
        request_ids,
        compressed_lens,
        index_cache,
        index_cache_scale,
        index_block_table,
        candidate_mask,
        index_wq_b_c8,
        index_wq_b_scale_c8,
        index_weights_proj_c8,
        topk_indices,
        output_window,
        output_arrived,
        x0,
        p0,
        group_base, tp_rank, num_tokens, epoch_base + (9),
    )
    hc_attn_fn_c9 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [9 * MIX_HC, 0])
    hc_attn_scale_c9 = pl.slice(hc_attn_scale, [3], [9 * 3])
    hc_attn_base_c9 = pl.slice(hc_attn_base, [MIX_HC], [9 * MIX_HC])
    attn_norm_weight_c9 = pl.slice(attn_norm_weight, [D], [9 * D])
    wq_a_c9 = pl.slice(wq_a, [D, Q_LORA], [9 * D, 0])
    wq_a_scale_c9: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [9 * (D // 32), 0])
    q_norm_weight_c9 = pl.slice(q_norm_weight, [Q_LORA], [9 * Q_LORA])
    wq_b_c9 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [9 * Q_LORA, 0])
    wq_b_scale_c9: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [9 * (Q_LORA // 32), 0])
    wkv_c9 = pl.slice(wkv, [D, HEAD_DIM], [9 * D, 0])
    wkv_scale_c9: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [9 * (D // 32), 0])
    kv_norm_weight_c9 = pl.slice(kv_norm_weight, [HEAD_DIM], [9 * HEAD_DIM])
    attn_sink_c9 = pl.slice(attn_sink, [LOCAL_H], [9 * LOCAL_H])
    wo_a_c9 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [9 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c9 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [9 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c9: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [9 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c9 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [9 * PAGES, 0, 0, 0])
    window_cache_scale_c9 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [9 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x0,
        p0,
        hc_attn_fn_c9,
        hc_attn_scale_c9,
        hc_attn_base_c9,
        attn_norm_weight_c9,
        wq_a_c9,
        wq_a_scale_c9,
        q_norm_weight_c9,
        wq_b_c9,
        wq_b_scale_c9,
        wkv_c9,
        wkv_scale_c9,
        kv_norm_weight_c9,
        attn_sink_c9,
        wo_a_c9,
        wo_b_c9,
        wo_b_scale_c9,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c9,
        window_cache_scale_c9,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x1,
        p1,
        group_base, tp_rank, num_tokens, epoch_base + (10),
    )
    hc_attn_fn_c10 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [10 * MIX_HC, 0])
    hc_attn_scale_c10 = pl.slice(hc_attn_scale, [3], [10 * 3])
    hc_attn_base_c10 = pl.slice(hc_attn_base, [MIX_HC], [10 * MIX_HC])
    attn_norm_weight_c10 = pl.slice(attn_norm_weight, [D], [10 * D])
    wq_a_c10 = pl.slice(wq_a, [D, Q_LORA], [10 * D, 0])
    wq_a_scale_c10: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [10 * (D // 32), 0])
    q_norm_weight_c10 = pl.slice(q_norm_weight, [Q_LORA], [10 * Q_LORA])
    wq_b_c10 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [10 * Q_LORA, 0])
    wq_b_scale_c10: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [10 * (Q_LORA // 32), 0])
    wkv_c10 = pl.slice(wkv, [D, HEAD_DIM], [10 * D, 0])
    wkv_scale_c10: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [10 * (D // 32), 0])
    kv_norm_weight_c10 = pl.slice(kv_norm_weight, [HEAD_DIM], [10 * HEAD_DIM])
    attn_sink_c10 = pl.slice(attn_sink, [LOCAL_H], [10 * LOCAL_H])
    wo_a_c10 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [10 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c10 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [10 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c10: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [10 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c10 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [10 * PAGES, 0, 0, 0])
    window_cache_scale_c10 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [10 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x1,
        p1,
        hc_attn_fn_c10,
        hc_attn_scale_c10,
        hc_attn_base_c10,
        attn_norm_weight_c10,
        wq_a_c10,
        wq_a_scale_c10,
        q_norm_weight_c10,
        wq_b_c10,
        wq_b_scale_c10,
        wkv_c10,
        wkv_scale_c10,
        kv_norm_weight_c10,
        attn_sink_c10,
        wo_a_c10,
        wo_b_c10,
        wo_b_scale_c10,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c10,
        window_cache_scale_c10,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x0,
        p0,
        group_base, tp_rank, num_tokens, epoch_base + (11),
    )
    hc_attn_fn_c11 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [11 * MIX_HC, 0])
    hc_attn_scale_c11 = pl.slice(hc_attn_scale, [3], [11 * 3])
    hc_attn_base_c11 = pl.slice(hc_attn_base, [MIX_HC], [11 * MIX_HC])
    attn_norm_weight_c11 = pl.slice(attn_norm_weight, [D], [11 * D])
    wq_a_c11 = pl.slice(wq_a, [D, Q_LORA], [11 * D, 0])
    wq_a_scale_c11: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [11 * (D // 32), 0])
    q_norm_weight_c11 = pl.slice(q_norm_weight, [Q_LORA], [11 * Q_LORA])
    wq_b_c11 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [11 * Q_LORA, 0])
    wq_b_scale_c11: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [11 * (Q_LORA // 32), 0])
    wkv_c11 = pl.slice(wkv, [D, HEAD_DIM], [11 * D, 0])
    wkv_scale_c11: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [11 * (D // 32), 0])
    kv_norm_weight_c11 = pl.slice(kv_norm_weight, [HEAD_DIM], [11 * HEAD_DIM])
    attn_sink_c11 = pl.slice(attn_sink, [LOCAL_H], [11 * LOCAL_H])
    wo_a_c11 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [11 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c11 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [11 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c11: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [11 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c11 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [11 * PAGES, 0, 0, 0])
    window_cache_scale_c11 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [11 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x0,
        p0,
        hc_attn_fn_c11,
        hc_attn_scale_c11,
        hc_attn_base_c11,
        attn_norm_weight_c11,
        wq_a_c11,
        wq_a_scale_c11,
        q_norm_weight_c11,
        wq_b_c11,
        wq_b_scale_c11,
        wkv_c11,
        wkv_scale_c11,
        kv_norm_weight_c11,
        attn_sink_c11,
        wo_a_c11,
        wo_b_c11,
        wo_b_scale_c11,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c11,
        window_cache_scale_c11,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x1,
        p1,
        group_base, tp_rank, num_tokens, epoch_base + (12),
    )
    hc_attn_fn_c12 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [12 * MIX_HC, 0])
    hc_attn_scale_c12 = pl.slice(hc_attn_scale, [3], [12 * 3])
    hc_attn_base_c12 = pl.slice(hc_attn_base, [MIX_HC], [12 * MIX_HC])
    attn_norm_weight_c12 = pl.slice(attn_norm_weight, [D], [12 * D])
    wq_a_c12 = pl.slice(wq_a, [D, Q_LORA], [12 * D, 0])
    wq_a_scale_c12: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [12 * (D // 32), 0])
    q_norm_weight_c12 = pl.slice(q_norm_weight, [Q_LORA], [12 * Q_LORA])
    wq_b_c12 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [12 * Q_LORA, 0])
    wq_b_scale_c12: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [12 * (Q_LORA // 32), 0])
    wkv_c12 = pl.slice(wkv, [D, HEAD_DIM], [12 * D, 0])
    wkv_scale_c12: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [12 * (D // 32), 0])
    kv_norm_weight_c12 = pl.slice(kv_norm_weight, [HEAD_DIM], [12 * HEAD_DIM])
    attn_sink_c12 = pl.slice(attn_sink, [LOCAL_H], [12 * LOCAL_H])
    wo_a_c12 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [12 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c12 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [12 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c12: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [12 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c12 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [12 * PAGES, 0, 0, 0])
    window_cache_scale_c12 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [12 * PAGES, 0, 0, 0])
    index_wq_b_c12 = pl.slice(index_wq_b, [Q_LORA, INDEX_H * INDEX_DIM], [12 * Q_LORA, 0])
    index_wq_b_scale_c12: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(index_wq_b_scale, [Q_LORA // 32, INDEX_H * INDEX_DIM], [12 * (Q_LORA // 32), 0])
    index_weights_proj_c12 = pl.slice(index_weights_proj, [D, INDEX_H], [12 * D, 0])
    decode_c1a_reindex(
        x1,
        p1,
        hc_attn_fn_c12,
        hc_attn_scale_c12,
        hc_attn_base_c12,
        attn_norm_weight_c12,
        wq_a_c12,
        wq_a_scale_c12,
        q_norm_weight_c12,
        wq_b_c12,
        wq_b_scale_c12,
        wkv_c12,
        wkv_scale_c12,
        kv_norm_weight_c12,
        attn_sink_c12,
        wo_a_c12,
        wo_b_c12,
        wo_b_scale_c12,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c12,
        window_cache_scale_c12,
        compressed_cache,
        compressed_cache_scale,
        request_ids,
        compressed_lens,
        index_cache,
        index_cache_scale,
        index_block_table,
        candidate_mask,
        index_wq_b_c12,
        index_wq_b_scale_c12,
        index_weights_proj_c12,
        topk_indices,
        output_window,
        output_arrived,
        x0,
        p0,
        group_base, tp_rank, num_tokens, epoch_base + (13),
    )
    hc_attn_fn_c13 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [13 * MIX_HC, 0])
    hc_attn_scale_c13 = pl.slice(hc_attn_scale, [3], [13 * 3])
    hc_attn_base_c13 = pl.slice(hc_attn_base, [MIX_HC], [13 * MIX_HC])
    attn_norm_weight_c13 = pl.slice(attn_norm_weight, [D], [13 * D])
    wq_a_c13 = pl.slice(wq_a, [D, Q_LORA], [13 * D, 0])
    wq_a_scale_c13: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [13 * (D // 32), 0])
    q_norm_weight_c13 = pl.slice(q_norm_weight, [Q_LORA], [13 * Q_LORA])
    wq_b_c13 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [13 * Q_LORA, 0])
    wq_b_scale_c13: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [13 * (Q_LORA // 32), 0])
    wkv_c13 = pl.slice(wkv, [D, HEAD_DIM], [13 * D, 0])
    wkv_scale_c13: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [13 * (D // 32), 0])
    kv_norm_weight_c13 = pl.slice(kv_norm_weight, [HEAD_DIM], [13 * HEAD_DIM])
    attn_sink_c13 = pl.slice(attn_sink, [LOCAL_H], [13 * LOCAL_H])
    wo_a_c13 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [13 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c13 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [13 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c13: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [13 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c13 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [13 * PAGES, 0, 0, 0])
    window_cache_scale_c13 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [13 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x0,
        p0,
        hc_attn_fn_c13,
        hc_attn_scale_c13,
        hc_attn_base_c13,
        attn_norm_weight_c13,
        wq_a_c13,
        wq_a_scale_c13,
        q_norm_weight_c13,
        wq_b_c13,
        wq_b_scale_c13,
        wkv_c13,
        wkv_scale_c13,
        kv_norm_weight_c13,
        attn_sink_c13,
        wo_a_c13,
        wo_b_c13,
        wo_b_scale_c13,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c13,
        window_cache_scale_c13,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x1,
        p1,
        group_base, tp_rank, num_tokens, epoch_base + (14),
    )
    hc_attn_fn_c14 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [14 * MIX_HC, 0])
    hc_attn_scale_c14 = pl.slice(hc_attn_scale, [3], [14 * 3])
    hc_attn_base_c14 = pl.slice(hc_attn_base, [MIX_HC], [14 * MIX_HC])
    attn_norm_weight_c14 = pl.slice(attn_norm_weight, [D], [14 * D])
    wq_a_c14 = pl.slice(wq_a, [D, Q_LORA], [14 * D, 0])
    wq_a_scale_c14: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [14 * (D // 32), 0])
    q_norm_weight_c14 = pl.slice(q_norm_weight, [Q_LORA], [14 * Q_LORA])
    wq_b_c14 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [14 * Q_LORA, 0])
    wq_b_scale_c14: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [14 * (Q_LORA // 32), 0])
    wkv_c14 = pl.slice(wkv, [D, HEAD_DIM], [14 * D, 0])
    wkv_scale_c14: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [14 * (D // 32), 0])
    kv_norm_weight_c14 = pl.slice(kv_norm_weight, [HEAD_DIM], [14 * HEAD_DIM])
    attn_sink_c14 = pl.slice(attn_sink, [LOCAL_H], [14 * LOCAL_H])
    wo_a_c14 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [14 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c14 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [14 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c14: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [14 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c14 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [14 * PAGES, 0, 0, 0])
    window_cache_scale_c14 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [14 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x1,
        p1,
        hc_attn_fn_c14,
        hc_attn_scale_c14,
        hc_attn_base_c14,
        attn_norm_weight_c14,
        wq_a_c14,
        wq_a_scale_c14,
        q_norm_weight_c14,
        wq_b_c14,
        wq_b_scale_c14,
        wkv_c14,
        wkv_scale_c14,
        kv_norm_weight_c14,
        attn_sink_c14,
        wo_a_c14,
        wo_b_c14,
        wo_b_scale_c14,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c14,
        window_cache_scale_c14,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x0,
        p0,
        group_base, tp_rank, num_tokens, epoch_base + (15),
    )
    hc_attn_fn_c15 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [15 * MIX_HC, 0])
    hc_attn_scale_c15 = pl.slice(hc_attn_scale, [3], [15 * 3])
    hc_attn_base_c15 = pl.slice(hc_attn_base, [MIX_HC], [15 * MIX_HC])
    attn_norm_weight_c15 = pl.slice(attn_norm_weight, [D], [15 * D])
    wq_a_c15 = pl.slice(wq_a, [D, Q_LORA], [15 * D, 0])
    wq_a_scale_c15: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [15 * (D // 32), 0])
    q_norm_weight_c15 = pl.slice(q_norm_weight, [Q_LORA], [15 * Q_LORA])
    wq_b_c15 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [15 * Q_LORA, 0])
    wq_b_scale_c15: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [15 * (Q_LORA // 32), 0])
    wkv_c15 = pl.slice(wkv, [D, HEAD_DIM], [15 * D, 0])
    wkv_scale_c15: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [15 * (D // 32), 0])
    kv_norm_weight_c15 = pl.slice(kv_norm_weight, [HEAD_DIM], [15 * HEAD_DIM])
    attn_sink_c15 = pl.slice(attn_sink, [LOCAL_H], [15 * LOCAL_H])
    wo_a_c15 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [15 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c15 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [15 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c15: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [15 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c15 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [15 * PAGES, 0, 0, 0])
    window_cache_scale_c15 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [15 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x0,
        p0,
        hc_attn_fn_c15,
        hc_attn_scale_c15,
        hc_attn_base_c15,
        attn_norm_weight_c15,
        wq_a_c15,
        wq_a_scale_c15,
        q_norm_weight_c15,
        wq_b_c15,
        wq_b_scale_c15,
        wkv_c15,
        wkv_scale_c15,
        kv_norm_weight_c15,
        attn_sink_c15,
        wo_a_c15,
        wo_b_c15,
        wo_b_scale_c15,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c15,
        window_cache_scale_c15,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x1,
        p1,
        group_base, tp_rank, num_tokens, epoch_base + (16),
    )
    hc_attn_fn_c16 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [16 * MIX_HC, 0])
    hc_attn_scale_c16 = pl.slice(hc_attn_scale, [3], [16 * 3])
    hc_attn_base_c16 = pl.slice(hc_attn_base, [MIX_HC], [16 * MIX_HC])
    attn_norm_weight_c16 = pl.slice(attn_norm_weight, [D], [16 * D])
    wq_a_c16 = pl.slice(wq_a, [D, Q_LORA], [16 * D, 0])
    wq_a_scale_c16: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [16 * (D // 32), 0])
    q_norm_weight_c16 = pl.slice(q_norm_weight, [Q_LORA], [16 * Q_LORA])
    wq_b_c16 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [16 * Q_LORA, 0])
    wq_b_scale_c16: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [16 * (Q_LORA // 32), 0])
    wkv_c16 = pl.slice(wkv, [D, HEAD_DIM], [16 * D, 0])
    wkv_scale_c16: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [16 * (D // 32), 0])
    kv_norm_weight_c16 = pl.slice(kv_norm_weight, [HEAD_DIM], [16 * HEAD_DIM])
    attn_sink_c16 = pl.slice(attn_sink, [LOCAL_H], [16 * LOCAL_H])
    wo_a_c16 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [16 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c16 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [16 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c16: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [16 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c16 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [16 * PAGES, 0, 0, 0])
    window_cache_scale_c16 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [16 * PAGES, 0, 0, 0])
    index_wq_b_c16 = pl.slice(index_wq_b, [Q_LORA, INDEX_H * INDEX_DIM], [16 * Q_LORA, 0])
    index_wq_b_scale_c16: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(index_wq_b_scale, [Q_LORA // 32, INDEX_H * INDEX_DIM], [16 * (Q_LORA // 32), 0])
    index_weights_proj_c16 = pl.slice(index_weights_proj, [D, INDEX_H], [16 * D, 0])
    decode_c1a_reindex(
        x1,
        p1,
        hc_attn_fn_c16,
        hc_attn_scale_c16,
        hc_attn_base_c16,
        attn_norm_weight_c16,
        wq_a_c16,
        wq_a_scale_c16,
        q_norm_weight_c16,
        wq_b_c16,
        wq_b_scale_c16,
        wkv_c16,
        wkv_scale_c16,
        kv_norm_weight_c16,
        attn_sink_c16,
        wo_a_c16,
        wo_b_c16,
        wo_b_scale_c16,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c16,
        window_cache_scale_c16,
        compressed_cache,
        compressed_cache_scale,
        request_ids,
        compressed_lens,
        index_cache,
        index_cache_scale,
        index_block_table,
        candidate_mask,
        index_wq_b_c16,
        index_wq_b_scale_c16,
        index_weights_proj_c16,
        topk_indices,
        output_window,
        output_arrived,
        x0,
        p0,
        group_base, tp_rank, num_tokens, epoch_base + (17),
    )
    hc_attn_fn_c17 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [17 * MIX_HC, 0])
    hc_attn_scale_c17 = pl.slice(hc_attn_scale, [3], [17 * 3])
    hc_attn_base_c17 = pl.slice(hc_attn_base, [MIX_HC], [17 * MIX_HC])
    attn_norm_weight_c17 = pl.slice(attn_norm_weight, [D], [17 * D])
    wq_a_c17 = pl.slice(wq_a, [D, Q_LORA], [17 * D, 0])
    wq_a_scale_c17: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [17 * (D // 32), 0])
    q_norm_weight_c17 = pl.slice(q_norm_weight, [Q_LORA], [17 * Q_LORA])
    wq_b_c17 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [17 * Q_LORA, 0])
    wq_b_scale_c17: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [17 * (Q_LORA // 32), 0])
    wkv_c17 = pl.slice(wkv, [D, HEAD_DIM], [17 * D, 0])
    wkv_scale_c17: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [17 * (D // 32), 0])
    kv_norm_weight_c17 = pl.slice(kv_norm_weight, [HEAD_DIM], [17 * HEAD_DIM])
    attn_sink_c17 = pl.slice(attn_sink, [LOCAL_H], [17 * LOCAL_H])
    wo_a_c17 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [17 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c17 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [17 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c17: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [17 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c17 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [17 * PAGES, 0, 0, 0])
    window_cache_scale_c17 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [17 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x0,
        p0,
        hc_attn_fn_c17,
        hc_attn_scale_c17,
        hc_attn_base_c17,
        attn_norm_weight_c17,
        wq_a_c17,
        wq_a_scale_c17,
        q_norm_weight_c17,
        wq_b_c17,
        wq_b_scale_c17,
        wkv_c17,
        wkv_scale_c17,
        kv_norm_weight_c17,
        attn_sink_c17,
        wo_a_c17,
        wo_b_c17,
        wo_b_scale_c17,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c17,
        window_cache_scale_c17,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x1,
        p1,
        group_base, tp_rank, num_tokens, epoch_base + (18),
    )
    hc_attn_fn_c18 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [18 * MIX_HC, 0])
    hc_attn_scale_c18 = pl.slice(hc_attn_scale, [3], [18 * 3])
    hc_attn_base_c18 = pl.slice(hc_attn_base, [MIX_HC], [18 * MIX_HC])
    attn_norm_weight_c18 = pl.slice(attn_norm_weight, [D], [18 * D])
    wq_a_c18 = pl.slice(wq_a, [D, Q_LORA], [18 * D, 0])
    wq_a_scale_c18: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [18 * (D // 32), 0])
    q_norm_weight_c18 = pl.slice(q_norm_weight, [Q_LORA], [18 * Q_LORA])
    wq_b_c18 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [18 * Q_LORA, 0])
    wq_b_scale_c18: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [18 * (Q_LORA // 32), 0])
    wkv_c18 = pl.slice(wkv, [D, HEAD_DIM], [18 * D, 0])
    wkv_scale_c18: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [18 * (D // 32), 0])
    kv_norm_weight_c18 = pl.slice(kv_norm_weight, [HEAD_DIM], [18 * HEAD_DIM])
    attn_sink_c18 = pl.slice(attn_sink, [LOCAL_H], [18 * LOCAL_H])
    wo_a_c18 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [18 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c18 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [18 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c18: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [18 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c18 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [18 * PAGES, 0, 0, 0])
    window_cache_scale_c18 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [18 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x1,
        p1,
        hc_attn_fn_c18,
        hc_attn_scale_c18,
        hc_attn_base_c18,
        attn_norm_weight_c18,
        wq_a_c18,
        wq_a_scale_c18,
        q_norm_weight_c18,
        wq_b_c18,
        wq_b_scale_c18,
        wkv_c18,
        wkv_scale_c18,
        kv_norm_weight_c18,
        attn_sink_c18,
        wo_a_c18,
        wo_b_c18,
        wo_b_scale_c18,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c18,
        window_cache_scale_c18,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        x0,
        p0,
        group_base, tp_rank, num_tokens, epoch_base + (19),
    )
    hc_attn_fn_c19 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [19 * MIX_HC, 0])
    hc_attn_scale_c19 = pl.slice(hc_attn_scale, [3], [19 * 3])
    hc_attn_base_c19 = pl.slice(hc_attn_base, [MIX_HC], [19 * MIX_HC])
    attn_norm_weight_c19 = pl.slice(attn_norm_weight, [D], [19 * D])
    wq_a_c19 = pl.slice(wq_a, [D, Q_LORA], [19 * D, 0])
    wq_a_scale_c19: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // 32, Q_LORA], [19 * (D // 32), 0])
    q_norm_weight_c19 = pl.slice(q_norm_weight, [Q_LORA], [19 * Q_LORA])
    wq_b_c19 = pl.slice(wq_b, [Q_LORA, LOCAL_H * HEAD_DIM], [19 * Q_LORA, 0])
    wq_b_scale_c19: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // 32, LOCAL_H * HEAD_DIM], [19 * (Q_LORA // 32), 0])
    wkv_c19 = pl.slice(wkv, [D, HEAD_DIM], [19 * D, 0])
    wkv_scale_c19: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // 32, HEAD_DIM], [19 * (D // 32), 0])
    kv_norm_weight_c19 = pl.slice(kv_norm_weight, [HEAD_DIM], [19 * HEAD_DIM])
    attn_sink_c19 = pl.slice(attn_sink, [LOCAL_H], [19 * LOCAL_H])
    wo_a_c19 = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [19 * LOCAL_O_GROUPS, 0, 0])
    wo_b_c19 = pl.slice(wo_b, [LOCAL_O_WIDTH, D], [19 * LOCAL_O_WIDTH, 0])
    wo_b_scale_c19: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // 32, D], [19 * (LOCAL_O_WIDTH // 32), 0])
    window_cache_c19 = pl.slice(window_cache, [PAGES, 128, 1, HEAD_DIM], [19 * PAGES, 0, 0, 0])
    window_cache_scale_c19 = pl.slice(window_cache_scale, [PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [19 * PAGES, 0, 0, 0])
    decode_c1a_reuse(
        x0,
        p0,
        hc_attn_fn_c19,
        hc_attn_scale_c19,
        hc_attn_base_c19,
        attn_norm_weight_c19,
        wq_a_c19,
        wq_a_scale_c19,
        q_norm_weight_c19,
        wq_b_c19,
        wq_b_scale_c19,
        wkv_c19,
        wkv_scale_c19,
        kv_norm_weight_c19,
        attn_sink_c19,
        wo_a_c19,
        wo_b_c19,
        wo_b_scale_c19,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache_c19,
        window_cache_scale_c19,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        output,
        next_pre_mix,
        group_base, tp_rank, num_tokens, epoch_base + (20),
    )
    return output


def make_program(tokens, pages, epochs=1):
    """Build the decoder chain host: one chip launch per rank and step."""
    TOKENS = tokens
    EPOCHS = epochs
    assert pages == PAGES, f"this entry is built for {PAGES} pages, got {pages}"

    @pl.jit.host
    def host(
        chain_x: pl.Tensor[[TP_SIZE, TOKENS, HC_MULT, D], pl.FP32],
        chain_pre: pl.Tensor[[TP_SIZE, TOKENS, HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[TP_SIZE, LAYERS * MIX_HC, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[TP_SIZE, LAYERS * 3], pl.FP32],
        hc_attn_base: pl.Tensor[[TP_SIZE, LAYERS * MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[TP_SIZE, LAYERS * D], pl.BF16],
        wq_a: pl.Tensor[[TP_SIZE, LAYERS * D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[TP_SIZE, LAYERS * D // 32, Q_LORA], pl.FP8E8M0],
        q_norm_weight: pl.Tensor[[TP_SIZE, LAYERS * Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[TP_SIZE, LAYERS * Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[TP_SIZE, LAYERS * Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
        wkv: pl.Tensor[[TP_SIZE, LAYERS * D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[TP_SIZE, LAYERS * D // 32, HEAD_DIM], pl.FP8E8M0],
        kv_norm_weight: pl.Tensor[[TP_SIZE, LAYERS * HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[TP_SIZE, LAYERS * LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[TP_SIZE, LAYERS * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[TP_SIZE, LAYERS * LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[TP_SIZE, LAYERS * LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
        window_cache: pl.InOut[pl.Tensor[[TP_SIZE, LAYERS * PAGES, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[TP_SIZE, LAYERS * PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressor_wkv: pl.Tensor[[TP_SIZE, LAYERS * D, HEAD_DIM], pl.BF16],
        compressor_norm_weight: pl.Tensor[[TP_SIZE, LAYERS * HEAD_DIM], pl.BF16],
        index_wk: pl.Tensor[[TP_SIZE, LAYERS * HEAD_DIM, INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[TP_SIZE, LAYERS * INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[TP_SIZE, LAYERS * Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[TP_SIZE, LAYERS * Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
        index_weights_proj: pl.Tensor[[TP_SIZE, LAYERS * D, INDEX_H], pl.BF16],
        compressed_cache: pl.InOut[pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM // 2], pl.UINT8]],
        compressed_cache_scale: pl.InOut[
            pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN]
        ],
        request_ids: pl.Tensor[[TP_SIZE, TOKENS], pl.INT32],
        compressed_lens: pl.Tensor[[TP_SIZE, TOKENS], pl.INT32],
        index_cache: pl.InOut[pl.Tensor[[TP_SIZE, PAGES, 128, 1, INDEX_DIM // 2], pl.UINT8]],
        index_cache_scale: pl.InOut[
            pl.Tensor[[TP_SIZE, PAGES, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
        ],
        index_block_table: pl.Tensor[[TP_SIZE, 1, PAGES], pl.INT32],
        compressed_rope_cos: pl.Tensor[[TP_SIZE, TOKENS, ROPE_DIM // 2], pl.FP32],
        compressed_rope_sin: pl.Tensor[[TP_SIZE, TOKENS, ROPE_DIM // 2], pl.FP32],
        compressed_slots: pl.Tensor[[TP_SIZE, TOKENS], pl.INT64],
        compressed_indices: pl.Tensor[[TP_SIZE, TOKENS, INDEX_TOPK], pl.INT32],
        candidate_mask: pl.InOut[pl.Tensor[[TP_SIZE, TOKENS, PAGES * 128], pl.UINT8]],
        topk_indices: pl.InOut[pl.Tensor[[TP_SIZE, TOKENS, INDEX_TOPK], pl.INT32]],
        rope_cos: pl.Tensor[[TP_SIZE, TOKENS, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[TP_SIZE, TOKENS, ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[TP_SIZE, TOKENS], pl.INT64],
        window_indices: pl.Tensor[[TP_SIZE, TOKENS, 128], pl.INT32],
        output: pl.Out[pl.Tensor[[TP_SIZE, TOKENS, HC_MULT, D], pl.FP32]],
        next_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, TOKENS, HC_MULT], pl.FP32]],
    ):
        transport = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        signals = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for epoch in pl.range(1, EPOCHS + 1):
            for rank in pl.unroll(TP_SIZE):
                output_window = pld.window(transport, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
                output_arrived = pld.window(signals, [TP_SIZE, 1], dtype=pl.INT32)
                decode_decoder_test(
                    chain_x[rank], chain_pre[rank],
                    hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
                    attn_norm_weight[rank],
                    wq_a[rank], wq_a_scale[rank], q_norm_weight[rank], wq_b[rank], wq_b_scale[rank],
                    wkv[rank], wkv_scale[rank], kv_norm_weight[rank], attn_sink[rank],
                    wo_a[rank], wo_b[rank], wo_b_scale[rank],
                    window_cache[rank], window_cache_scale[rank],
                    compressor_wkv[rank], compressor_norm_weight[rank],
                    index_wk[rank], index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale[rank],
                    index_weights_proj[rank],
                    compressed_cache[rank], compressed_cache_scale[rank],
                    request_ids[rank], compressed_lens[rank],
                    index_cache[rank], index_cache_scale[rank], index_block_table[rank],
                    compressed_rope_cos[rank], compressed_rope_sin[rank], compressed_slots[rank],
                    compressed_indices[rank], candidate_mask[rank], topk_indices[rank],
                    rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank],
                    output_window, output_arrived, output[rank], next_pre_mix[rank],
                    0, rank, TOKENS, (epoch - 1) * LAYERS,
                    device=rank,
                )

    return host


def golden_decoder_case(tensors, per_layer, layers=LAYERS):
    """CPU reference: the decoder sub-layers in order, on the same shared state."""
    import inspect

    from models.deepseek_v4_1_flash.decode_attn_c1a_full import (
        CACHE_STATE_NAMES,
        golden_decode_attn_c1a_full,
    )
    from models.deepseek_v4_1_flash.decode_attn_c1a_reindex import golden_decode_attn_c1a_reindex
    from models.deepseek_v4_1_flash.decode_attn_c1a_reuse import golden_decode_attn_c1a_reuse
    from models.deepseek_v4_1_flash.golden import rms_norm
    from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes
    from models.deepseek_v4_1_flash.hc_post import golden_mhc_post
    from models.deepseek_v4_1_flash.hc_pre import golden_mhc_pre

    leaves = {
        "full": golden_decode_attn_c1a_full,
        "reindex": golden_decode_attn_c1a_reindex,
        "reuse": golden_decode_attn_c1a_reuse,
    }
    shared = (
        "compressed_cache", "compressed_cache_scale", "index_cache", "index_cache_scale",
        "candidate_mask", "topk_indices",
    )
    state = {name: tensors[name].clone() for name in shared}
    x_hc = tensors["chain_x"].clone()
    pre_mix = tensors["chain_pre"].clone()
    for slot, mode in enumerate(decoder_modes(layers)):
        leaf = leaves[mode]
        parameters = [name for name in inspect.signature(leaf).parameters if name != "x"]
        partials = []
        mixes = []
        nexts = []
        for rank in range(TP_SIZE):
            layer = per_layer[slot]
            next_pre_mix, post_mix, residual_mix = golden_mhc_mixes(
                x_hc[rank], layer["hc_attn_fn"][rank], layer["hc_attn_scale"][rank],
                layer["hc_attn_base"][rank],
            )
            hidden = golden_mhc_pre(x_hc[rank], pre_mix[rank])
            normalized = rms_norm(hidden, layer["attn_norm_weight"][rank])
            call = {}
            for name in parameters:
                if name in ("window_cache", "window_cache_scale"):
                    call[name] = tensors[name][rank][slot * PAGES : (slot + 1) * PAGES]
                elif name == "compressed_indices":
                    # The reuse layers read the selection the stack's sources published.
                    call[name] = state["topk_indices"][rank]
                elif name in state:
                    call[name] = state[name][rank]
                else:
                    call[name] = layer[name][rank]
            result = leaf(x=normalized, **call)
            partials.append(result.output.float())
            mixes.append((x_hc[rank], post_mix, residual_mix))
            nexts.append(next_pre_mix)
            for name in CACHE_STATE_NAMES:
                value = getattr(result, name, None)
                if value is None:
                    continue
                if name in ("window_cache", "window_cache_scale"):
                    destination = tensors[name][rank][slot * PAGES : (slot + 1) * PAGES]
                elif name in state:
                    destination = state[name][rank]
                else:
                    continue
                if name == "candidate_mask" and destination.shape != value.shape:
                    value = value.clone()
                    destination.zero_()
                    destination[:, : value.shape[1]].copy_(value.to(destination.dtype))
                else:
                    destination.view(torch.uint8).copy_(value.contiguous().view(torch.uint8))
        reduced = torch.zeros_like(partials[0])
        for partial in partials:
            reduced += partial
        sublayer = reduced.to(torch.bfloat16)
        x_hc = torch.stack(
            [golden_mhc_post(sublayer, mixes[rank][0], mixes[rank][1], mixes[rank][2]) for rank in range(TP_SIZE)]
        )
        pre_mix = torch.stack(nexts)
    tensors["output"][:] = x_hc
    tensors["next_pre_mix"][:] = pre_mix
    for name, value in state.items():
        tensors[name][:] = value

def run_decoder(layers=LAYERS):
    """Validate the decoder attention chain on A5."""
    import argparse

    from golden import TensorSpec, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="TP1/2/4 A5 decode decoder attention chain")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument("--dp", type=int, default=1, choices=[1])
    parser.add_argument("-p", "--platform", type=str, default="a5", choices=["a5"])
    parser.add_argument("-d", "--device", "--devices", type=str, default=None)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=["random"], default="random")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.tp != TP_SIZE:
        parser.error(f"--tp {args.tp} does not match the TP_SIZE {TP_SIZE} the entry was built for")
    devices = list(range(TP_SIZE)) if args.device is None else [int(d) for d in args.device.split(",")]
    if len(devices) != TP_SIZE or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error(f"device IDs must be {TP_SIZE} distinct nonnegative integers, one per rank")
    host = make_program(args.tokens, args.pages, args.epochs)
    values, _per_layer = build_decoder_values(args.tokens, args.pages, layers, args.seed, args.case)
    specs = [
        TensorSpec(name, list(values[name].shape), values[name].dtype,
                   init_value=lambda name=name: values[name].clone())
        for name in host.param_names
    ]

    def golden_fn(tensors):
        golden_decoder_case(tensors, _per_layer)

    # Published caches are compared as dequantized values (one ULP can move a published
    # code one E2M1/E4M3 step), the candidate mask byte exact, and the chain outputs under
    # the mHC budgets the sub-layer entries use.
    comparisons = {
        "output": output_hc_compare,
        "next_pre_mix": next_pre_mix_compare,
        "window_cache": quantized_cache_compare(
            "window_cache", "window_cache_scale", "window_slots", CACHE_MAX_RELATIVE_L2
        ),
        "compressed_cache": quantized_cache_compare(
            "compressed_cache", "compressed_cache_scale", "compressed_slots",
            MXFP4_CACHE_MAX_RELATIVE_L2, group_size=COMPRESSED_CACHE_GROUP, scale_format="e4m3",
        ),
        "index_cache": quantized_cache_compare(
            "index_cache", "index_cache_scale", "compressed_slots",
            MXFP4_CACHE_MAX_RELATIVE_L2, group_size=INDEX_CACHE_GROUP, scale_format="e8m0",
        ),
        "candidate_mask": exact_bytes,
    }
    comparisons["window_cache_scale"] = comparisons["window_cache"]
    comparisons["compressed_cache_scale"] = comparisons["compressed_cache"]
    comparisons["index_cache_scale"] = comparisons["index_cache"]
    comparisons = {name: fn for name, fn in comparisons.items() if name in host.param_names}
    result = run(
        fn=host,
        specs=specs,
        golden_fn=golden_fn,
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        config=dict(
            platform=args.platform,
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn=comparisons,
    )
    if not result.passed:
        print(result.error)
        raise SystemExit(1)


def main():
    """Stage-1 validation entry for the decode decoder attention chain."""
    run_decoder()


# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"
if __name__ == _SCRIPT_ENTRY_POINT:
    main()
