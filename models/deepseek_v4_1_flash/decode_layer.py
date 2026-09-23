# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode Block golden composition and mode-selected program dispatch."""

import argparse
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import pypto.language as pl
import pypto.language.distributed as pld

from models.deepseek_v4_1_flash import config as C, moe as M
from models.deepseek_v4_1_flash.attention_tp import OUTPUT_T_DYN

from models.deepseek_v4_1_flash.attention_common import AttentionGoldenResult
from models.deepseek_v4_1_flash.decode_layer_plan import (
    REPRESENTATIVE_LAYER_IDS,
    DecodeLayerKind,
    DecodeLayerPlan,
    load_decode_attention_module,
    resolve_decode_layer_plan,
)
from models.deepseek_v4_1_flash.golden import rms_norm
from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes
from models.deepseek_v4_1_flash.hc_post import golden_mhc_post
from models.deepseek_v4_1_flash.hc_pre import golden_mhc_pre
from models.deepseek_v4_1_flash.moe import golden_moe, moe


@dataclass(frozen=True)
class DecodeLayerGoldenResult:
    """Block outputs, intermediate boundaries, and attention state updates."""

    output: torch.Tensor
    next_pre_mix: torch.Tensor
    attention_input: torch.Tensor
    attention_output: torch.Tensor
    attention_hidden: torch.Tensor
    ffn_input: torch.Tensor
    ffn_output: torch.Tensor
    attention: AttentionGoldenResult


_MOE_KERNEL_READY = False


def _attention_golden(mode, kind: DecodeLayerKind):
    golden = getattr(mode, "ATTENTION_GOLDEN", None) or getattr(mode, "GOLDEN", None)
    if golden is not None:
        return golden
    names = {
        DecodeLayerKind.C1A_FULL: "golden_decode_attn_c1a_full",
        DecodeLayerKind.C1A_REINDEX: "golden_decode_attn_c1a_reindex",
        DecodeLayerKind.C1A_REUSE: "golden_decode_attn_c1a_reuse",
    }
    try:
        return getattr(mode, names[kind])
    except (AttributeError, KeyError) as error:
        raise AttributeError(f"{mode.__name__} does not expose the {kind.name} Attention golden") from error


def attention_half_skip_reason(layer_id: int) -> str | None:
    """Return the selected mode's Attention-only readiness reason."""
    plan = resolve_decode_layer_plan(layer_id)
    mode = load_decode_attention_module(plan.kind)
    skip_reason = getattr(mode, "skip_reason", None)
    if skip_reason is not None:
        return skip_reason(layer_id)
    return None if getattr(mode, "KERNEL_READY", True) else f"{plan.kind.name} attention kernel is pending"


def decode_layer_kernel_skip_reason(layer_id: int) -> str | None:
    """Return dependencies that prevent full Block device compilation."""
    missing = []
    attention_reason = attention_half_skip_reason(layer_id)
    if attention_reason:
        missing.append(attention_reason)
    if not _MOE_KERNEL_READY:
        missing.append("EP8 MoE kernel integration")
    return None if not missing else "decode_layer requires " + " and ".join(missing)


def decode_layer_attention_inputs(layer_id: int) -> tuple[str, ...]:
    """Return the exact golden inputs consumed by the resolved attention mode."""
    plan = resolve_decode_layer_plan(layer_id)
    mode = load_decode_attention_module(plan.kind)
    golden = _attention_golden(mode, plan.kind)
    return tuple(inspect.signature(golden).parameters)


def _select_inputs(function, values: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    selected = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if name in overrides:
            selected[name] = overrides[name]
        elif name in values:
            selected[name] = values[name]
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            raise KeyError(f"missing {function.__name__} input {name}")
    return selected


def golden_decode_layer(
    layer_id: int,
    x_hc: torch.Tensor,
    incoming_pre_mix: torch.Tensor,
    hc_attn_fn: torch.Tensor,
    hc_attn_scale: torch.Tensor,
    hc_attn_base: torch.Tensor,
    attn_norm_weight: torch.Tensor,
    hc_ffn_fn: torch.Tensor,
    hc_ffn_scale: torch.Tensor,
    hc_ffn_base: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    attention_inputs: Mapping[str, Any],
    moe_inputs: Mapping[str, Any],
    num_tokens: int | None = None,
) -> DecodeLayerGoldenResult:
    """Evaluate the mHC-Attention-mHC-MoE-mHC Block order."""
    if num_tokens is not None and num_tokens != x_hc.shape[0]:
        raise ValueError("Block golden currently requires all capacity rows to be active")
    plan = resolve_decode_layer_plan(layer_id)
    mode = load_decode_attention_module(plan.kind)
    attention_golden = _attention_golden(mode, plan.kind)
    attn_pre, attn_post, attn_residual = golden_mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base)
    attention_input = golden_mhc_pre(x_hc, incoming_pre_mix)
    normalized_attention = rms_norm(attention_input, attn_norm_weight)
    attention_kwargs = _select_inputs(attention_golden, attention_inputs, {"x": normalized_attention})
    attention = attention_golden(**attention_kwargs)
    attention_hidden = golden_mhc_post(attention.output, x_hc, attn_post, attn_residual)

    next_pre_mix, ffn_post, ffn_residual = golden_mhc_mixes(
        attention_hidden, hc_ffn_fn, hc_ffn_scale, hc_ffn_base
    )
    ffn_input = golden_mhc_pre(attention_hidden, attn_pre)
    # ``golden_moe`` owns the FFN mHC composition and mutates its tensor map;
    # passing its complete map keeps the reference boundary aligned with the
    # device MoE entry and avoids wrapping it in a second mHC sequence.
    moe_tensors = dict(moe_inputs)
    moe_tensors.update(
        x_hc=attention_hidden,
        pre_mix=attn_pre,
        hc_ffn_fn=hc_ffn_fn,
        hc_ffn_scale=hc_ffn_scale,
        hc_ffn_base=hc_ffn_base,
        norm_weight=ffn_norm_weight,
        x=ffn_input,
    )
    if num_tokens is not None:
        moe_tensors["num_tokens"] = num_tokens
    if (
        attention_hidden.ndim == 4
        and attention_hidden.shape[0] == EP_SIZE
        and attention_hidden.shape[1] == MOE_TOKENS
    ):
        golden_moe(moe_tensors)
        ffn_output = moe_tensors["x_next"]
    else:
        # The compact CPU Block fixture exercises ordering and mHC boundaries;
        # its deliberately small expert tensors do not satisfy the released
        # EP8 packed-weight ABI. Keep that fixture independent of device-sized
        # expert buffers while the full MoE golden remains authoritative for
        # integration-sized inputs.
        ffn_output = golden_mhc_post(
            torch.zeros_like(ffn_input, dtype=torch.bfloat16),
            attention_hidden,
            ffn_post,
            ffn_residual,
        )
    output = ffn_output
    return DecodeLayerGoldenResult(
        output=output,
        next_pre_mix=next_pre_mix,
        attention_input=attention_input,
        attention_output=attention.output,
        attention_hidden=attention_hidden,
        ffn_input=ffn_input,
        ffn_output=ffn_output,
        attention=attention,
    )


def make_decode_layer_program(layer_id, world_size, epochs, *, stage="attention", specs=None):
    """Dispatch program construction to the selected mode before JIT discovery."""
    if stage == "attention":
        if specs is None:
            raise ValueError("Attention host entry requires tensor specs")
        plan = resolve_decode_layer_plan(layer_id)
        mode = load_decode_attention_module(plan.kind)
        if getattr(mode, "COMPOSITION_ABI", "spec-driven") == "native":
            raise NotImplementedError(
                f"{plan.kind.name} uses its mode-native make_program(tokens, pages, epochs) entry"
            )
        return mode.make_program(world_size, epochs, specs)
    if stage == "block":
        reason = decode_layer_kernel_skip_reason(layer_id)
        raise NotImplementedError(reason or "Block hardware fixture is pending integration")
    raise ValueError(f"unknown decode stage: {stage}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=("attention", "block"), default="attention")
    parser.add_argument("--cpu-golden", action="store_true")
    parser.add_argument("--layer-id", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.cpu_golden:
        if args.stage != "block":
            parser.error("--cpu-golden requires --stage block")
        from models.deepseek_v4_1_flash._golden_smoke import (
            run_decode_layer_goldens,
            run_two_layer_decode_chain,
        )

        run_decode_layer_goldens(golden_decode_layer, REPRESENTATIVE_LAYER_IDS.values())
        run_two_layer_decode_chain(golden_decode_layer)
        return
    if args.stage == "block":
        parser.error(decode_layer_kernel_skip_reason(args.layer_id) or "Block hardware fixture is pending")
    plan = resolve_decode_layer_plan(args.layer_id)
    mode = load_decode_attention_module(plan.kind)
    mode.main()


__all__ = [
    "DecodeLayerGoldenResult",
    "DecodeLayerKind",
    "DecodeLayerPlan",
    "REPRESENTATIVE_LAYER_IDS",
    "attention_half_skip_reason",
    "decode_layer_attention_inputs",
    "decode_layer_kernel_skip_reason",
    "golden_decode_layer",
    "make_decode_layer_program",
    "resolve_decode_layer_plan",
]



D = M.D
DP_SIZE = C.DP_SIZE
EP_SIZE = M.EP_SIZE
HC_MULT = M.HC_MULT
MOE_INTER = C.MOE_INTER
MOE_TOKENS = C.MOE_TOKENS
N_EXPERTS = C.N_EXPERTS
ROUTE_T_DYN = C.ROUTE_T_DYN
TP_SIZE = C.TP_SIZE
AUX_WIDTH = M.AUX_WIDTH
HC_DIM = M.HC_DIM
MIX_HC = M.MIX_HC
MX_GROUP = M.MX_GROUP
MX_PACKED_LANE_COLS = M.MX_PACKED_LANE_COLS
MX_W1_PACKED_ROWS = M.MX_W1_PACKED_ROWS
MX_W2_PACKED_ROWS = M.MX_W2_PACKED_ROWS
MX_W3_PACKED_ROWS = M.MX_W3_PACKED_ROWS
N_LOCAL_EXPERTS = M.N_LOCAL_EXPERTS
RECV_MAX = M.RECV_MAX
ROUTE_WIDTH = M.ROUTE_WIDTH

DECODER_SLAB = ((C.DECODER_CAPACITY + C.TP_SIZE - 1) // C.TP_SIZE) if C.DECODER_CAPACITY else 1

# The padded MoE workspace is a different extent from the owner slab the
# attention half works on (``MOE_TOKENS`` versus ``DECODER_SLAB`` rows), so it
# needs its own dynamic symbol: one DynVar cannot be bound to two extents inside
# a single call.
MOE_PAD_DYN = pl.dynamic("V41_MOE_PAD_DYN")

@pl.jit
def decoder_moe_rank(
    x_hc: pl.Tensor[[OUTPUT_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[OUTPUT_T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    next_pre_mix: pl.Out[pl.Tensor[[OUTPUT_T_DYN, HC_MULT], pl.FP32]],
    x_mixed: pl.Out[pl.Tensor[[OUTPUT_T_DYN, D], pl.BF16]],
    x_next: pl.Out[pl.Tensor[[OUTPUT_T_DYN, HC_MULT, D], pl.FP32]],
    padded_x: pl.InOut[pl.Tensor[[MOE_PAD_DYN, HC_MULT, D], pl.FP32]],
    padded_pre: pl.InOut[pl.Tensor[[MOE_PAD_DYN, HC_MULT], pl.FP32]],
    padded_next: pl.InOut[pl.Tensor[[MOE_PAD_DYN, HC_MULT, D], pl.FP32]],
    padded_next_pre: pl.InOut[pl.Tensor[[MOE_PAD_DYN, HC_MULT], pl.FP32]],
    padded_mixed: pl.InOut[pl.Tensor[[MOE_PAD_DYN, D], pl.BF16]],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    recycle: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    active_tokens: pl.Tensor[[DP_SIZE], pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Pad owner rows and rendezvous before recycling any EP communication window.

    Every rank reaches this call through the preceding block's output. The
    all-rank entry barrier therefore acknowledges completion of every previous
    combine reader before any rank can publish into the reused windows.  The
    padded workspace is caller-owned (the host allocates one ``MOE_TOKENS``-row
    buffer per rank and reuses it for every layer), so its token dimension stays
    the same DynVar the attention half uses.
    """
    slab = pl.tensor.dim(x_hc, 0)
    active = pl.read(active_tokens, [ep_rank // TP_SIZE])
    local_count = pl.max(0, pl.min(slab, active - (ep_rank % TP_SIZE) * slab))
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decoder_ep_recycle") as reuse_ready:
        for peer in pl.range(EP_SIZE):
            if peer != ep_rank:
                pld.system.notify(recycle, peer=peer, offsets=[ep_rank, 0],
                                  value=1, op=pld.NotifyOp.AtomicAdd)
        for peer in pl.range(EP_SIZE):
            if peer != ep_rank:
                pld.system.wait(recycle, offsets=[peer, 0], expected=moe_epoch,
                                cmp=pld.WaitCmp.Ge)
    with pl.spmd(MOE_TOKENS, name_hint="decoder_moe_pad", deps=[reuse_ready]):
        row = pl.tile.get_block_idx()
        if row < local_count:
            padded_x[row:row + 1, :, :] = x_hc[row:row + 1, :, :]
        else:
            padded_x[row:row + 1, :, :] = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)
    # The pre-mix row is four FP32 wide (16 bytes), below the 32-byte tile row the
    # backend requires, so copy/initialize it with scalar writes from one block.
    with pl.spmd(1, name_hint="decoder_moe_pad_pre", deps=[reuse_ready]):
        pad_first = pl.tile.get_block_idx()
        for pad_row in pl.range(pad_first, MOE_TOKENS):
            for pad_col in pl.range(HC_MULT):
                if pad_row < local_count:
                    pl.write(padded_pre, [pad_row, pad_col], pl.read(pre_mix, [pad_row, pad_col]))
                else:
                    pl.write(padded_pre, [pad_row, pad_col], pl.cast(0.0, pl.FP32))
    moe(padded_x, padded_pre, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_weight, gate_weight, correction_bias, routed_w1, routed_w1_scale, routed_w2, routed_w2_scale, routed_w3, routed_w3_scale, mxfp4_pair_lut, shared_w1, shared_w1_scale, shared_w2, shared_w2_scale, shared_w3, shared_w3_scale, padded_next_pre, padded_mixed, padded_next, recv_meta, recv_x, recv_scale, recv_weights, recv_routes, arrived, data_arrived, routed_output, combine_arrived, local_count, ep_rank, moe_epoch)
    with pl.spmd(slab, name_hint="decoder_moe_unpad"):
        row = pl.tile.get_block_idx()
        if row < local_count:
            x_next[row:row + 1, :, :] = padded_next[row:row + 1, :, :]
            x_mixed[row:row + 1, :] = padded_mixed[row:row + 1, :]
        else:
            x_next[row:row + 1, :, :] = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)
            x_mixed[row:row + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
    with pl.spmd(1, name_hint="decoder_moe_unpad_pre"):
        unpad_first = pl.tile.get_block_idx()
        for unpad_row in pl.range(unpad_first, slab):
            for unpad_col in pl.range(HC_MULT):
                if unpad_row < local_count:
                    pl.write(next_pre_mix, [unpad_row, unpad_col], pl.read(padded_next_pre, [unpad_row, unpad_col]))
                else:
                    pl.write(next_pre_mix, [unpad_row, unpad_col], pl.cast(0.0, pl.FP32))
    return x_next, next_pre_mix


if __name__ == "__main__":
    main()
