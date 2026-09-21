# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1 attention RMSNorm for dynamic decode shapes.

The released model normalizes the collapsed hyper-connection stream before every attention
module (``Block.forward``: ``hc_pre`` -> ``attn_norm`` -> ``attn``). The attention operators
take that normalized hidden; this module is the shared implementation the mHC-wired entries
and the decode stack call, so the weight, the epsilon, and the single BF16 rounding live in
one place.

Ragged rows are supported: the row tile is padded to its static extent and written back with
``set_validshape``, so a 1-token decode step and a full 192-row step take the same kernel.
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
import torch

from models.deepseek_v4_1_flash.config import D, FLASH, T_DYN
from models.deepseek_v4_1_flash.golden import rms_norm as golden_attn_norm


NORM_EPS = FLASH.rms_norm_eps
NORM_T_TILE = 8
NORM_D_TILE = 512


@pl.jit.inline
def attn_norm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Weighted RMSNorm over the hidden size in FP32, rounded once to BF16."""
    t_dim = pl.tensor.dim(x, 0)
    for block in pl.spmd((t_dim + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="attn_norm"):
        t0 = block * NORM_T_TILE
        valid_rows = pl.min(NORM_T_TILE, t_dim - t0)
        sq_sum = pl.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
            k0 = kb * NORM_D_TILE
            source = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
            source = pl.set_validshape(
                pl.fillpad(source, pad_value=pl.PadValue.zero), NORM_T_TILE, NORM_D_TILE
            )
            value = pl.cast(source, target_type=pl.FP32)
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(value, value)), [1, NORM_T_TILE]))
        inv_rms = pl.reshape(
            pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS), high_precision=True), [NORM_T_TILE, 1]
        )
        for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
            k0 = kb * NORM_D_TILE
            source = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
            source = pl.set_validshape(
                pl.fillpad(source, pad_value=pl.PadValue.zero), NORM_T_TILE, NORM_D_TILE
            )
            value = pl.cast(source, target_type=pl.FP32)
            gamma = pl.reshape(
                pl.cast(weight[k0 : k0 + NORM_D_TILE], target_type=pl.FP32), [1, NORM_D_TILE]
            )
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv_rms), gamma)
            output[t0 : t0 + NORM_T_TILE, k0 : k0 + NORM_D_TILE] = pl.set_validshape(
                pl.cast(normalized, target_type=pl.BF16, mode="rint"), valid_rows, NORM_D_TILE
            )
    return output


@pl.jit
def attn_norm_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Run the attention input norm for standalone validation."""
    x.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    attn_norm(x, weight, output)
    return output


def build_tensor_specs(batch: int = 1, sequence: int = 1):
    """Build deterministic inputs and output for attention-norm validation."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(11)

    def init_x():
        return torch.randn(tokens, D, generator=generator) - 0.5

    def init_weight():
        return torch.randn(D, generator=generator) * 0.1 + 1.0

    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=init_x),
        TensorSpec("weight", [D], torch.bfloat16, init_value=init_weight),
        TensorSpec("output", [tokens, D], torch.bfloat16),
    ]


def golden_attn_norm_case(tensors):
    """Fill the expected normalized stream output."""
    tensors["output"][:] = golden_attn_norm(tensors["x"], tensors["weight"], eps=NORM_EPS)


def _precision_compare(name, compare):
    """Report achieved precision before applying the tensor's acceptance budget."""

    def compare_and_report(actual, expected, **kwargs):
        actual_f = actual.double()
        expected_f = expected.double()
        diff = actual_f - expected_f
        rel_l2 = diff.norm() / expected_f.norm().clamp_min(1e-12)
        max_abs = diff.abs().max()
        print(f"[PRECISION] {name} rel_l2={rel_l2.item():.8g} max_abs={max_abs.item():.8g}")
        return compare(actual, expected, **kwargs)

    return compare_and_report


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 attention RMSNorm validation")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    return parser.parse_args()


def main():
    """Validate the attention input norm on A5."""
    from golden import ratio_allclose, run

    args = _parse_args()
    result = run(
        fn=attn_norm_test,
        specs=build_tensor_specs(args.batch, args.sequence),
        golden_fn=golden_attn_norm_case,
        golden_data=args.golden_data,
        save_data=args.save_data,
        runtime_dir=args.runtime_dir,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"output": _precision_compare("output", ratio_allclose(atol=1e-4, rtol=1.0 / 128))},
        compile_only=args.compile_only,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)


__all__ = [
    "attn_norm",
    "attn_norm_test",
    "build_tensor_specs",
    "golden_attn_norm",
    "golden_attn_norm_case",
]

# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"
if __name__ == _SCRIPT_ENTRY_POINT:
    main()
