# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for the DeepSeek-V4.1 Flash decode decoder (layers 20-39)."""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

module = sys.modules.get("pypto")
HAS_PYPTO = (
    not getattr(module, "__pypto_stub__", False)
    if module is not None
    else importlib.util.find_spec("pypto") is not None
)
requires_pypto = pytest.mark.skipif(not HAS_PYPTO, reason="model imports require real PyPTO")

MODEL_DIR = Path(__file__).parents[2] / "models" / "deepseek_v4_1_flash"
ENTRY = "decode_decoder.py"
RANK_ENTRIES = {
    "decode_c1a_full.py": "decoder_c1a_full_rank",
    "decode_c1a_reindex.py": "decoder_c1a_reindex_rank",
    "decode_c1a_reuse.py": "decoder_c1a_reuse_rank",
}


def _tree(name: str, directory: Path = MODEL_DIR) -> ast.Module:
    return ast.parse((directory / name).read_text())


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls(node: ast.AST) -> list[str]:
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            names.append(child.func.id)
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            names.append(child.func.attr)
    return names


# ---------------------------------------------------------------------------
# Schedule and capacity contracts
# ---------------------------------------------------------------------------


@requires_pypto
def test_decoder_schedule_composes_the_released_source_layers():
    from models.deepseek_v4_1_flash.decode_layer_plan import DecodeLayerKind, resolve_decoder_plan

    plan = resolve_decoder_plan(20, 40)
    assert [step.layer_id for step in plan] == list(range(20, 40))
    kinds = {step.layer_id: step.kind for step in plan}
    assert kinds[20] is DecodeLayerKind.C1A_FULL
    # Ratio-1 layers keep the single released compressed-KV source (layer 20) and
    # the five index sources own a fresh selection; every other layer reuses one.
    assert {step.kv_source_layer_id for step in plan} == {20}
    index_sources = {step.layer_id for step in plan if step.index_source_layer_id == step.layer_id}
    assert index_sources == {20, 24, 28, 32, 36}
    reindex = index_sources - {20}
    for layer_id in reindex:
        assert kinds[layer_id] is DecodeLayerKind.C1A_REINDEX
    reuse = set(range(21, 40)) - reindex
    assert {kinds[layer] for layer in reuse} == {DecodeLayerKind.C1A_REUSE}
    assert {step.index_source_layer_id for step in plan if step.kind is DecodeLayerKind.C1A_REUSE} <= index_sources


@requires_pypto
def test_decoder_segment_must_stay_inside_the_decoder_layers():
    from models.deepseek_v4_1_flash.decode_layer_plan import resolve_decoder_plan

    with pytest.raises(ValueError):
        resolve_decoder_plan(19, 40)
    with pytest.raises(ValueError):
        resolve_decoder_plan(20, 41)
    with pytest.raises(ValueError):
        resolve_decoder_plan(30, 30)


@requires_pypto
def test_decoder_layout_matches_the_planned_owner_mapping():
    from models.deepseek_v4_1_flash.decode_common import DecoderLayout

    layout = DecoderLayout(8, tp=4, ep=8)
    assert (layout.dp, layout.slab, layout.moe_capacity) == (2, 2, 16)
    assert layout.receive_capacity == layout.ep * layout.moe_capacity
    # Compact equal-size batches: both DP groups hold the full capacity.
    assert layout.counts([8, 8]).tolist() == [2, 2, 2, 2, 2, 2, 2, 2]
    # A partial prefix keeps the same physical slab; owners past the active range are empty.
    assert layout.counts([5, 1]).tolist() == [2, 2, 1, 0, 1, 0, 0, 0]
    with pytest.raises(ValueError):
        layout.counts([9, 8])


def test_decoder_host_selects_one_capacity_before_importing_kernels():
    """The entry seeds ``--decoder-capacity`` so a bare pytest run is importable."""
    source = (MODEL_DIR / ENTRY).read_text()
    seed = source.index('sys.argv += ["--decoder-capacity"')
    assert seed < source.index("import torch") < source.index("from models.deepseek_v4_1_flash import config")
    tree = _tree(ENTRY)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"make_program", "build_validation_values", "golden_decoder", "validate", "main"} <= names
    assert "argparse" in source and "--decoder-capacity" in source


@requires_pypto
def test_decoder_metadata_assigns_moe_owners_from_the_slab():
    from models.deepseek_v4_1_flash.metadata import build_forward_metadata

    slab = 2
    starts = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    history = torch.tensor([5, 5, 5, 5], dtype=torch.int32)
    table = torch.zeros(4, 1, dtype=torch.int32)
    metadata = build_forward_metadata(
        starts, history, table, {20: table}, {}, source_layer_ids=(20,), owner_slab_size=slab
    )
    assert metadata.moe_token_owners.tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        build_forward_metadata(
            starts, history, table, {20: table}, {}, source_layer_ids=(21,), owner_slab_size=slab
        )
    # One owner slab covers ``slab * TP`` rows; a longer batch cannot be placed.
    with pytest.raises(ValueError):
        build_forward_metadata(
            torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32),
            torch.tensor([5, 5, 5, 5, 5], dtype=torch.int32),
            table,
            {20: table},
            {},
            source_layer_ids=(20,),
            owner_slab_size=slab,
        )


# ---------------------------------------------------------------------------
# Device entry contracts
# ---------------------------------------------------------------------------


def test_decoder_rank_entries_reuse_the_sharded_composition():
    for name, entry in RANK_ENTRIES.items():
        tree = _tree(name)
        function = _function(tree, entry)
        calls = _calls(function)
        sharded = next(call for call in calls if call.startswith("decode_c1a_") and call.endswith("_sharded"))
        assert sharded in calls
        assert entry.endswith("_rank")


def test_decoder_moe_entry_pads_into_caller_workspace():
    tree = _tree("decode_layer.py")
    function = _function(tree, "decoder_moe_rank")
    annotations = {argument.arg: ast.unparse(argument.annotation) for argument in function.args.args}
    for name in ("padded_x", "padded_pre", "padded_next", "padded_next_pre", "padded_mixed"):
        assert name in annotations
        assert annotations[name].startswith("pl.InOut["), name
    calls = _calls(function)
    assert "moe" in calls and "create_tensor" not in calls
    assert "local_count" in {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}


def test_decoder_golden_pads_the_moe_half_like_the_device():
    tree = _tree(ENTRY)
    golden = _function(tree, "golden_decoder")
    calls = _calls(golden)
    assert "golden_moe" in calls and "local_counts" in calls
    text = ast.unparse(golden)
    # Every C1A mode feeds the chain reference; the padded workspace is filled by the golden itself.
    for expected in ("golden_decode_c1a_full_case", "golden_decode_c1a_reindex_case", "golden_decode_c1a_reuse_case"):
        assert expected in (MODEL_DIR / ENTRY).read_text()
    assert "MOE_TOKENS" in text
    # The independent chain feeds its own outputs forward, never device results.
    assert "tensors['output']" in text and "tensors['attention_hidden']" in text


def test_decoder_entry_is_an_a5_ci_case():
    tree = _tree(ENTRY)
    sentinel = _function(tree, "test_precision")
    assert any(
        isinstance(decorator, ast.Call) and getattr(decorator.func, "attr", "") == "parametrize"
        for decorator in sentinel.decorator_list
    )
    source = (MODEL_DIR / ENTRY).read_text()
    assert "_SCRIPT_ENTRY_POINT" in source
    assert "if __name__ == '__main__':" not in source
    assert "# ci: a5" in source
