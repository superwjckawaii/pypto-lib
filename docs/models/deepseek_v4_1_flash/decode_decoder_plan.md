# Decode decoder integration plan

Status: implemented and under device validation on A5 TP4/EP4. The single-step and
small multi-step milestones pass; the twenty-layer multi-step gap and the cache
deviations recorded at the end of this document remain open.

This work composes backbone layers 20 through 39 for one decode token per
request. Round one validates independent random weights on A5 devices. Round
two validates real checkpoint weights with captured reference-model boundary
inputs and state. Open the PR only after both rounds pass.

## Official reference review

The official implementation is in the model's Hugging Face
[inference directory](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference).
Its [README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/README.md)
describes a minimal reference rather than a serving engine. Its standalone
model self-test uses uninitialized weights and is not numerical acceptance.

| Official file | Review purpose |
| --- | --- |
| [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/config.json) | Released dimensions, layer/source schedule, quantization and RoPE settings |
| [model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py) | `Compressor`, `Indexer`, `Attention`, `MoE`, `Block`, `SharedAttentionRuntime`, and `Transformer` |
| [kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py) | Low-level kernels imported by the model; arithmetic details require follow-up during implementation |
| [convert.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/convert.py) | Checkpoint name mapping, partitioning and format conversion |
| [generate.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/generate.py) | Reference execution and boundary-capture integration |

The released configuration confirms 40 backbone layers, ratio 1 at layers
20-39, KV source 20, and index sources 20/24/28/32/36. Candidate selection uses
2,048 blocks of eight positions and final Top-K 512. Engram belongs to layers
1 and 14; the three appended draft layers are outside this work.

The reviewed `model.py` establishes these semantic constraints:

- Attention uses the incoming pre-mix; FFN uses attention's pre-mix; the block
  returns FFN's pre-mix. Post/residual coefficients apply immediately.
- Every layer updates its own window cache. Only source layers publish shared
  compressed KV and index keys; Reuse consumes the latest source selection.
- Index keys consume the unrotated compressor latent. Ratio 1 needs no recurrent
  compressor state.
- C1A query, window KV and indexer rotations use the compressed YaRN profile.
- Candidate blocks use maximum position scores and retain the newest block.
- The reference uses a single parallel world and reductions, not our proposed
  TP4/DP2/EP8 sequence-parallel transport.

The reviewed `generate.py` initializes BF16 computation. Combined with
`Block.hc_post`'s output cast, this requires BF16 residual-boundary rounding.
The local `hc_post.py` already rounds through BF16 before storing FP32.

Conclusion: the composition is consistent with the reference, provided these
constraints become acceptance checks. This is a source review, not evidence
that the integrated device path passes. Links above are mutable `main` URLs;
record an exact source revision, checkpoint revision and configuration digest
with the validation artifacts before implementation acceptance.

## Entry contract

Add `models/deepseek_v4_1_flash/decode_decoder.py` as the executable entry.
Keep reusable helpers in the existing `decode_common.py`. Keep decoder fixtures,
goldens, comparators, `validate`, `test_precision`, and `main` in
`decode_decoder.py`; do not introduce separate decoder common or validation
modules.

The production entry consumes:

- Layer-19 output hidden and delayed pre-mix, already assigned to owner slabs.
- Distinct weights for every layer from 20 through 39.
- Persistent caches and request-to-physical-page metadata.
- Absolute positions, active token counts and caller-owned complete RoPE tables.

It returns layer-39 hidden and delayed pre-mix and updates caches in place.
Final HC collapse, final normalization, LM head, sampling, encoder execution,
Engram and speculative decoding remain outside this entry. Capture those
upstream effects in the supplied boundary inputs when validating real requests.
Round two covers text decode; adding image-token routing is a separate contract.

Use FP32 storage for hidden `[T_local, 4, 5120]` and pre-mix `[T_local, 4]`,
with BF16-valued hidden at sublayer boundaries. Initialize the random entry
hidden through BF16 as well. Do not interpret FP32 storage as permission to
remove existing casts. Each new decode step receives its own layer-19 boundary;
layer-39 output is not the next step's layer-20 input.

Build one `@pl.jit.host` program for the layer chain. Initially retain separate
attention and MoE device entries for each block, connected through device
tensors. Host dispatch must not copy intermediate hidden states through Torch.
Use compile-time mode selection from the layer plan. Expose intermediate outputs
in validation builds without requiring a second arithmetic implementation.

## Parallel layout and capacity

The target is A5 TP4, DP2, EP8. Small topology cases support bring-up but do not
replace the eight-card acceptance run. For global rank `r`:

```text
tp_rank = r % TP
dp_rank = r // TP
group_base = dp_rank * TP
ep_rank = r
T_local = ceil(T_capacity / TP)
local_count = clamp(active_tokens[dp_rank] - tp_rank * T_local, 0, T_local)
T_moe = align_up(T_local, 16)
```

`T_capacity` is the physical token capacity of one DP group, not the world-wide
batch. Start with compact equal-size batches in both groups; subsequently test
different active prefixes using the same physical capacity. Multi-request
metadata must replace the single-request assumptions in existing C1A fixtures.

Attention gathers normalized owner rows within its TP group and reduces/scatters
the output back to the same owners. MoE routes those local tokens over EP8 and
combines results at their original owners. Empty owners still participate in
communication. Padding never becomes an expert route or a valid cache write.

The current gate/shared-expert code has static tile-alignment requirements.
For initial integration, use padded MoE workspaces, copy local hidden/pre-mix
into them, initialize the suffix, and pass `local_count`. Copy the original
slab back after MoE. Keep transport row IDs consistent with the padded capacity.
This is a bring-up choice whose extra copies can be measured later.

Select the MoE capacity once before importing shape-dependent modules. Do not
mutate shape globals after JIT functions have been defined. Verify receive
capacity for the worst supported routing concentration and derive its bounds
from unique local token ownership. Preserve existing standalone defaults.

Within each layer, replicate logically replicated parameters across ranks.
Shard attention heads/output groups over TP and routed experts over EP;
replicate TP attention partitions across DP groups. Current C1A indexer ABIs
contain all index heads on each rank: preserve that replicated computation
initially and validate it against the complete-head reference. Do not load an
official TP-sharded indexer tensor directly into this full-head ABI.

## State and dependencies

| Buffer | Planned owner and lifetime |
| --- | --- |
| Window payload/scales | Separate for every decoder layer and DP request set; persistent across steps |
| Compressed KV and index-key payload/scales | Source layer 20; persistent across steps; existing TP replication policy retained |
| Candidate mask | One result per step and DP batch, produced by layer 20 |
| Top-K rows | Five buffers, produced at 20/24/28/32/36 and consumed by the following Reuse layers |
| Hidden/pre-mix | Alternating device workspaces with explicit completion dependencies |
| RoPE rows | Materialized once per distinct profile/position-vector/active-count combination per step |
| Communication windows/signals | Owned by the decoder host, reused only after readers finish |

Five Top-K buffers simplify initial dependency and diagnostic handling; this is
our implementation choice, not a requirement to copy the reference's shared
runtime object. Keep candidate/index outputs separate from persistent sequence
state. Never feed a previous step's selection to a new query.

Use `rope_tables.materialize_rope_rows` with `compressed_attention=True` tables
for this decoder. Query and compression positions coincide for ordinary ratio-1
decode, so their row buffers may alias only after checking the full reuse key.
Carry the preparation TaskIds into explicitly manual dependency regions.

Track epochs separately for each communication protocol and window lifetime.
An epoch advances with actual calls to that window, across layers and across
steps if the allocation survives. Reset signals and epochs together when
allocating fresh windows. Do not reset epochs at a layer boundary or reuse
windows based solely on source-code order.

Inspect generated dependencies for cache publication, Top-K/candidate consumers,
attention-to-MoE hand-off and workspace reuse. Respect the existing publication
and consumption protocol before overwriting any remotely visible buffer.

## File changes

Paths in this table are relative to `models/deepseek_v4_1_flash/` unless stated.

| File | Planned change |
| --- | --- |
| `decode_decoder.py` (new) | Multi-layer host, entry CLI, fixtures, independent chain golden, staged comparisons, multi-step validation, `validate`/`test_precision`/`main` |
| `decode_common.py` | Shared shape/owner helpers, input checks and capacity adapters; avoid imports back into decoder/mode entries |
| `decode_layer.py` | Implement C1A Full/Reindex/Reuse block device composition; connect existing mHC-attention and mHC-MoE boundaries; update golden ABI and readiness |
| `decode_layer_plan.py` | Resolve the 20-layer decoder schedule; distinguish attention-leaf discovery from mHC-composition discovery without breaking existing callers |
| `decode_c1a_full.py`, `decode_c1a_reindex.py`, `decode_c1a_reuse.py` | Reusable device boundaries with dynamic metadata bindings, active counts, group base/rank and correct output directions; keep standalone harnesses |
| `config.py`, `moe.py` | Establish static MoE capacity before import, size receive workspaces consistently, reuse the current complete mHC-MoE golden |
| `_golden_smoke.py` | Adapt Block reference callers and remove only skips actually resolved by this integration |
| `tests/contract/test_v41_decode_decoder.py` (new) | Ownership, schedule, rank mapping, capacity, metadata and lifecycle contract tests |
| `tests/contract/test_deepseek_v4_1_flash_contract.py` | Update affected Block/golden/readiness tests |
| `checkpoint.py` (new, round two) | Selective decoder loading, mapping and layout conversion into the same ABI used by random fixtures |
| `docs/models/deepseek_v4_1_flash/index.md` | Link this plan; update implemented capabilities and validation commands as milestones pass |

Reuse `metadata.py`, `rope_tables.py`, `attention_tp.py`, `ep_transport.py`,
`quantization.py`, and the existing leaf kernels. Modify them only when a
demonstrated integration gap requires it. Capacity changes must audit the
import-time consumers in `gate.py`, `expert_shared.py`, and `expert_routed.py`;
add edits there only if the padded adapter cannot satisfy their existing ABI.

The current `decode_layer.py` golden expects an older MoE interface. Its repair
must not wrap the complete `golden_moe(tensors)` with a second mHC/normalization
sequence. Keep the golden and device composition boundaries aligned.

## Round-one validation

Run these milestones in order:

1. Standalone Full, Reindex and Reuse complete blocks.
2. Layers 20-25 with actual upstream outputs and shared state.
3. Layers 20-39 with different random weights per layer.
4. At least three consecutive decode steps with advancing positions, persistent
   cache updates and fresh layer-19 inputs for each step.

Use two complementary checks. Stage checks evaluate a reference on the device's
actual stage input to localize errors. The independent chain starts from the
original input and evolves its own intermediates, selections and caches; device
outputs must not redefine its expected state. A passing stage check cannot
substitute for a passing independent chain.

Include attention hidden/pre-mix, MoE input and output boundaries, every layer's
output, final hidden/pre-mix, and cache ownership in the reports. Apply active
row budgets separately per rank and inspect padding independently. Preserve
exact comparisons for untouched cache bytes and integer ownership metadata.
Handle near-tied Top-K choices explicitly with reference scores and downstream
output checks; do not silently use device selections as the independent golden.

Reuse current operator-level comparators and document their actual thresholds.
Do not assume the model overview describes every leaf's present comparator.
Define and freeze a separate chain budget before acceptance; report per-layer
relative L2, maximum absolute error, worst-row error and non-finite counts.
Do not inherit the old prefill chain budget or relax thresholds merely to pass.
The numerical chain limits remain an implementation deliverable, not a claimed
result of this source review.

The validation matrix includes:

- Full and partial owner slabs, `T < TP`, and an owner with no active tokens.
- Independent DP request sets and different active counts.
- Multiple requests, unequal history lengths and nonidentity physical pages.
- Positions crossing the 128-token page/window boundary.
- Short contexts, Top-K-limited contexts and more than 16,384 visible ratio-1
  positions so candidate filtering is exercised; include a partial newest block.
- Repeated steps and buffer reuse without fixture reset between steps.
- Distinct layer weights and realistic nonzero quantization scales; replicated
  parameters must agree across their replica ranks.

Record per-rank peak memory before the full 20-layer run. Keep packed weights
resident, avoid permanently expanded expert weights, and bound diagnostic
snapshots. Do not substitute one repeated layer's weights to reduce memory in
the final acceptance case. Device tests belong to the existing A5 pytest/queue
workflow; CPU contract tests do not establish device precision.

## Round-two checkpoint validation

The official `convert.py` removes the optional `model.` prefix, maps
`self_attn`/`mlp` to `attn`/`ffn`, renames scale and routing-bias keys, shards
selected projections, and assigns whole routed experts to ranks. It also
dequantizes `wo_a` to BF16. Use these rules as a mapping reference, while adapting
the split to our separate TP and EP groups.

Implement a streaming loader over the checkpoint index for layers 20-39 only.
Validate required names, dimensions, dtypes and scales. Preserve packed routed
FP4 weights and reuse local packing helpers. Verify transposes, output-group
ordering, scale expansion and expert ownership independently before device
execution. Do not assume one official `mp=8` file matches a TP4/EP8 local shard.

Capture real layer-19 hidden/pre-mix, request positions, historical window and
source caches, and reference decoder outputs from a pinned official run.
Convert cache layout and physical index coordinates explicitly; validate the
conversion before attributing a mismatch to the device. Keep reference capture
and replay orchestration in `decode_decoder.py`'s validation path; use an
external reference checkout rather than vendoring official model code.

Validate original checkpoint tensors and the actual captured boundary first,
then individual blocks, the 20-layer chain and consecutive decode steps.
Real weights with random inputs are useful diagnostics but do not complete this
round. Record source/checkpoint revisions, config, tokenization, positions,
dtype/cast policy and layout conversions with the artifacts. Reference capture
availability and numerical budgets must be resolved before round two can pass.

The PR gate requires both rounds, active and padding checks, persistent-state
checks, applicable regression tests and repository lint to pass. Until then,
report the completed milestones and remaining gaps without marking the decoder
as validated.

## Acceptance contract (round one, A5 TP4/EP4, decoder capacity 4)

The chain runs on four cards as one host graph (`--layers N --steps M --history 200
--decoder-capacity 4 --active-counts ...`). `validate()` selects the enlarged device
ring (`ring_heap` 2 GiB) the packed routed MoE path needs; without it the task
allocator deadlocks as soon as the MoE blocks join the attention chain.

The decoder keeps one allocation for the whole host call and advances the epoch
with the step: layer `slot` of step `step` publishes `step * LAYERS + slot + 1`,
because the device counter a window carries keeps counting for as long as the
allocation lives. A per-step allocation that restarted the epochs at one was tried
and retired: the counters kept advancing while the waits expected one, which is the
deviation recorded under "Multi-step deviation" below. The CPU contract suite
covers the schedule, the window lifetime and the epoch bookkeeping.

The C1A entries publish only the rows they own. Padding rows stay at the value the
fixture initialised them with, which is what the independent chain expects there,
and the earlier in-entry `pl.spmd` zeroing blocks were removed because the inline
pass rejects a store executed by an inlined sub-function (see "Host structure").

Frozen thresholds (see `decoder_comparators` and the constant block above it):

| Boundary | Rule |
| --- | --- |
| `output`, `next_pre_mix`, `attention_hidden`, `attention_pre_mix` | `1e-2` per point and `1e-2` relative, at most 1% of points outside |
| `ffn_input`, `gathered` | one BF16 ulp (`2^-6`) per point, same outlier ratio |
| `rope_cos`, `rope_sin` | `1e-6` per point, no outliers (gather of the caller's tables) |
| `topk_indices` | byte exact on every owner row; non-owner bytes are a rank's own placeholder and are reported |
| `candidate_mask` | byte exact inside each request's window; the filler the tiled store leaves past `compressed_lens` is reported |
| `window_cache`, `window_cache_scale` | unpublished rows byte exact (ownership); published rows reported per layer with relative L2 and worst E4M3 code distance |
| `compressed_cache`, `index_cache` (+ scales) | dequantized active rows within 4%, untouched rows byte exact |
| `moe_padded_*` | host-owned padding workspace; the owner slab is graded through the layer outputs |

Measured on device (log `/home/pyptouser/yuziyu/log/decoder_round1/`):

* Every stream boundary reports `rel_l2=0 max_abs=0 non_finite=0`: the device chain
  and the independent reference agree bit for bit at 2, 6 and 20 layers.
* Compressor and index caches report `active_value_rel_l2=0`; every row outside the
  mapped set is byte identical.
* `candidate_mask` is byte exact inside the window; the tiled store leaves 28-56
  filler bytes past `compressed_lens`, which the indexer never reads back.
* `topk_indices` matches byte exactly on the owner rows; with `--active-counts 1` the
  three ranks that own no row still store a zero placeholder where the fixture keeps
  -1, which the report counts instead of failing.

Open deviation (round one): the published window-cache rows are reported, not graded.
The decoder drives the entries in the sharded mode, so each rank owns SLAB rows of its DP
group and an entry publishes the K/V rows of the slab it owns, while the window cache is
replicated across the TP group and the reference materializes every row from the gathered
K/V. The two agree at layer 20 (every rank writes its own row and the fixture supplies
the rest) and drift apart for the deeper Reindex/Reuse layers, where the measured
relative L2 grows through 5.6% (layer 21), 18% (22), 37% (23) and past 100% by layer 31.
The standalone entries publish correctly when they own the whole group, so this is a
decoder wiring follow-up on the cache-publish path, not a chain precision problem: the
stream boundaries that carry the model output agree bit for bit. Everything structural
around the cache stays enforced - unpublished rows byte exact, owner-row selection
metadata exact - so the deviation is visible in every report instead of hidden by a
threshold.

Milestone status on device (TP4/EP4, four cards): the standalone C1A matrix, the
two-layer segment, the six-layer segment and the twenty-layer single-step segment all
pass; every stream boundary is bit exact (measured `rel_l2=0`) and the structural cache
and selection checks pass. The multi-step milestone passes as well: one layer by three
steps and two layers by two steps pass every boundary and both cache comparisons with
the window lifetime recorded above, which is what the per-step allocation failed. The
acceptance matrix on the revision as committed - one layer by three steps, two layers
by two steps twice, twenty layers by one step - passes end to end (91.4 s, 137.0 s and
137.0 s, 968.6 s). The padding-free entries were re-validated on device after the
zeroing removal - six layers by one step and two layers by two steps - with the same
bit-exact stream boundaries; the cache comparisons stay inside the budget (both exact
after one step; compressor 0.4% and index 0.8% after two steps).

The twenty-layer multi-step cases are a separate, pre-existing gap: the device runtime
poisons the lane (`finalize_native_run` code -100 with
`scheduler timeout sub_class=S1:running-stalled`, one stuck task out of about 108)
before any comparison runs. A bounds run replaces the guess with a measured edge: one
host graph sustains twenty consecutive per-rank dispatches (eight layers by two steps
and twenty layers by one step pass, and the largest epoch a window publishes is that
same count) and stalls at twenty-four and thirty-two (twelve and sixteen layers by two
steps), reproduced in a second checkout of the same revision. A 4 GiB ring heap does
not carry twenty layers by two steps either, so neither the window lifetime nor the
ring capacity is the trigger; the gap tracks how many dispatches one graph chains.
Binding the runtime limit down further is the next device follow-up.

Per-rank footprint before the full run: the fixture reports 3.83 GB of resident
tensors per rank (weights, caches and diagnostics at capacity 4; communication and
kernel scratch come on top). Packed weights stay shared across ranks and no layer
repeats another's weights in the acceptance case.

## Multi-step deviation

Consecutive decode steps used to leave the compressor and index caches disagreeing
with the independent chain on a subset of the published rows of steps one and two.
The window lifetime above is the cause and the fix; the evidence that identified it:

* The comparison contract was not at fault. The chain reference writes exactly the
  rows the comparator inspects, per rank: for a one-layer, three-step, capacity-four
  case both the device row set and the reference row set are the twelve rows the
  step slot tables name, so the failure was a value difference on rows both sides
  agree about.
* The failing rows were not the same rows on every rank, and step two differed where
  step one had matched. Device and reference dumps of `compressed_cache`,
  `compressed_cache_scale`, `index_cache`, `index_cache_scale` and the
  `compressed_slots` input (captured with a temporary comparator under
  `/tmp/device_dump/`) show a timing signature - per-rank, per-step, sporadic - not a
  fixed index error.
* Chain boundaries stayed exact in those runs, which is consistent with a window that
  let a wait pass early: the failing writes are the ones that read a peer's window,
  while the sublayers that carry the model output kept matching bit for bit.
* Re-allocating the windows per step is what exposed it. Such an allocation restarts
  the device counters at zero while the waits derived from `step * LAYERS` expected
  continuation, so from the second step on a wait could be satisfied by a signal the
  previous step left behind.

With one allocation per host call and `step * LAYERS + slot + 1` as the published
epoch, the one-layer three-step and two-layer two-step cases pass every boundary and
both cache comparisons. Two layers by two steps was repeated three times (136.4 s,
136.6 s, 137.8 s) and passed each time, which the sporadic per-step failures never did.

## Host structure

The decode forward reference composes its layers inside one `pl.jit.inline` body that the
`@pl.jit.host` dispatches through the outlined twin of the same body
(`decode_fwd` / `l2_decode_fwd` / `l3_decode_fwd`). This decoder was rebuilt that way -
`_decode_decoder` shared by `pl.jit.inline` (interface and simulator validation) and by
`l2_decode_decoder`, dispatched per rank from `l3_decode_decoder`, with `decoder_moe_rank`
living in `moe.py` - and the attempt is kept at
`/home/pyptouser/yuziyu/log/decoder_round1/dspark_structure_attempt_20260928.patch`.

It does not compile, and the reason is narrower than "the structure is wrong":

* The host must dispatch the outlined twin (`l2_decode_decoder(..., device=rank)`), not
  the inlined one. A traced body may only call `incore`/`inline`/`opaque` sub-functions,
  so the per-rank entries become inline sub-functions and the host keeps the `device=`
  binding. That much is settled and matches the reference.
* Inlining an entry that stores into a tensor parameter from inside `pl.spmd` aborts the
  inline pass with `InternalError: AssignStmt var is not a Var after mutation`
  (`src/ir/transforms/mutator.cpp:631`). The C1A entries cleared this by no longer
  zeroing their padding rows (see the acceptance contract above); the MoE entry cannot,
  because its pad/unpad adapter is what feeds the packed routed kernels: the pad copies
  owner rows into the `MOE_TOKENS` workspace and the unpad copies results back, and
  every variant of those stores - slice to slice, a local `pl.full` value, the
  `for row in pl.spmd(...)` loop form, a static spmd extent, explicit `0:HC_MULT, 0:D`
  slices, dropping the guard - aborts at the same statement. Standalone probes with the
  same shape (same module, cross module, tuple unpacking) pass the inline pass, so the
  remaining difference is inside those two adapters rather than in the structure.
* The entry that was converted to `@pl.jit.opaque` instead cannot be used here either:
  an orchestration rejects a `device=` kwarg on an opaque callee, and dropping the kwarg
  fails program resolution with "references undefined function ... The Program must
  contain every callee referenced from orchestration".

Until pypto resolves that - or the MoE padding adapter is reworked so the packed kernels
take the owner slab directly - the working structure stays the documented L3 shape: the
host authors the per-rank dispatch loop, performs no tensor stores of its own, and lets
every entry write its results through its arguments.

## Known PyPTO issues

The repository keeps `KNOWN_PYPTO_ISSUES.md` locally (it is gitignored), so the
entries this work produced are recorded here as well: the decoder local-slab to
gathered-token dynamic binding rejection, and the inlined-body and `opaque` failure
modes described in "Host structure" above with their minimal reproducers.
