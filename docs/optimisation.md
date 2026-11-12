# YOLO26N Inference Optimisation

GPU: NVIDIA GeForce RTX 5070 (sm_120 / Blackwell)  
Model: YOLO26N, FP32, batch=1, 640×640  
Runtime: vision-rs with teenygrad CUDA graphs  
Date: 2026-05-17

> **Note:** these figures were measured with the fused Conv2d+BN+SiLU kernel. The optimiser
> currently in use doesn't apply that fusion yet, so today's code is slower: on the same GPU,
> `bench` gives about 58 ms/img (57.7 ms GPU-only) as of 2026-11-12.

---

## Baseline benchmark

| Backend | Inference (ms/img) | FPS | mAP50-95 | Notes |
|---|---|---|---|---|
| vision-rs (CUDA graphs) | 9.07 (8.41 GPU-only) | 110 | †0.794 | fused kernels, sm_120 |
| PyTorch eager | 11.03 | 91 | 0.450 | val2017, COCO protocol |
| TorchScript | 5.32 | 188 | 0.445 | val2017, COCO protocol |
| ONNX Runtime | 3.30 | 303 | 0.445 | val2017, COCO protocol |
| TensorRT FP32 | 1.22 | 820 | 0.445 | val2017, COCO protocol |
| MNN | 27.2 | 37 | 0.445 | val2017, COCO protocol |
| NCNN | 44.7 | 22 | 0.464 | val2017, COCO protocol |

† vision-rs mAP is not comparable to the others. See the mAP caveat section below.

vision-rs is **7.4× slower than TensorRT** on GPU kernel time (8.41ms vs 1.22ms).
We are faster than PyTorch eager, which confirms CUDA-graph compilation and
weight folding are working. The gap to TensorRT is entirely in kernel efficiency.

Wall-time breakdown at batch=1 (pinned memory, measured via nsys):

| Phase | Time |
|---|---|
| H→D transfer (4.9 MB input) | ~413µs |
| GPU kernel execution | ~8,410µs |
| D→H transfer (2.8 MB outputs) | ~222µs |
| **Total** | **~9,070µs** |

---

## mAP caveat

The †0.794 figure for vision-rs is not a fair comparison to the 0.445 reported
by the other backends, for two independent reasons:

**1. Evaluated on training data.**  
The vision-rs bench command runs against coco128, a 128-image subset of
*COCO train2017*. The YOLO26N weights were trained on these images. All other
backends report mAP on *val2017* (5,000 images the model has never seen).
Evaluating on training data inflates mAP significantly; even the stricter
`evaluate_map` path in the verify command gives ~0.60 on coco128 (vs 0.445 on val2017).

**2. Lenient evaluator.**  
`evaluate_map_score` (used by bench) allows any-GT matching — a detection can
match any unmatched ground-truth box. The COCO protocol used by ultralytics is
strictly one-to-one with IoU swept from 0.50 to 0.95. The lenient evaluator
also uses float-arithmetic letterbox padding rather than integer division,
which shifts GT box positions slightly and further inflates recall.

To get a comparable number, vision-rs would need to run against COCO val2017
with a pycocotools-compatible evaluator.

---

## Bug fixed: TMA overflow in Conv2dBnSiluTiled (commit 120e74234)

During this session a correctness bug was found and fixed that caused a
significant mAP regression when graph fusion was enabled.

**Symptom**: `graph.optimise()` dropped verify mAP from 0.5998 → 0.4427.

**Root cause**: `Conv2dBnSiluTiledForward::forward_output_row_stride` rounded
`y_col_stride` up to the next multiple of 4, but not to a multiple of
`BLOCK_OW=16`. When `OW % BLOCK_OW != 0` — which is true for OW=40 and OW=20,
the two smaller YOLO26N feature-map widths — the final OW-tile's TMA store
wrote `BLOCK_OW - (OW % BLOCK_OW)` extra columns past the end of the logical
row. The TMA descriptor is shaped `[B*C_OUT, OH*y_col_stride]`; writes that
stay within this 2-D shape are not suppressed, so the overflow silently
corrupted the first few columns of the next `oh` row's activations.

At OW=40 (40 % 16 = 8): 8 columns corrupted per row.  
At OW=20 (20 % 16 = 4): 12 columns corrupted per row.

**Fix** (`forward_output_row_stride`):

```rust
// Before (wrong):
ow.max(self.block_ow as usize).next_multiple_of(4)

// After (correct):
ow.next_multiple_of(self.block_ow as usize)
```

**General rule**: any tiled kernel using TMA 2D descriptors with a padded row
stride must set `y_col_stride = OW.next_multiple_of(BLOCK_OW)`, not just
`next_multiple_of(4)`.

---

## nsys kernel-level breakdown

Profiled with:

```
nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node \
  --kill=none --stats=true ...
```

The `--cuda-graph-trace=node` flag is required to see individual kernel timings
inside CUDA graphs; without it nsys reports only the graph launch.

All figures are per-inference (100-run totals divided by 100).

| Kernel | Calls | Time/infer | % GPU | Avg | Min | Max | σ |
|---|---|---|---|---|---|---|---|
| conv2d_bn_silu_tiled | 40 | 4.16ms | 50.4% | 104µs | 5µs | 556µs | 122µs |
| conv2d_forward | 15 | 1.61ms | 19.5% | 107µs | 1µs | 356µs | 95µs |
| conv2d_bn_silu | 7 | 1.20ms | 14.5% | 171µs | 16µs | 727µs | 231µs |
| flash_attention2 | 4 | 0.58ms | 7.1% | 146µs | 143µs | 147µs | 0.6µs |
| conv2d_bn_silu_gemm | 40 | 0.40ms | 4.8% | 10µs | 4µs | 33µs | 6µs |
| maxpool2d | 3 | 0.073ms | 0.9% | 24µs | 24µs | 25µs | 0.1µs |
| upsample_nearest2d | 2 | 0.070ms | 0.8% | 35µs | 26µs | 44µs | 9µs |
| channel_cat | 54 | 0.069ms | 0.8% | 1.3µs | 0.5µs | 8µs | 1.1µs |
| channel_chunk | 18 | 0.022ms | 0.3% | 1.2µs | 0.7µs | 3µs | 0.6µs |
| nchw_bias_add | 6 | 0.021ms | 0.3% | 3.6µs | 1µs | 8µs | 2.7µs |
| elemwise_add | 20 | 0.016ms | 0.2% | 0.8µs | 0.5µs | 2µs | 0.4µs |
| batch_norm_2d_nchw | 9 | 0.014ms | 0.2% | 1.5µs | 1.4µs | 2µs | 0.1µs |
| psa_pack_qkv | 2 | 0.009ms | 0.1% | 4.3µs | 4.1µs | 4.4µs | 0.04µs |
| psa_extract_v_nchw | 2 | 0.005ms | 0.1% | 2.4µs | 2.3µs | 2.7µs | 0.1µs |
| psa_merge_attn_nchw | 2 | 0.003ms | 0.04% | 1.6µs | 1.5µs | 1.7µs | 0.06µs |

**Convolutions total: 7.37ms (84.8% of GPU time)**

Memory transfer stats (GPU-side, measured by nsys cuda_gpu_mem_time_sum):

| Operation | Count | Avg | Total |
|---|---|---|---|
| Host→Device | 100 | 413µs | 41.3ms |
| Device→Host | 200 | 111µs | 22.2ms |
| cuMemset (scratch) | 8000 | 0.6µs | 4.8ms |

---

## Key observations

**1. Convolution variance is the primary signal.**  
`conv2d_bn_silu_tiled` has σ=122µs against avg=104µs (σ/avg > 1). A 5µs execution
means the GPU is doing almost nothing — the tile parameters are badly mismatched for
small tensors. A 556µs execution (100× more) is a large spatial conv. One fixed tile
configuration cannot be optimal across 40 call sites spanning early stem (large spatial,
few channels) through final FPN (small spatial, many channels).

**2. Three separate conv+BN+SiLU kernel variants.**  
`conv2d_bn_silu_tiled` (40 calls), `conv2d_bn_silu` (7 calls), and
`conv2d_bn_silu_gemm` (40 calls) implement the same fused op via different code paths.
The non-tiled path has even higher variance (σ=231µs) on fewer calls, suggesting it
handles the shapes least suited to the tiled path and also handles them poorly.

**3. No tensor cores.**  
All kernels use NCHW layout. On Blackwell (sm_120), tensor cores (WGMMA) require the
channel dimension to be the contiguous/innermost dimension in memory, which is NHWC.
NCHW convolutions cannot use tensor cores regardless of tile size, capping compute
efficiency at scalar FP32 throughput.

**4. 54 channel_cat + 80 memset launches per inference.**  
Each channel_cat takes 1.3µs — essentially just launch overhead work. The GPU time
is small (0.069ms total) but each is a separate serialised kernel in the schedule.
The 80 memsets zero TMA descriptor scratch buffers that are deterministically reused.

**5. flash_attention2 is well-behaved.**  
σ=0.6µs against avg=146µs — memory-bandwidth-bound and stable. Not a priority
target compared to convolutions, but tunable.

---

## Improvements — ranked by estimated saving

### #1 — Per-shape tile autotuning (conv2d_bn_silu_tiled)
**Estimated saving: ~2.1ms | Effort: Medium**

The σ/avg > 1 ratio on `conv2d_bn_silu_tiled` is definitive evidence that one fixed
tile configuration is wrong for most of the 40 call sites. A per-shape autotuner
benchmarks candidate tile sizes (block dimensions, warp counts, pipeline stages,
register pressure) for each unique `(C_in, C_out, H, W, kH, kW, stride)` combination
at model-compile time and stores the winner. This is exactly what TensorRT's engine
build does.

Conservative estimate: 2× average speedup across the 40 tiled calls eliminates
~2.1ms. Higher potential if Blackwell-specific warp-specialised persistent kernel
patterns (e.g. ping-pong mainloops) are also adopted.

No algorithmic change required — same fused conv+BN+SiLU, different tile parameters
per call site.

### #2 — NHWC layout + tensor-core GEMM for all convolutions
**Estimated saving: ~3.5ms | Effort: High (see detailed section below)**

All conv kernels use NCHW. Switching the entire compute graph to NHWC enables
WGMMA tensor-core instructions and cuDNN-style implicit GEMM algorithms. Convolutions
total 7.37ms; a conservative 2× average speedup saves 3.7ms, reducing GPU time below
5ms. This overlaps with #1 — doing both is not additive, but NHWC with good tiling is
strictly better than either alone.

This is the highest-ceiling change and the endgame for closing the TensorRT gap.
See the NHWC migration section for the full scope.

### #3 — Consolidate conv+BN+SiLU kernel variants
**Estimated saving: ~0.4ms | Effort: Medium**

`conv2d_bn_silu` (non-tiled, 7 calls at 171µs avg) exists as a separate code path
from `conv2d_bn_silu_tiled`. The non-tiled path handles shapes that don't fit the
tiled kernel well — and handles them badly (σ=231µs). Unifying to a single path
with per-shape tile selection from #1 eliminates the routing decision and lets the
autotuner optimise all 47 call sites together. This falls out naturally from #1 rather
than being a standalone task.

### #4 — flash_attention2 block size for sm_120
**Estimated saving: ~0.15ms | Effort: Low**

4 calls, avg=146µs, σ=0.6µs — consistent and memory-bandwidth-bound. sm_120 has
larger L2 and more shared memory per SM than previous architectures. Adjusting Q/K/V
tile sizes and pipeline stages to fill the larger SMEM should yield a 1.3–1.5×
speedup. The low variance makes this a clean, isolated optimisation target.

### #5 — Eliminate 80 cuMemset calls (persistent scratch buffers)
**Estimated saving: ~0.05ms | Effort: Low**

The 80 `cuMemsetD8` calls per inference zero TMA descriptor scratch buffers before
each kernel. CUDA graphs execute with fixed topology and no concurrent kernels on the
same stream, so scratch regions written by kernel N are never read by any concurrent
kernel — only by kernel N+1 or later in a subsequent graph replay, which overwrites
them anyway. Zeroing once at graph-capture time and never again eliminates all 80
operations at zero algorithmic cost.

### #6 — Fuse channel_cat into downstream conv input reads
**Estimated saving: ~0.07ms | Effort: Medium**

54 channel concatenation kernels at 1.3µs each. The concatenated tensor is always
immediately consumed by a convolution. If the conv kernel loads its input as two
separate non-contiguous slices (rather than requiring a contiguous merged buffer), the
cat kernel is eliminated entirely. Saving: 0.069ms of kernel time plus one full
read+write of the intermediate tensor avoided.

### #7 — Fuse nchw_bias_add and elemwise_add into conv epilogue
**Estimated saving: ~0.04ms | Effort: Low**

26 separate element-wise passes (6 bias adds, 20 residual adds) that each do one
global memory read+write after a conv. Writing bias and residual additions into the
conv output epilogue — while results are still in registers before the final tile
store — eliminates the extra global memory round-trip. Minor absolute saving but
removes 26 kernel boundaries.

### Summary

| # | Opportunity | Est. saving | Effort |
|---|---|---|---|
| 1 | Per-shape tile autotuning | ~2.1ms | Medium |
| 2 | NHWC + tensor-core GEMM | ~3.5ms | High |
| 3 | Consolidate conv+BN+SiLU variants | ~0.4ms | Medium (falls out of #1) |
| 4 | flash_attention2 tuning for sm_120 | ~0.15ms | Low |
| 5 | Persistent scratch buffers | ~0.05ms | Low |
| 6 | channel_cat fusion | ~0.07ms | Medium |
| 7 | Bias/residual add into conv epilogue | ~0.04ms | Low |

**Practical sequencing:** #5 and #7 are non-invasive — do first. #1 is the
highest-return self-contained task. #2 is a project; tackle after #1 is landed so
the autotuner baseline is clean. #3 falls out of #1. #4 and #6 are cleanup passes.

---

## NHWC migration — detailed scope

### What NHWC is and why it matters

PyTorch's default tensor format is NCHW: element `(n, c, h, w)` lives at memory
offset `n·C·H·W + c·H·W + h·W + w`. The innermost (fastest-changing) dimension is
`W` — spatial columns.

NHWC places `(n, h, w, c)` at `n·H·W·C + h·W·C + w·C + c`. The innermost
dimension is `C` — channels.

This distinction determines which dimension is contiguous in memory, which in turn
controls:

1. **Coalescing** — a warp loading `C_in` values for one spatial position in NHWC
   gets them in a single burst. In NCHW those values are `H·W` apart.
2. **Tensor cores** — WGMMA instructions on Blackwell require the reduction dimension
   (`C_in`) to be contiguous. NHWC satisfies this; NCHW does not.
3. **Implicit GEMM mapping** — convolution cast as `A × B = C` where
   `A = [N·H_out·W_out, C_in·kH·kW]`, `B = [C_in·kH·kW, C_out]`. The K dimension
   is `C_in·kH·kW`. NHWC keeps `C_in` contiguous in `A`, which is what WGMMA needs.

TensorRT converts the graph to NHWC during engine build. This is a primary reason
for the 7.4× gap, not a secondary one.

### What must change in the kernels

The kernels themselves must be rewritten — you cannot feed NHWC data to an NCHW
kernel and get correct results. The index arithmetic is hardcoded.

NCHW address: `n*C*H*W + c*H*W + h*W + w`  
NHWC address: `n*H*W*C + h*W*C + w*C + c`

Beyond correctness, the rewrite is where the performance comes from:

- **Shared memory tiling** — pack channel values together instead of spatial positions
- **Loop ordering** — channel loop becomes innermost, enabling vectorised loads
- **WGMMA** — once channels are contiguous, emit tensor-core instructions directly
- **Epilogue** — bias and residual adds happen in-register during the output tile store

### Weight layout

Safetensors weights are stored in PyTorch's default `[C_out, C_in, kH, kW]` order
(NCHW). For NHWC implicit GEMM the weight matrix should be `[C_out, kH, kW, C_in]`
so that `C_in` is contiguous along the K dimension. This is a one-time transpose
per weight tensor at load time — no retraining, no accuracy change.

### Op-by-op impact across the graph

Every op that reads a conv output now receives NHWC data. They fall into four
categories:

**No change in behaviour — layout-agnostic**

- `elemwise_add` (residual adds, 20/inference) — purely element-wise. Treat the
  tensor as a flat array. Same code, zero performance difference.

**Slightly better in NHWC**

- `upsample_nearest2d` (2/inference) — copies spatial positions wholesale. In NHWC
  each `(h, w)` position has `C` contiguous values; the copy is better coalesced.

- `nchw_bias_add` — disappears. Bias addition folds into the conv output epilogue
  while results are still in registers; no separate kernel pass at all.

**Need rewriting, roughly neutral**

- `batch_norm_2d` (9/inference, 0.014ms total) — BN reduces over `(N, H, W)` to
  compute per-channel mean/variance. In NCHW these are contiguous per channel; in
  NHWC they are strided (`C` apart). The reduction phase is less cache-friendly. The
  normalization application phase (`(x - mean) * gamma + beta`) is more
  cache-friendly since channels are contiguous. Net effect is roughly a wash. At
  0.2% of GPU time the outcome is immaterial.

- `maxpool2d` (3/inference, 0.073ms total) — loads a `(kH, kW)` spatial
  neighbourhood. In NHWC adjacent spatial positions are `C` apart, so the spatial
  gather reads with larger strides. cuDNN handles this with specialised pooling
  kernels. At 0.9% of GPU time any regression is negligible.

**Need new kernel logic**

- `channel_cat` (54/inference, 0.069ms total) — in NCHW, concatenation along `C`
  is two contiguous block copies. In NHWC it becomes an interleave: for every
  `(n, h, w)` position, write `[A_channels | B_channels]` together. Requires a
  gather-scatter kernel. Still parallelisable and reasonably coalesced for large `C`,
  but different code. The correct long-term fix is to fuse cat into the downstream
  conv's input loading so no separate kernel is needed.

- `channel_chunk` (18/inference, 0.022ms total) — the inverse of cat, same analysis.

- PSA attention ops (`psa_pack_qkv`, `psa_extract_v_nchw`, `psa_merge_attn_nchw`,
  3 kernels, 0.017ms total) — operate on `[B, heads, seq, dim]` tensors that are
  detached from spatial layout. Straightforward rewrites; flash attention itself is
  layout-independent once Q/K/V are extracted.

### The all-or-nothing constraint

Any NCHW↔NHWC boundary inside the graph that is not at the graph edge requires an
explicit transpose kernel. A transpose of a `(1, 256, 80, 80)` feature map is a full
read+write of ~6.5 MB; at 620 GB/s that costs ~10µs. Five layout mismatches inside
the graph burns 50µs — wiping out gains on all the small ops and leaving only the
conv speedup on the table.

The goal is **zero interior transposes**:

1. Graph entry: one `(N, C, H, W) → (N, H, W, C)` kernel as the first node (or do
   it on the host before the H→D transfer)
2. All 192 DAG nodes emit NHWC
3. Graph exit: output tensors (boxes `[N, 300, 4]`, scores `[N, 300, 80]`) have no
   spatial structure, so NHWC vs NCHW is irrelevant

This means every kernel in the DAG must be migrated before the change can be enabled.
A partial migration that leaves some ops in NCHW forces transposes at every boundary
and is likely slower than the NCHW baseline.

### Migration approach

Because the constraint is all-or-nothing, a feature-flag approach works well:

1. Introduce a `MemoryLayout` enum (`Nchw | Nhwc`) in the graph IR
2. Track layout per tensor edge; the compiler inserts explicit `Transpose` nodes at
   any `Nchw↔Nhwc` boundary
3. Migrate kernels one family at a time behind the flag; the transpose nodes absorb
   the cost during transition
4. Once all kernel families support NHWC, enable the layout globally and the
   transpose nodes compile away (they only appear where layouts mismatch)
5. Remove the NCHW kernel paths once NHWC is the default

The autotuner work from improvement #1 should be done in NCHW first so there is a
clean performance baseline; then the NHWC migration has an unambiguous before/after.

### Expected outcome

With NHWC and per-shape tile autotuning together, convolutions (currently 7.37ms)
should approach 1.5–2ms on sm_120 for a YOLO26N-scale network. Combined with the
low-effort wins (#5, #7), total GPU time should reach the 2–3ms range, putting
vision-rs within 2× of TensorRT. Closing the remaining gap requires matching
TensorRT's algorithm selection quality and potentially int8 quantisation.
