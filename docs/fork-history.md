# Fork History — `anchikhorov/larql`

## Context

- **Remote**: `git@github.com:anchikhorov/larql.git`
- **Fork author** (git commits): **Alexander Anhikhorov** (alexander.anchihorov@gmail.com)
- **Head**: 5 commits by the fork owner, all at the tip of `origin/main` (pushed)
- **Divergence point**: first fork commit `114aeba7` sits directly on upstream merge
  `ab7a08c1` (PR #171, `feat/quant-ternary-a8` from `chrishayuk/larql`). Upstream has
  since accumulated 557+ commits, so this branch is pinned to an older base.
- **Scope**: 18 files, **+1078 / −354** lines across 5 commits

---

## Commit Details

### 1. `114aeba7` — GGUF application-level streaming pipeline + Q8₀ NaN fix

**Why:** Models beyond RAM capacity (e.g. 72 GB+) cannot be eagerly loaded.  
**What:**
- mmap-based `GgufTensorSource` / `GgufWeightSource` — memory-efficient, sequential
  tensor reads from a single GGUF file
- `StreamingContext` dispatches between safetensors and GGUF backends in
  `maybe_write_model_weights` (supports `QuantFormat::None` + Q4K)
- Both `gguf-to-vindex` and `safetensors-to-vindex` routes in `convert_cmd` now go
  through `build_vindex_streaming`; the eager path is retained only for BitNet I2_S
- Differential test (`test_streaming_stages_moe`) compares 8 output `.bin` files
  byte-for-byte between eager and streaming GGUF paths

**Bug fix — Q8₀ NaN root cause:**
- `TYPE_Q8_0` and `TYPE_Q5_0` constants were **swapped vs the GGUF wire format**
  (the correct values are Q5₀ = 6, Q8₀ = 8)
- This caused Q8₀ data to be decoded with Q5₀'s 22‑byte block decoder instead of
  the correct 34‑byte decoder, reading `i8` quant bytes as `f16` scales across
  block boundaries
- On a 12.6 GB Gemma4‑coding Q8₀ model this produced **~45 M NaN values** in the
  1 B‑element embedding matrix (~4.5%)
- Added NaN guard (`retain` non‑finite) + `total_cmp` in `down_meta.rs`

| Files touched | Insertions | Deletions |
|---|---|---|
| 7 | 513 | 146 |

---

### 2. `e4230b0e` — PackedBF16 expert-weight streaming extraction (Gemma 4)

**Why:** Gemma 4 hybrid MoE stores expert weights as `PackedBF16`; the streaming
reader returned `None` before.  
**What:**
- `GgufWeightSource::get_packed_bf16` in `tensor_io.rs` — reads raw BF16 tensor bytes
  from the GGUF mmap (was returning `None`)
- `EXPERTS_PACKED_BIN` filename constant in `filenames.rs`
- `PACKED_BF16` writer arm in `write_f32.rs` — loops layers, writes blobs to
  `experts_packed.bin`, emits `kind::PACKED_BF16` manifest entries
- `PACKED_BF16` loader support in `load_f32.rs` — records byte ranges, transfers
  mmap ownership, wires both into `ModelWeights`

| Files touched | Insertions | Deletions |
|---|---|---|
| 4 | 117 | 4 |

---

### 3. `e6805847` — `f16_decode_cache` lock split

**Why:** HNSW warmup functions (`warmup_hnsw_all_layers` / `warmup_hnsw_units`) use
`rayon::par_iter` across layers. Each worker calls `resolve_gate()` to get the `f32`
gate matrix. In the `f16` mmap path, `resolve_gate` held the **single global
`f16_decode_cache` Mutex** across the entire `decode_f16()` call — an
O(num_features × hidden_size) memory decode. Under `par_iter`, N workers serialised
on this one Mutex: one decoded while N‑1 blocked on `futex(FUTEX_WAIT_BITSET)`.  
**What:**
- Split into double‑checked locking: quick check → drop lock → decode → acquire
  lock → insert. Workers that miss the cache decode independently.
- `get_or_insert_with` resolves the rare race where two workers decode the same
  cold layer (one result wins, the other is dropped).

| Files touched | Insertions | Deletions |
|---|---|---|
| 1 | 39 | 11 |

---

### 4. `470de357` — Qwen3.6 architecture gaps + tokenizer fallback

**Why:** Qwen3.6 family wasn't recognised; `convert` lacked a tokenizer fallback.  
**What:**
- Qwen3.6 architecture addition in `qwen.rs` (+44 lines)
- Tokenizer fallback in `convert_cmd.rs` — reconstructs a tokenizer from GGUF
  metadata (WordLevel or BPE depending on presence of merges)
- Safetensors-loading tweaks
- Streaming‑context wiring

| Files touched | Insertions | Deletions |
|---|---|---|
| 4 | 106 | 6 |

---

### 5. `0b4e9d72` — Parallel down-meta matmul + BLAS thread limit + timing

**Why:**
- The down‑meta projection (`embed @ w_down` for every layer) was sequential,
  dominating extraction time on large models.
- When parallelised via rayon, BLAS oversubscription kicked in: OpenBLAS spawns
  its own thread team, so N rayon workers + M BLAS internal threads = N×M
  contention, empirically observed as "starts on all cores, then one core at a time."

**What:**
- Replaced sequential `for layer` loop with `(0..num_layers).into_par_iter().map(…)`
- Set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `GOTO_NUM_THREADS` to `"1"` at process start (only when unset) so rayon
  provides the parallelism cleanly
- Added `LARQL_EXTRACT_DOWN_META_TIMING=1` debug flag with `OnceLock` resolver
  — prints per-layer timing, in‑flight concurrency counter, and summary stats
- Fixed `convert_cmd.rs` compile error: widened return type from
  `Box<dyn Error>` to `Box<dyn Error + Send + Sync>` to match `parse()` signature

| Files touched | Insertions | Deletions |
|---|---|---|
| 6 | 318 | 202 |

---

## Narrative

The fork is a **feature branch that adds GGUF streaming extraction** (bounded memory
for huge transformer models) and **broadens model-family coverage** (Gemma 4 hybrid
MoE with PackedBF16 experts, Qwen3.6), while **hardening correctness & concurrency**
(NaN wire-format bug, parallel down-meta, lock contention, BLAS oversubscription).

The work follows a pragmatic "make it work → support real hardware → make it fast"
arc:
1. Core streaming pipeline + correctness (NaN fix)
2. MoE expert format coverage (PackedBF16)
3. Concurrency fix (lock split)
4. Model-family gap (Qwen3.6)
5. Performance with the BLAS‑rayon fix

---

## Open / watch items

- `qwen3.6-support-checklist.md` (staged but not examined here) is the audit of
  remaining Qwen3.6 gaps: prefix remap, shared expert keys, SSM, MTP.
- The fork is pinned to an old upstream base (`ab7a08c1`) while upstream
  `chrishayuk/larql` has advanced by 557+ commits. A future rebase or merge
  will be non-trivial.
