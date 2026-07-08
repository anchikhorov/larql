# Qwen3.6 Support Verification Checklist for LARQL

Context: attempting to convert Qwen3.6-27B / Qwen3.6-35B-A3B (Unsloth GGUF and HF safetensors builds) to a `.vindex` file using `larql convert gguf-to-vindex` / `safetensors-to-vindex`. Conversion currently fails or is architecturally unsupported. Please audit the codebase against the points below and report, per point, whether it is (a) implemented, (b) partially implemented, or (c) not implemented.

## 1. Hybrid attention architecture (Gated DeltaNet / SSM layers)

Qwen3.6 (inherited from Qwen3.5) uses a **hybrid layer stack**, not uniform softmax attention:

- Roughly 1 in every ~11 layers is standard attention: tensors named `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`, `attn_k_norm` (GQA + QK-Norm).
- The remaining layers (~90%) are **Gated DeltaNet / linear-attention (state-space) layers**, with tensors named: `ssm_a`, `ssm_alpha`, `ssm_conv1d`, `ssm_beta`, `ssm_dt.bias`, `ssm_norm`, `ssm_out` (verified directly against GGUF tensor names of `unsloth/Qwen3.6-27B-MTP-GGUF`).

Checks needed:
- [ ] Does `ModelArchitecture` (or equivalent) detect per-layer type (attention vs SSM) rather than assuming uniform attention across all layers?
- [ ] Is there any parser/handler for `ssm_*` tensors at all, or are they silently skipped/ignored during extraction?
- [ ] If skipped, does that silently degrade `DESCRIBE`/`WALK` output quality (analogous to the documented MXFP4 quantization issue), or does it hard-fail?
- [ ] Is the residual-stream additivity assumption (used for boundary/window compression) still valid when a chunk of layers is SSM-based rather than attention-based, or does the SSM state (not just attention) need separate handling for correctness?

## 2. Multi-Token Prediction (MTP) module

Some Qwen3.6 GGUF builds include an `mtp.*` tensor group (e.g. `mtp.pre_fc_norm_embedding.weight`).

Checks needed:
- [ ] Is `mtp.*` recognized and either correctly ignored (safe to drop for vindex purposes) or intentionally used?
- [ ] Confirm it does not cause a hard load failure when present.

## 3. Vision-language (VLM) tensor structure

The HF safetensors release of Qwen3.6-35B-A3B is natively multimodal:

- Text backbone is nested under `model.language_model.*` (not flat `model.*` as in Qwen2.5/Qwen3 text-only checkpoints).
- Vision tower present under `model.visual.*` (patch_embed, pos_embed, etc.).

Observed failure: `Error: missing tensor: embed_tokens.weight` — loader expects `embed_tokens.weight` and does not resolve `model.language_model.embed_tokens.weight`.

Checks needed:
- [ ] Does the safetensors loader support a `language_model.` prefix remap for text backbone tensors?
- [ ] Does it explicitly skip `visual.*` tensors rather than erroring on encountering them?
- [ ] Is there a code path confirming `language_model` string never appears anywhere in `larql-vindex/src/` (grep returned empty on the Divinci-AI fork as of this writing) — confirm current status.
- [ ] Some GGUF builds (e.g. Coder-focused finetunes) may omit vision weights or carry stale `image-text-to-text` HF tags without actual vision tensors — confirm the vindex extractor doesn't assume presence/absence based on repo tags alone, but inspects actual tensors.

## 4. MoE routing (A3B variant specifically)

Qwen3.6-35B-A3B uses **256 experts (8 routed + 1 shared)**, distinct from Mixtral-style MoE (no shared expert) and from Qwen2.5-MoE (fine-grained segmentation, no shared expert either, per Qwen3 technical report).

Checks needed:
- [ ] Does the `moe_svd.rs` router-weighted SVD aggregation path explicitly handle a shared-expert-plus-routed-experts topology (vs. routed-only)?
- [ ] Is expert count (256) and top-k routing (8) read from config rather than hardcoded to a different MoE profile (e.g. Mixtral's 8 total experts)?

## 5. Tokenizer / config handling

- [ ] Confirm `tokenizer.json` resolution logic for GGUF inputs — currently requires manual placement alongside the `.gguf` file even when the source repo bundles it separately (e.g. base-model tokenizer reuse for finetunes/merges).

## Suggested verification method

For each point above, the fastest verification is not to read documentation but to:
1. Dump all unique tensor names from a real Qwen3.6 checkpoint (safetensors index or GGUF) and diff against what `ModelArchitecture`/tensor-name matching code in `larql-vindex` and `larql-core` actually looks for.
2. Attempt extraction at `--level browse` only (cheapest) and check whether failure occurs at tensor-resolution time (config/naming issue, easy fix) vs. requiring new parsing logic for SSM/MTP tensor types (architectural gap, not a quick fix).

## Summary of what's confirmed NOT supported today (Divinci-AI fork, checked directly)

- `language_model.` prefix: not referenced anywhere in `larql-vindex/src/` (empty grep result).
- SSM/Gated DeltaNet tensors (`ssm_*`): no evidence of dedicated parsing found; likely treated as unknown/unsupported tensor type.
- These two gaps alone are sufficient to block both the 27B dense and 35B-A3B MoE Qwen3.6 variants regardless of the MoE-specific SVD code already present in the fork.
