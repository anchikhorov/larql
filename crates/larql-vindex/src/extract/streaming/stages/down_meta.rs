//! Stage 3 — down meta (streaming).

use ndarray::Array2;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::error::VindexError;
use crate::extract::constants::FEATURE_PROJECTION_BATCH;
use crate::extract::debug;
use crate::extract::stage_labels::*;
use crate::extract::streaming::context::StreamingContext;
use crate::extract::streaming::tensor_io::normalize_key;
use crate::format::filenames::*;

impl<'a> StreamingContext<'a> {
    /// Stage 3 — down meta (streaming).
    ///
    /// Auto-resume: skip the entire down-meta phase if the prior run
    /// already wrote `down_meta.bin`. The file is opaque to us here
    /// (we don't reload it), but the loader at the end uses it
    /// directly off disk via `mmap`, and the config-write doesn't
    /// need any per-layer state from this phase — so a clean skip is
    /// safe.
    pub(in crate::extract::streaming) fn write_down_meta(&mut self) -> Result<(), VindexError> {
        let resumed_down = self
            .checkpoint
            .is_complete(crate::extract::checkpoint::ExtractPhase::DownMeta);
        self.callbacks.on_stage(STAGE_DOWN_META);
        if resumed_down {
            eprintln!(
                "  Skipping down_meta phase (reusing existing {})",
                DOWN_META_BIN,
            );
        }
        let mut all_down_meta: Vec<Option<Vec<Option<crate::FeatureMeta>>>> =
            vec![None; self.num_layers];

        let embed = self
            .embed
            .as_ref()
            .expect("embeddings stage must run before down_meta stage");

        // Build whole-word vocab once
        let (_ww_ids, _ww_embed) = crate::extract::build_helpers::build_whole_word_vocab(
            self.tokenizer,
            embed,
            self.vocab_size,
            self.hidden_size,
        );

        if !resumed_down {
            // Capture state for parallel workers.  Copies + shared refs
            // are safe: tensor_source and arch are read-only (mmap),
            // embed is a shared buffer, tokenizer is thread-safe.
            let expert_format = self.expert_format;
            let is_moe = self.is_moe;
            let n_experts = self.n_experts;
            let down_top_k = self.down_top_k;
            let summary_k = self.summary_features_per_expert;
            let num_layers = self.num_layers;
            let prefixes: Vec<String> = self.prefixes.clone();
            let tensor_source = &self.tensor_source;
            let tokenizer = self.tokenizer;
            let arch = &*self.arch;

                        use rayon::prelude::*;
                        let show_timing = debug::extract_down_meta_timing();
                        let in_flight = AtomicUsize::new(0);
                        let max_in_flight = AtomicUsize::new(0);
                        let t0 = Instant::now();
                        if show_timing {
                            eprintln!(
                                "down_meta: {} layers × {} rayon threads starting…",
                                num_layers,
                                ::rayon::current_num_threads(),
                            );
                        }
                        let results: Result<
                            Vec<(usize, Vec<Option<crate::FeatureMeta>>)>,
                            VindexError,
                        > = (0..num_layers)
                            .into_par_iter()
                            .map(|layer| -> Result<_, VindexError> {
                                if show_timing {
                                    let cur = in_flight.fetch_add(1, Ordering::Relaxed) + 1;
                                    max_in_flight.fetch_max(cur, Ordering::Relaxed);
                                }
                    // ── Get down matrices for this layer ──
                    let prefix_refs: Vec<&str> =
                        prefixes.iter().map(|s| s.as_str()).collect();
                    let down_matrices: Vec<Array2<f32>> = if expert_format
                        == larql_models::ExpertFormat::PackedMxfp4
                    {
                        let (shard_mmaps, tensor_index) =
                            match tensor_source.safetensors_view() {
                                Some(v) => v,
                                None => return Ok((layer, Vec::new())),
                            };
                        let blocks_key =
                            arch.packed_down_blocks_key(layer).unwrap_or_default();
                        let scales_key =
                            arch.packed_down_scales_key(layer).unwrap_or_default();
                        if let (Some(bi), Some(si)) = (
                            tensor_index.get(&blocks_key),
                            tensor_index.get(&scales_key),
                        ) {
                            let bst = safetensors::SafeTensors::deserialize(
                                &shard_mmaps[bi.0].mmap,
                            )
                            .map_err(|e| VindexError::Parse(e.to_string()))?;
                            let sst = safetensors::SafeTensors::deserialize(
                                &shard_mmaps[si.0].mmap,
                            )
                            .map_err(|e| VindexError::Parse(e.to_string()))?;
                            let bv = bst
                                .tensor(&bi.1)
                                .map_err(|e| VindexError::Parse(e.to_string()))?;
                            let sv = sst
                                .tensor(&si.1)
                                .map_err(|e| VindexError::Parse(e.to_string()))?;
                            let shape = bv.shape();
                            let n_exp = shape[0];
                            let out_features = shape[1];
                            let groups = shape[2];
                            let in_features = groups * 32;
                            let experts =
                                crate::format::quant::mxfp4::dequantize_all_experts(
                                    bv.data(),
                                    sv.data(),
                                    n_exp,
                                    out_features,
                                    groups,
                                )?;
                            experts
                                .into_iter()
                                .map(|data| {
                                    Array2::from_shape_vec(
                                        (out_features, in_features),
                                        data,
                                    )
                                    .unwrap()
                                })
                                .collect()
                        } else {
                            return Ok((layer, Vec::new()));
                        }
                    } else if expert_format == larql_models::ExpertFormat::PackedBF16
                        && is_moe
                    {
                        let down_key =
                            normalize_key(&arch.ffn_down_key(layer), &prefix_refs);
                        match tensor_source.get_tensor_f32(&down_key)? {
                            Some(t) => vec![t],
                            None => return Ok((layer, Vec::new())),
                        }
                    } else if is_moe && n_experts > 0 {
                        let mut mats = Vec::new();
                        for expert in 0..n_experts {
                            if let Some(key) = arch.expert_ffn_down_key(layer, expert) {
                                let nk = normalize_key(&key, &prefix_refs);
                                if let Some(t) = tensor_source.get_tensor_f32(&nk)? {
                                    mats.push(t);
                                }
                            }
                        }
                        mats
                    } else {
                        let down_key =
                            normalize_key(&arch.ffn_down_key(layer), &prefix_refs);
                        match tensor_source.get_tensor_f32(&down_key)? {
                            Some(t) => vec![t],
                            None => return Ok((layer, Vec::new())),
                        }
                    };

                    if down_matrices.is_empty() {
                        return Ok((layer, Vec::new()));
                    }

                    // ── Compute embed @ w_down and extract top-K ──
                    let mut layer_meta: Vec<Option<crate::FeatureMeta>> = Vec::new();
                    let mut feature_offset = 0usize;
                    for w_down in &down_matrices {
                        let full_features = w_down.shape()[1];
                        let num_features = if summary_k > 0 && full_features > summary_k {
                            summary_k
                        } else {
                            full_features
                        };
                        let batch_size = FEATURE_PROJECTION_BATCH;

                        for batch_start in (0..num_features).step_by(batch_size) {
                            let batch_end =
                                (batch_start + batch_size).min(num_features);

                            let w_chunk = w_down
                                .slice(ndarray::s![.., batch_start..batch_end])
                                .to_owned();
                            let cpu = larql_compute::CpuBackend;
                            use larql_compute::MatMul;
                            let chunk_logits =
                                cpu.matmul(embed.view(), w_chunk.view());

                            for feat in batch_start..batch_end {
                                let col = chunk_logits.column(feat - batch_start);
                                let mut scores: Vec<(usize, f32)> =
                                    col.iter().copied().enumerate().collect();
                                let nan_count =
                                    scores.iter().filter(|(_, s)| s.is_nan()).count();
                                if nan_count > 0 {
                                    eprintln!(
                                        "  warning: {} NaN scores in down_meta layer={} feat={} (of {} total)",
                                        nan_count, layer, feat, scores.len(),
                                    );
                                    scores.retain(|(_, s)| !s.is_nan());
                                }
                                let k = down_top_k.min(scores.len());
                                if k > 0 && k < scores.len() {
                                    scores.select_nth_unstable_by(k, |a, b| {
                                        b.1.total_cmp(&a.1)
                                    });
                                }
                                scores.truncate(k);
                                scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

                                let top_k_entries: Vec<larql_models::TopKEntry> = scores
                                    .into_iter()
                                    .filter_map(|(idx, logit)| {
                                        tokenizer
                                            .decode(&[idx as u32], true)
                                            .ok()
                                            .map(|s| s.trim().to_string())
                                            .filter(|s| !s.is_empty())
                                            .map(|token| larql_models::TopKEntry {
                                                token,
                                                token_id: idx as u32,
                                                logit,
                                            })
                                    })
                                    .collect();

                                let (top_token, top_token_id, c_score) =
                                    if let Some(first) = top_k_entries.first() {
                                        (first.token.clone(), first.token_id, first.logit)
                                    } else {
                                        (String::new(), 0, 0.0)
                                    };

                                let feat_idx = feature_offset + feat;
                                if layer_meta.len() <= feat_idx {
                                    layer_meta.resize(feat_idx + 1, None);
                                }
                                layer_meta[feat_idx] = Some(crate::FeatureMeta {
                                    top_token,
                                    top_token_id,
                                    c_score,
                                    top_k: top_k_entries,
                                });
                            }
                        }
                        feature_offset += num_features;
                    }

                    if show_timing {
                        in_flight.fetch_sub(1, Ordering::Relaxed);
                    }

                    Ok((layer, layer_meta))
                })
                .collect();

                        if show_timing {
                            eprintln!(
                                "down_meta: {:.2?} | layers: {num_layers} | max_in_flight: {} | rayon_threads: {}",
                                t0.elapsed(),
                                max_in_flight.load(Ordering::Relaxed),
                                ::rayon::current_num_threads(),
                            );
                        }

            let results = results?;
            for (layer, meta) in results {
                if !meta.is_empty() {
                    all_down_meta[layer] = Some(meta);
                }
            }
        }

        if !resumed_down {
            // Final write (idempotent — same content as the last
            // per-layer snapshot above when the loop ran to completion).
            crate::format::down_meta::write_binary(
                self.output_dir,
                &all_down_meta,
                self.down_top_k,
            )?;
            self.callbacks.on_stage_done(STAGE_DOWN_META, 0.0);
            self.checkpoint.mark(
                crate::extract::checkpoint::ExtractPhase::DownMeta,
                self.output_dir,
            )?;
        }
        Ok(())
    }
}
