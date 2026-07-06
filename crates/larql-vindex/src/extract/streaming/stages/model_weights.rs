//! Stage 6 — model weights (if extract level requires them).

use crate::config::types::QuantFormat;
use crate::error::VindexError;
use crate::extract::streaming::context::StreamingContext;
use crate::extract::streaming::tensor_io::GgufWeightSource;

impl<'a> StreamingContext<'a> {
    /// Stage 6 — model weights (if extract level requires them).
    ///
    /// With quant=q4k we always materialise weights regardless of the
    /// declared level — the Q4_K writer emits all of attn, FFN, norms,
    /// lm_head in one pass and makes `--level browse --quant q4k`
    /// incoherent, so q4k implicitly promotes to "all".
    ///
    /// Both safetensors and GGUF inputs are supported. Safetensors uses
    /// `StreamingWeights` (safetensors-crate view); GGUF uses
    /// `GgufWeightSource` (per-tensor streaming through `ggml::dequantize`).
    pub(in crate::extract::streaming) fn maybe_write_model_weights(
        &mut self,
    ) -> Result<(), VindexError> {
        let needs_weights = self.extract_level.writes_attn() || self.quant != QuantFormat::None;
        if !needs_weights {
            return Ok(());
        }

        // Thread the extract level into the write options so the
        // writer can skip attn/FFN/lm_head sections per tier.
        let mut level_opts = self.weight_opts;
        level_opts.level = self.extract_level;

        // Dispatch between safetensors-backed and GGUF-backed weight sources.
        if let Some(gguf_src) = self.tensor_source.gguf_source() {
            let gguf_source = GgufWeightSource {
                src: gguf_src,
                arch: &*self.arch,
                num_layers: self.num_layers,
            };
            match self.quant {
                QuantFormat::None => {
                    crate::format::weights::write_model_weights_with_opts(
                        &gguf_source,
                        self.output_dir,
                        self.callbacks,
                        level_opts,
                    )?;
                }
                QuantFormat::Q4K => {
                    crate::format::weights::write_model_weights_kquant_with_opts(
                        &gguf_source,
                        self.output_dir,
                        self.callbacks,
                        self.q4k_opts,
                    )?;
                }
            }
        } else {
            let (shard_mmaps, tensor_index) = (
                self.tensor_source.safetensors_mmap_refs(),
                self.tensor_source.safetensors_index(),
            );
            let (shard_mmaps, tensor_index) = match (shard_mmaps, tensor_index) {
                (Some(m), Some(i)) => (m, i),
                _ => {
                    return Err(VindexError::Parse(
                        "neither safetensors nor GGUF tensors available for weight writing"
                            .to_string(),
                    ));
                }
            };
            let streaming_source = crate::format::weights::StreamingWeights {
                shard_mmaps: &shard_mmaps,
                tensor_index,
                arch: &*self.arch,
                num_layers: self.num_layers,
            };
            match self.quant {
                QuantFormat::None => {
                    crate::format::weights::write_model_weights_with_opts(
                        &streaming_source,
                        self.output_dir,
                        self.callbacks,
                        level_opts,
                    )?;
                }
                QuantFormat::Q4K => {
                    crate::format::weights::write_model_weights_kquant_with_opts(
                        &streaming_source,
                        self.output_dir,
                        self.callbacks,
                        self.q4k_opts,
                    )?;
                }
            }
        }
        Ok(())
    }
}
