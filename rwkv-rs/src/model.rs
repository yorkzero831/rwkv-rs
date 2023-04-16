use crate::loader;
use crate::loader::{LoadError, LoadProgress};
use crate::rwkv_session::{InferenceSession, InferenceSessionParameters};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::ptr::null;

pub struct Model {
    pub(crate) hparams: Hyperparameters,

    pub(crate) emb: ggml_rwkv::Tensor,

    pub(crate) ln0_weight: ggml_rwkv::Tensor,
    pub(crate) ln0_bias: ggml_rwkv::Tensor,

    pub(crate) layers: Vec<Layer>,

    tensors: HashMap<String, ggml_rwkv::Tensor>,

    pub(crate) ln_out_weight: ggml_rwkv::Tensor,
    pub(crate) ln_out_bias: ggml_rwkv::Tensor,

    pub(crate) head: ggml_rwkv::Tensor,

    // Must be kept alive for the model
    _context: ggml_rwkv::Context,
}

impl Model {
    pub(crate) fn new(
        context: ggml_rwkv::Context,
        hparams: Hyperparameters,
        wtype: ggml_rwkv::Type,
    ) -> Model {
        let n_embd = hparams.n_embed;
        let n_layer = hparams.n_layer;
        let n_vocab = hparams.n_vocab;

        let mut tensors = HashMap::new();

        let emb_weight = context.new_tensor_2d(wtype, n_embd, n_vocab);
        let ln0_weight = context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd);
        let ln0_bias = context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd);
        let ln_out_weight = context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd);
        let ln_out_bias = context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd);
        let head_weight = context.new_tensor_2d(wtype, n_embd, n_vocab);

        tensors.insert("emb.weight".to_owned(), emb_weight.share());
        tensors.insert("blocks.0.ln0.weight".to_owned(), ln0_weight.share());
        tensors.insert("blocks.0.ln0.bias".to_owned(), ln0_bias.share());
        tensors.insert("ln_out.weight".to_owned(), ln_out_weight.share());
        tensors.insert("ln_out.bias".to_owned(), ln_out_bias.share());
        tensors.insert("head.weight".to_owned(), head_weight.share());

        let mut layers = Vec::new();

        for i in 0..n_layer {
            let layer = Layer {
                ln1_weight: context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd),
                ln1_bias: context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd),

                att_time_mix_k: context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd),
                att_time_mix_v: context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd),
                att_time_mix_r: context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd),
                att_time_first: context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd),
                att_time_decay: context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd),

                att_key: context.new_tensor_2d(wtype, n_embd, n_embd),
                att_value: context.new_tensor_2d(wtype, n_embd, n_embd),
                att_receptance: context.new_tensor_2d(wtype, n_embd, n_embd),
                att_output: context.new_tensor_2d(wtype, n_embd, n_embd),

                ln2_weight: context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd),
                ln2_bias: context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd),

                ffn_time_mix_k: context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd),
                ffn_time_mix_r: context.new_tensor_1d(ggml_rwkv::Type::F32, n_embd),
                ffn_key: context.new_tensor_2d(wtype, n_embd, 4 * n_embd),
                ffn_value: context.new_tensor_2d(wtype, 4 * n_embd, n_embd),
                ffn_receptance: context.new_tensor_2d(wtype, n_embd, n_embd),
            };

            tensors.insert(format!("blocks.{i}.ln1.weight"), layer.ln1_weight.share());
            tensors.insert(format!("blocks.{i}.ln1.bias"), layer.ln1_bias.share());

            tensors.insert(
                format!("blocks.{i}.att.time_mix_k"),
                layer.att_time_mix_k.share(),
            );
            tensors.insert(
                format!("blocks.{i}.att.time_mix_v"),
                layer.att_time_mix_v.share(),
            );
            tensors.insert(
                format!("blocks.{i}.att.time_mix_r"),
                layer.att_time_mix_r.share(),
            );
            tensors.insert(
                format!("blocks.{i}.att.time_first"),
                layer.att_time_first.share(),
            );
            tensors.insert(
                format!("blocks.{i}.att.time_decay"),
                layer.att_time_decay.share(),
            );

            tensors.insert(format!("blocks.{i}.att.key.weight"), layer.att_key.share());
            tensors.insert(
                format!("blocks.{i}.att.value.weight"),
                layer.att_value.share(),
            );
            tensors.insert(
                format!("blocks.{i}.att.receptance.weight"),
                layer.att_receptance.share(),
            );
            tensors.insert(
                format!("blocks.{i}.att.output.weight"),
                layer.att_output.share(),
            );

            tensors.insert(format!("blocks.{i}.ln2.weight"), layer.ln2_weight.share());
            tensors.insert(format!("blocks.{i}.ln2.bias"), layer.ln2_bias.share());

            tensors.insert(
                format!("blocks.{i}.ffn.time_mix_k"),
                layer.ffn_time_mix_k.share(),
            );
            tensors.insert(
                format!("blocks.{i}.ffn.time_mix_r"),
                layer.ffn_time_mix_r.share(),
            );
            tensors.insert(format!("blocks.{i}.ffn.key.weight"), layer.ffn_key.share());
            tensors.insert(
                format!("blocks.{i}.ffn.value.weight"),
                layer.ffn_value.share(),
            );
            tensors.insert(
                format!("blocks.{i}.ffn.receptance.weight"),
                layer.ffn_receptance.share(),
            );

            layers.push(layer);
        }

        Model {
            hparams,
            emb: emb_weight,
            ln0_weight,
            ln0_bias,
            layers,
            tensors,
            ln_out_weight,
            ln_out_bias,
            head: head_weight,
            _context: context,
        }
    }

    pub(crate) fn tensors(&self) -> &HashMap<String, ggml_rwkv::Tensor> {
        &self.tensors
    }

    /// Load the model from `path`
    ///
    /// The status of the loading process will be reported through `load_progress_callback`.
    pub fn load(
        path: impl AsRef<Path>,
        load_progress_callback: impl FnMut(LoadProgress),
    ) -> Result<Model, LoadError> {
        loader::load(path, load_progress_callback)
    }

    /// Starts a new `InferenceSession` for this model.
    pub fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession {
        InferenceSession::new(params, self)
    }
}

/// The hyperparameters of the model.
#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct Hyperparameters {
    /// n_vocab
    pub n_vocab: usize,
    /// n_ctx
    pub n_layer: usize,
    /// n_embd
    pub n_embed: usize,
    /// f16_
    pub f16_: u32,
}

pub struct Layer {
    pub(crate) ln1_weight: ggml_rwkv::Tensor,
    pub(crate) ln1_bias: ggml_rwkv::Tensor,

    // RWKV, also called "attention" by the author.
    pub(crate) att_time_mix_k: ggml_rwkv::Tensor,
    pub(crate) att_time_mix_v: ggml_rwkv::Tensor,
    pub(crate) att_time_mix_r: ggml_rwkv::Tensor,
    pub(crate) att_time_first: ggml_rwkv::Tensor,
    pub(crate) att_time_decay: ggml_rwkv::Tensor,
    pub(crate) att_key: ggml_rwkv::Tensor,
    pub(crate) att_value: ggml_rwkv::Tensor,
    pub(crate) att_receptance: ggml_rwkv::Tensor,
    pub(crate) att_output: ggml_rwkv::Tensor,

    pub(crate) ln2_weight: ggml_rwkv::Tensor,
    pub(crate) ln2_bias: ggml_rwkv::Tensor,

    // FFN.
    pub(crate) ffn_time_mix_k: ggml_rwkv::Tensor,
    pub(crate) ffn_time_mix_r: ggml_rwkv::Tensor,
    pub(crate) ffn_key: ggml_rwkv::Tensor,
    pub(crate) ffn_value: ggml_rwkv::Tensor,
    pub(crate) ffn_receptance: ggml_rwkv::Tensor,
}
