#![deny(missing_docs)]
//! rwkv-rs is a Rust port of the rwkv.cpp project. This allows running inference for RWKV model on a CPU with good performance using full precision, f16 or 4-bit quantized versions of the model.
use crate::vocabulary::{TokenBias, TokenId};

use thiserror::Error;

mod loader;
mod model;
mod rwkv_session;
mod util;
mod vocabulary;

/// The end of text token.
pub const EOT_TOKEN_ID: TokenId = 2; // Hardcoded (for now?)

#[derive(Clone, Debug, PartialEq)]
/// The parameters that drive text generation.
pub struct InferenceParameters {
    /// The number of threads to use.
    pub n_threads: usize,
    /// [InferenceSession::feed_prompt] processes the prompt in batches of tokens.
    /// This controls how large an individual batch is.
    pub n_batch: usize,
    ///  Top-K: The top K words by score are kept during sampling.
    pub top_k: usize,
    /// Top-p: The cumulative probability after which no more words are kept for sampling.
    pub top_p: f32,
    /// The penalty for repeating tokens. Higher values make the generation less
    /// likely to get into a loop, but may harm results when repetitive outputs
    /// are desired.
    pub repeat_penalty: f32,
    /// Temperature used for sampling.
    pub temperature: f32,
    /// A list of tokens to bias against in the process of generation.
    pub bias_tokens: TokenBias,
    /// Whether or not previous tokens should be played back in [InferenceSession::inference_with_prompt].
    pub play_back_previous_tokens: bool,
}
impl Default for InferenceParameters {
    fn default() -> Self {
        Self {
            n_threads: 8,
            n_batch: 8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.30,
            temperature: 0.80,
            bias_tokens: TokenBias::default(),
            play_back_previous_tokens: false,
        }
    }
}

#[derive(Error, Debug)]
/// Errors encountered during the inferencep rocess.
pub enum InferenceError {
    #[error("an invalid token was encountered during tokenization")]
    /// During tokenization, one of the produced tokens was invalid / zero.
    TokenizationFailed,
    #[error("the context window is full")]
    /// The context window for the model is full.
    ContextFull,
    #[error("reached end of text")]
    /// The model has produced an end of text token, signalling that it thinks that the text should end here.
    ///
    /// Note that this error *can* be ignored and inference can continue, but the results are not guaranteed to be sensical.
    EndOfText,
    #[error("the user-specified callback returned an error")]
    /// The user-specified callback returned an error.
    UserCallback(Box<dyn std::error::Error>),
}

/// Used in a call to `evaluate` to request information from the transformer.
#[derive(Default, Debug, Clone)]
pub struct EvaluateOutputRequest {
    /// Returns all the logits for the provided batch of tokens.
    /// Output shape is `n_batch * n_vocab`.
    pub all_logits: Option<Vec<f32>>,
    /// Returns the embeddings for the provided batch of tokens
    /// Output shape is `n_batch * n_embd`.
    pub embeddings: Option<Vec<f32>>,
}
