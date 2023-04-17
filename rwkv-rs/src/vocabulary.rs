use crate::InferenceError;
use std::path::Path;
use std::{collections::HashMap, str::FromStr};

use tokenizers::Tokenizer;

/// The identifier of a token in a vocabulary.
pub type TokenId = i32;
pub(crate) type Token = Vec<u8>;
pub(crate) type TokenScore = f32;

/// The vocabulary used by a model.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Maps every integer (index) token id to its corresponding token
    tokenizer: Tokenizer,
}
impl Vocabulary {
    pub(crate) fn new(path: impl AsRef<Path>) -> Vocabulary {
        let tokenizer = Tokenizer::from_file(path).unwrap();
        Vocabulary { tokenizer }
    }

    pub fn encode(&self, input: &str) -> Vec<i32> {
        let input_tokens_encoding = self.tokenizer.encode(input, false).unwrap();
        let input_tokens = input_tokens_encoding.get_ids();

        let mut ids = Vec::with_capacity(input.len());
        for t in input_tokens {
            ids.push(*t as i32)
        }
        ids
    }

    pub fn decode(&self, input: i32) -> String {
        self.tokenizer
            .decode(Vec::from([input as u32]), false)
            .unwrap()
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
/// A list of tokens to bias during the process of inferencing.
///
/// When a biased token is encountered, the bias will be used
/// instead of the inferred logit during the sampling process.
///
/// This can be used to disable the generation of responses
/// with specific tokens by setting their corresponding bias
/// to -1.0.
pub struct TokenBias(HashMap<TokenId, f32>);

impl TokenBias {
    /// Create a [TokenBias] from an existing `Vec`.
    pub fn new(mut v: HashMap<TokenId, f32>) -> Self {
        Self(v)
    }

    /// Retrieves the bias for a given token, if available.
    pub fn get(&self, tid: TokenId) -> Option<f32> {
        self.0.get(&tid).map(|x| *x)
    }
}

impl FromStr for TokenBias {
    type Err = String;

    /// A comma separated list of token biases. The list should be in the format
    /// "TID=BIAS,TID=BIAS" where TID is an integer token ID and BIAS is a
    /// floating point number.
    /// For example, "1=-1.0,2=-1.0" sets the bias for token IDs 1
    /// (start of document) and 2 (end of document) to -1.0 which effectively
    /// disables the model from generating responses containing those token IDs.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let x = s
            .split(',')
            .map(|kv| {
                let (k, v) = kv
                    .trim()
                    .split_once('=')
                    .ok_or_else(|| "Missing '=' in bias item".to_owned())?;
                let tid: TokenId = k
                    .trim()
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?;
                let bias: f32 = v
                    .trim()
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?;
                Result::<_, String>::Ok((tid, bias))
            })
            .collect::<Result<_, _>>()?;
        Ok(TokenBias::new(x))
    }
}

impl std::fmt::Display for TokenBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
