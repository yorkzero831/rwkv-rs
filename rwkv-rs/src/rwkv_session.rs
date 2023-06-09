use crate::model::Model;
use crate::util::mulf;
use crate::vocabulary::TokenId;
use crate::{EvaluateOutputRequest, InferenceParameters};
use ggml_rwkv::{ComputationGraph, Tensor, Type};
use partial_sort::PartialSort;
use rand::{distributions::WeightedIndex, prelude::Distribution};
use std::cmp::max;
use std::ptr::copy_nonoverlapping;

const SCRATCH_SIZE: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
/// Parameters for an inference session.
pub struct InferenceSessionParameters {
    /// The number of tokens to consider for the repetition penalty.
    pub repetition_penalty_last_n: usize,
    pub n_thread: usize,
}

impl Default for InferenceSessionParameters {
    fn default() -> Self {
        Self {
            repetition_penalty_last_n: 512,
            n_thread: max(1, num_cpus::get()),
        }
    }
}

pub struct InferenceSession {
    pub(crate) _session_ctx: ggml_rwkv::Context,

    // Original size of the memory used to create this context.
    pub(crate) memory_size: usize,

    // Parameters for the session.
    pub(crate) params: InferenceSessionParameters,

    pub(crate) token_index: Tensor,
    pub(crate) state: Tensor,
    pub(crate) state_parts: Vec<Tensor>,
    pub(crate) logits: Tensor,

    pub(crate) graph: ggml_rwkv::ComputationGraph,

    /// How much memory is required per token for the temporary context used
    /// during inference.
    pub(crate) mem_per_token: usize,

    /// All tokens generated by this inference session
    pub(crate) tokens: Vec<TokenId>,

    /// The logits that were last predicted by the network. Zeroed out otherwise.
    pub(crate) last_logits: Vec<f32>,
}

const FP32_SIZE: usize = 4;

impl InferenceSession {
    pub(crate) fn new(params: InferenceSessionParameters, model: &Model) -> InferenceSession {
        let n_embd = model.hparams.n_embed;
        let n_layer = model.hparams.n_layer;
        let n_vocab = model.hparams.n_vocab;

        let ctx_size = {
            // Use 64-bit math to prevent overflow.
            let mut ctx_size: usize = 0;
            // Intermediary vectors for calculation; there are around 100 calls to ggml
            ctx_size += mulf!(100, n_embd, ggml_rwkv::type_sizef(ggml_rwkv::Type::F32));
            // State, in and out
            ctx_size += mulf!(
                (2 * 5),
                n_layer,
                n_embd,
                ggml_rwkv::type_sizef(ggml_rwkv::Type::F32)
            );
            // Logits
            ctx_size += mulf!(n_vocab, ggml_rwkv::type_sizef(ggml_rwkv::Type::F32));
            // +256 MB just for any overhead
            // TODO This is too much for smaller models; need a more proper and robust way of measuring required memory
            ctx_size += (256) * 1024 * 1024;

            ctx_size
        };
        let session_ctx = ggml_rwkv::Context::init(ctx_size);
        let state = session_ctx.new_tensor_1d(Type::F32, n_layer * 5 * n_embd);

        // x = self.w.emb.weight[token]
        let token_index = session_ctx.new_tensor_1d(Type::I32, 1);
        let mut x = session_ctx.op_get_rows(&model.emb, &token_index);

        // x = self.layer_norm(x, self.w.blocks[0].ln0)
        x = session_ctx.op_rwkv_layer_norm(&x, &model.ln0_weight, &model.ln0_bias);

        // We collect parts of new state here. Each part is (n_embed) vector.

        let mut v0: Tensor;
        let mut v1: Tensor;
        let mut v2: Tensor;
        let mut v3: Tensor;
        let mut v4: Tensor;

        let mut state_parts = Vec::with_capacity(n_layer * 5);
        for (i, layer) in model.layers.iter().enumerate() {
            // RWKV/time mixing
            {
                let x0 = session_ctx.op_rwkv_layer_norm(&x, &layer.ln1_weight, &layer.ln1_bias);
                let x_prev =
                    session_ctx.op_view_1d(&state, n_embd, (5 * i + 1) * n_embd * FP32_SIZE);
                // xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
                // xv = x * time_mix_v + state[5 * i + 1] * (1 - time_mix_v)
                // xr = x * time_mix_r + state[5 * i + 1] * (1 - time_mix_r)
                let xk = session_ctx.op_add(
                    &session_ctx.op_mul(&x0, &layer.att_time_mix_k),
                    &session_ctx.op_mul(&x_prev, &session_ctx.op_1_minus_x(&layer.att_time_mix_k)),
                );

                let xv = session_ctx.op_add(
                    &session_ctx.op_mul(&x0, &layer.att_time_mix_v),
                    &session_ctx.op_mul(&x_prev, &session_ctx.op_1_minus_x(&layer.att_time_mix_v)),
                );

                let xr = session_ctx.op_add(
                    &session_ctx.op_mul(&x0, &layer.att_time_mix_r),
                    &session_ctx.op_mul(&x_prev, &session_ctx.op_1_minus_x(&layer.att_time_mix_r)),
                );
                // state[5 * i + 1] = x
                // state_parts[5 * i + 1] = x0;
                v1 = x0;

                // r = torch.sigmoid(rw @ xr)
                let r = session_ctx.op_sigmoid(&session_ctx.op_mul_mat(&layer.att_receptance, &xr));
                // k = kw @ xk
                let k = session_ctx.op_mul_mat(&layer.att_key, &xk);
                // v = vw @ xv
                let v = session_ctx.op_mul_mat(&layer.att_value, &xv);

                // aa = state[5 * i + 2]
                // bb = state[5 * i + 3]
                // pp = state[5 * i + 4]
                let aa = session_ctx.op_view_1d(&state, n_embd, (5 * i + 2) * n_embd * FP32_SIZE);
                let bb = session_ctx.op_view_1d(&state, n_embd, (5 * i + 3) * n_embd * FP32_SIZE);
                let pp = session_ctx.op_view_1d(&state, n_embd, (5 * i + 4) * n_embd * FP32_SIZE);

                // ww = time_first + k
                let mut ww = session_ctx.op_add(&layer.att_time_first, &k);
                // qq = torch.maximum(pp, ww)
                let mut qq = session_ctx.op_max(&pp, &ww);
                // e1 = torch.exp(pp - qq)
                let mut e1 = session_ctx.op_exp(&session_ctx.op_sub(&pp, &qq));
                // e2 = torch.exp(ww - qq)
                let mut e2 = session_ctx.op_exp(&session_ctx.op_sub(&ww, &qq));
                // a = e1 * aa + e2 * v
                let a =
                    session_ctx.op_add(&session_ctx.op_mul(&e1, &aa), &session_ctx.op_mul(&e2, &v));
                // b = e1 * bb + e2
                let b = session_ctx.op_add(&session_ctx.op_mul(&e1, &bb), &e2);
                // wkv = a / b
                let wkv = session_ctx.op_div(&a, &b);
                ww = session_ctx.op_add(&pp, &layer.att_time_decay);
                // qq = torch.maximum(ww, k)
                qq = session_ctx.op_max(&ww, &k);
                // e1 = torch.exp(ww - qq)
                e1 = session_ctx.op_exp(&session_ctx.op_sub(&ww, &qq));
                // e2 = torch.exp(k - qq)
                e2 = session_ctx.op_exp(&session_ctx.op_sub(&k, &qq));
                // state[5 * i + 2] = e1 * aa + e2 * v
                // state_parts[5 * i + 2] = session_ctx.op_add(
                //     &session_ctx.op_mul(&e1, &aa),
                //     &session_ctx.op_mul(&e2, &v),
                // );
                v2 =
                    session_ctx.op_add(&session_ctx.op_mul(&e1, &aa), &session_ctx.op_mul(&e2, &v));
                // state[5 * i + 3] = e1 * bb + e2
                // state_parts[5 * i + 3] = session_ctx.op_add(
                //     &session_ctx.op_mul(&e1, &bb),
                //     &e2,
                // );
                v3 = session_ctx.op_add(&session_ctx.op_mul(&e1, &bb), &e2);
                // state[5 * i + 4] = qq
                // state_parts[5 * i + 4] = qq;
                v4 = qq;
                // ow @ (r * wkv)
                x = session_ctx.op_add(
                    &x,
                    &session_ctx.op_mul_mat(&layer.att_output, &session_ctx.op_mul(&r, &wkv)),
                );
            }

            // FFN/channel mixing
            {
                // self.layer_norm(x, self.w.blocks[i].ln2)
                let x0 = session_ctx.op_rwkv_layer_norm(&x, &layer.ln2_weight, &layer.ln2_bias);
                // state[5 * i + 0]
                let x_prev =
                    session_ctx.op_view_1d(&state, n_embd, (5 * i + 0) * n_embd * FP32_SIZE);
                // xk = x * time_mix_k + state[5 * i + 0] * (1 - time_mix_k)
                // xr = x * time_mix_r + state[5 * i + 0] * (1 - time_mix_r)
                let xk = session_ctx.op_add(
                    &session_ctx.op_mul(&x0, &layer.ffn_time_mix_k),
                    &session_ctx.op_mul(&x_prev, &session_ctx.op_1_minus_x(&layer.ffn_time_mix_k)),
                );
                let xr = session_ctx.op_add(
                    &session_ctx.op_mul(&x0, &layer.ffn_time_mix_r),
                    &session_ctx.op_mul(&x_prev, &session_ctx.op_1_minus_x(&layer.ffn_time_mix_r)),
                );
                // state[5 * i + 0] = x
                // state_parts[5 * i + 0] = x0;
                v0 = x0;

                // r = torch.sigmoid(rw @ xr)
                let r = session_ctx.op_sigmoid(&session_ctx.op_mul_mat(&layer.ffn_receptance, &xr));
                // k = torch.square(torch.relu(kw @ xk))
                let k = session_ctx
                    .op_sqr(&session_ctx.op_relu(&session_ctx.op_mul_mat(&layer.ffn_key, &xk)));
                // r * (vw @ k)
                x = session_ctx.op_add(
                    &x,
                    &session_ctx.op_mul(&r, &session_ctx.op_mul_mat(&layer.ffn_value, &k)),
                );
            }
            state_parts.push(v0);
            state_parts.push(v1);
            state_parts.push(v2);
            state_parts.push(v3);
            state_parts.push(v4);
        }

        // x = self.layer_norm(x, self.w.ln_out)
        x = session_ctx.op_rwkv_layer_norm(&x, &model.ln_out_weight, &model.ln_out_bias);

        // x = (self.w.head.weight @ x).float()
        let logits = session_ctx.op_mul_mat(&model.head, &x);
        let mut graph = ComputationGraph::new(params.n_thread);
        graph.build_forward_expand(&logits);

        for i in 0..n_layer * 5 {
            graph.build_forward_expand(&state_parts[i])
        }

        InferenceSession {
            _session_ctx: session_ctx,
            memory_size: ctx_size,
            params,
            token_index,
            state,
            state_parts,
            logits,
            graph,
            mem_per_token: 0,
            tokens: vec![],
            last_logits: vec![0.0; n_vocab],
        }
    }

    pub fn sample_top_p_top_k(
        &self,
        params: &InferenceParameters,
        rng: &mut impl rand::Rng,
    ) -> TokenId {
        let logits = &self.last_logits;
        let n_logits = logits.len();
        let mut logits_id = Vec::<(f32, TokenId)>::with_capacity(n_logits);

        {
            let scale = 1.0 / params.temperature;
            for (i, &logit) in logits.iter().enumerate() {
                let tid = i as TokenId;

                let val = if let Some(logit_override) = params.bias_tokens.get(tid) {
                    logit_override
                } else if self.repetition_penalty_tokens().contains(&(i as TokenId)) {
                    // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                    // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main

                    // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if logits[i] < 0.0 {
                        logit * scale * params.repeat_penalty
                    } else {
                        logit * scale / params.repeat_penalty
                    }
                } else {
                    logit * scale
                };
                logits_id.push((val, tid));
            }
        }

        // find the top K tokens
        {
            logits_id.partial_sort(params.top_k, |a, b| {
                // Sort descending
                b.0.total_cmp(&a.0)
            });
            logits_id.truncate(params.top_k);
        }

        let maxl = logits_id
            .iter()
            .map(|x| x.0)
            .max_by(f32::total_cmp)
            .unwrap();

        // compute probs for the top K tokens
        let mut probs: Vec<f32> = logits_id
            .iter()
            .copied()
            .map(|(k, _)| (k - maxl).exp())
            .collect();
        let sum: f32 = probs.iter().copied().sum();

        // Normalize the probs
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top p sampling
        if params.top_p < 1.0 {
            let mut cumsum = 0.0;
            for i in 0..probs.len() {
                cumsum += probs[i];
                if cumsum >= params.top_p {
                    probs.truncate(i + 1);
                    logits_id.truncate(i + 1);
                    break;
                }
            }

            cumsum = 1.0 / cumsum;
            for p in probs.iter_mut() {
                *p *= cumsum;
            }
        }

        let dist = WeightedIndex::new(&probs).expect("WeightedIndex error");
        let idx = dist.sample(rng);

        logits_id[idx].1
    }

    fn repetition_penalty_tokens(&self) -> &[TokenId] {
        &self.tokens[self
            .tokens
            .len()
            .saturating_sub(self.params.repetition_penalty_last_n)..]
    }

    pub fn evaluate(
        &mut self,
        model: &Model,
        input_tokens: &[TokenId],
        // output_request: &mut EvaluateOutputRequest,
    ) {
        let n_layer = model.hparams.n_layer;
        let n_embed = model.hparams.n_embed;
        let n_vocab = model.hparams.n_vocab;

        let ctx = &self._session_ctx;

        // for first time
        // if self.mem_per_token == 0 {
        //     ctx.op_set_f32(&self.state, 0.0f32);
        //     for i in 0..n_layer {
        //         ctx.op_set_f32(
        //             &ctx.op_view_1d(&self.state, n_embed, (5 * i + 4) * n_embed * FP32_SIZE),
        //             -1e30,
        //         );
        //     }
        // }

        for token in input_tokens {
            ctx.op_set_i32(&self.token_index, 0);
            ctx.op_set_i32_1d(&self.token_index, 0, *token);

            ctx.graph_compute(&mut self.graph);

            for i in 0..(n_layer * 5) {
                let part = &self.state_parts[i];
                unsafe {
                    let src = part.data();
                    let dst = self.state.data();
                    let count = part.get_ne()[0] as usize * FP32_SIZE;
                    copy_nonoverlapping(src, dst.wrapping_offset((i * n_embed) as isize), count);
                }
            }

            unsafe {
                let src = self.logits.data();
                let dst = self.last_logits.as_mut_ptr();

                self.logits
                    .read_data(0, bytemuck::cast_slice_mut(self.last_logits.as_mut_slice()));
            }
        }

        println!("{}mb", self._session_ctx.used_mem() / 1024 / 1024)
    }
}
