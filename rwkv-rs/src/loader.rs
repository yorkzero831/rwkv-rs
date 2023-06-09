use crate::model::{Hyperparameters, Model};
use std::env::temp_dir;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};
use std::process::abort;

use crate::rwkv_session::InferenceSessionParameters;
use crate::util::mulf;
use thiserror::Error;

/// Each variant represents a step within the process of loading the model.
/// These can be used to report progress to the user.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LoadProgress<'a> {
    /// The hyperparameters have been loaded from the model.
    HyperparametersLoaded(&'a Hyperparameters),
    /// The context has been created.
    ContextSize {
        /// The size of the context.
        bytes: usize,
    },
    /// A part of the model is being loaded.
    PartLoading {
        /// The path to the model part.
        file: &'a Path,
        /// The current part (0-indexed).
        current_part: usize,
        /// The number of total parts.
        total_parts: usize,
    },
    /// A tensor from the current part has been loaded.
    PartTensorLoaded {
        /// The path to the model part.
        file: &'a Path,
        /// The current tensor (0-indexed).
        current_tensor: usize,
        /// The number of total tensors.
        tensor_count: usize,
    },
    /// A model part has finished fully loading.
    PartLoaded {
        /// The path to the model part.
        file: &'a Path,
        /// The number of bytes in the part.
        byte_size: usize,
        /// The number of tensors in the part.
        tensor_count: usize,
    },
}

#[derive(Error, Debug)]
/// Errors encountered during the loading process.
pub enum LoadError {
    #[error("could not open file {path:?}")]
    /// A file failed to open.
    OpenFileFailed {
        /// The original error.
        source: std::io::Error,
        /// The path that failed.
        path: PathBuf,
    },
    #[error("no parent path for {path:?}")]
    /// There is no parent path for a given path.
    NoParentPath {
        /// The path without a parent.
        path: PathBuf,
    },
    #[error("unable to read exactly {bytes} bytes")]
    /// Reading exactly `bytes` from a file failed.
    ReadExactFailed {
        /// The original error.
        source: std::io::Error,
        /// The number of bytes that were attempted to be read.
        bytes: usize,
    },
    #[error("non-specific I/O error")]
    /// A non-specific IO error.
    IO(#[from] std::io::Error),
    #[error("could not convert bytes to a UTF-8 string")]
    /// One of the strings encountered was not valid UTF-8.
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("invalid integer conversion")]
    /// One of the integers encountered could not be converted to a more appropriate type.
    InvalidIntegerConversion(#[from] std::num::TryFromIntError),
    #[error("invalid magic number for {path:?}")]
    /// An invalid magic number was encountered during the loading process.
    InvalidMagic {
        /// The path that failed.
        path: PathBuf,
    },
    #[error("invalid file format version {value}")]
    /// The version of the format is not supported by this version of `llama-rs`.
    InvalidFormatVersion {
        /// The version that was encountered.
        value: u32,
    },
    #[error("invalid value {ftype} for `f16` in hyperparameters")]
    /// The `f16` hyperparameter had an invalid value.
    HyperparametersF16Invalid {
        /// The format type that was encountered.
        ftype: u32,
    },
    #[error("unknown tensor `{tensor_name}` in {path:?}")]
    /// The tensor `tensor_name` was encountered during the loading of `path`, but was not seen during
    /// the model prelude.
    UnknownTensor {
        /// The name of the tensor.
        tensor_name: String,
        /// The path that failed.
        path: PathBuf,
    },
    #[error("the tensor `{tensor_name}` has the wrong size in {path:?}")]
    /// The tensor `tensor_name` did not match its expected size.
    TensorWrongSize {
        /// The name of the tensor.
        tensor_name: String,
        /// The path that failed.
        path: PathBuf,
    },
    /// The tensor `tensor_name` did not have the expected format type.
    #[error("invalid ftype {ftype} for tensor `{tensor_name}` in {path:?}")]
    InvalidFtype {
        /// The name of the tensor.
        tensor_name: String,
        /// The format type that was encountered.
        ftype: u32,
        /// The path that failed.
        path: PathBuf,
    },
}

pub const RWKV_FORMAT_VERSION: u32 = 100;

pub fn load(
    path: impl AsRef<Path>,
    mut load_progress_callback: impl FnMut(LoadProgress),
) -> Result<Model, LoadError> {
    use std::fs::File;
    use std::io::BufReader;

    let main_path = path.as_ref();

    let file_size = main_path.metadata().unwrap().len();

    let mut reader =
        BufReader::new(
            File::open(main_path).map_err(|e| LoadError::OpenFileFailed {
                source: e,
                path: main_path.to_owned(),
            })?,
        );

    // Verify magic
    let is_legacy_model: bool = match read_u32(&mut reader)? {
        ggml_rwkv::FILE_MAGIC => false,
        ggml_rwkv::FILE_MAGIC_UNVERSIONED => true,
        magic => {
            println!("MAGIC: {:#01x}", magic);
            return Err(LoadError::InvalidMagic {
                path: main_path.to_owned(),
            });
        }
    };

    // Load format version
    if !is_legacy_model {
        #[allow(unused_variables)]
        let version: u32 = match read_u32(&mut reader)? {
            RWKV_FORMAT_VERSION => RWKV_FORMAT_VERSION,
            version => return Err(LoadError::InvalidFormatVersion { value: version }),
        };
    }

    // =================
    // Load hyper params
    // =================

    // NOTE: Field order matters! Data is laid out in the file exactly
    // in this order.
    let hparams = Hyperparameters {
        n_vocab: read_i32(&mut reader)?.try_into()?,
        n_embed: read_i32(&mut reader)?.try_into()?,
        n_layer: read_i32(&mut reader)?.try_into()?,
        f16_: read_i32(&mut reader)?.try_into()?,
    };

    load_progress_callback(LoadProgress::HyperparametersLoaded(&hparams));

    // for the big tensors, we have the option to store the data in 16-bit
    // floats or quantized in order to save memory and also to speed up the
    // computation
    let wtype = match hparams.f16_ {
        0 => ggml_rwkv::Type::F32,
        1 => ggml_rwkv::Type::F16,
        2 => ggml_rwkv::Type::Q4_0,
        3 => ggml_rwkv::Type::Q4_1,
        invalid => return Err(LoadError::HyperparametersF16Invalid { ftype: invalid }),
    };

    let n_embed = hparams.n_embed;
    let n_layer = hparams.n_layer;
    let n_vocab = hparams.n_vocab;

    println!(
        "size:{}, file_size: {}",
        ggml_rwkv::type_sizef(ggml_rwkv::Type::F32),
        file_size
    );
    println!(
        "n_embd:{}, n_layer:{}, n_vocab: {}",
        n_embed, n_layer, n_vocab
    );

    println!(
        "F16:{}, F32:{}, Q4_0: {}, Q4_1:{}",
        ggml_rwkv::type_sizef(ggml_rwkv::Type::F16),
        ggml_rwkv::type_sizef(ggml_rwkv::Type::F32),
        ggml_rwkv::type_sizef(ggml_rwkv::Type::Q4_0),
        ggml_rwkv::type_sizef(ggml_rwkv::Type::Q4_1)
    );

    let ctx_size2 = {
        // Use 64-bit math to prevent overflow.
        let mut ctx_size: usize = 0;
        ctx_size += mulf!(1, file_size);

        // Intermediary vectors for calculation; there are around 100 calls to ggml
        ctx_size += mulf!(100, n_embed, ggml_rwkv::type_sizef(ggml_rwkv::Type::F32));
        // State, in and out
        ctx_size += mulf!(
            (2 * 5),
            n_layer,
            n_embed,
            ggml_rwkv::type_sizef(ggml_rwkv::Type::F32)
        );
        // Logits
        ctx_size += mulf!(n_vocab, ggml_rwkv::type_sizef(ggml_rwkv::Type::F32));
        // +256 MB just for any overhead
        // TODO This is too much for smaller models; need a more proper and robust way of measuring required memory
        ctx_size += (256) * 1024 * 1024;

        //load_progress_callback(LoadProgress::ContextSize { bytes: ctx_size });

        ctx_size
    };

    let ctx_size = {
        // Use 64-bit math to prevent overflow.
        let mut ctx_size: usize = 0;

        //n_embd:2560, n_layer:32, n_vocab: 50277
        //F16:2, F32:4, Q4_0: 0.625, Q4_1:0.75

        // quantize only 2d tensors, except embedding and head matrices.
        // emb.weight, head.weight
        ctx_size += mulf!(2, n_embed, n_vocab, ggml_rwkv::type_sizef(wtype));
        // ln_out.weight ln_out.bias
        ctx_size += mulf!(2, n_embed, ggml_rwkv::type_sizef(ggml_rwkv::Type::F32));
        // ln0.weight, ln0.bias
        // only exist in blocks.0
        ctx_size += mulf!(2, n_embed, ggml_rwkv::type_sizef(ggml_rwkv::Type::F32));
        // ln1.weight, ln1.bias, ln2.weight, ln2.bias
        ctx_size += mulf!(
            n_layer,
            4,
            n_embed,
            ggml_rwkv::type_sizef(ggml_rwkv::Type::F32)
        );
        // att.time_decay, att.time_first, att.time_mix_k, att.time_mix_v, att.time_mix_r
        ctx_size += mulf!(
            n_layer,
            5,
            n_embed,
            ggml_rwkv::type_sizef(ggml_rwkv::Type::F32)
        );
        // att.key.weight, att.value.weight, att.receptance.weight, att.output.weight
        ctx_size += mulf!(n_layer, 4, n_embed, n_embed, ggml_rwkv::type_sizef(wtype));
        // ffn.time_mix_k, ffn.time_mix_r
        ctx_size += mulf!(
            n_layer,
            2,
            n_embed,
            ggml_rwkv::type_sizef(ggml_rwkv::Type::F32)
        );
        // ffn.key.weight
        ctx_size += mulf!(n_layer, n_embed, 4 * n_embed, ggml_rwkv::type_sizef(wtype));
        // ffn.receptance.weight
        ctx_size += mulf!(n_layer, n_embed, n_embed, ggml_rwkv::type_sizef(wtype));
        // ffn.value.weight
        ctx_size += mulf!(n_layer, 4 * n_embed, n_embed, ggml_rwkv::type_sizef(wtype));
        // // Intermediary vectors for calculation; there are around 100 calls to ggml
        // ctx_size += mulf!(100, n_embd, ggml_rwkv::type_sizef(ggml_rwkv::Type::F32));
        // // State, in and out
        // ctx_size += mulf!((2 * 5), n_layer, n_embd, ggml_rwkv::type_sizef(ggml_rwkv::Type::F32));
        // // Logits
        // ctx_size += mulf!(n_vocab, ggml_rwkv::type_sizef(ggml_rwkv::Type::F32));
        // +256 MB just for any overhead
        // TODO This is too much for smaller models; need a more proper and robust way of measuring required memory
        ctx_size += (256) * 1024 * 1024;

        load_progress_callback(LoadProgress::ContextSize { bytes: ctx_size });

        ctx_size
    };

    println!("context size: {} size2: {}", ctx_size, ctx_size2);

    // Initialize the context
    let context = ggml_rwkv::Context::init(ctx_size);

    let model = Model::new(context, hparams, wtype);

    // hence rwkv usually has only one model file, so ignore partially read.

    let mut total_size = 0;
    let mut n_tensors = 0;
    loop {
        let is_eof = reader.fill_buf().map(|b| b.is_empty())?;

        if is_eof {
            break;
        }

        let n_dims = usize::try_from(read_i32(&mut reader)?)?;
        let key_length = read_i32(&mut reader)?;
        let data_t = read_i32(&mut reader)?;

        let ggml_data_type = match data_t {
            0 => ggml_rwkv::Type::F32,
            1 => ggml_rwkv::Type::F16,
            2 => ggml_rwkv::Type::Q4_0,
            3 => ggml_rwkv::Type::Q4_1,
            invalid => {
                return Err(LoadError::HyperparametersF16Invalid {
                    ftype: invalid as u32,
                })
            }
        };

        let mut nelements = 1;
        let mut ne = [1i64, 1i64];

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_dims {
            ne[i] = read_i32(&mut reader)? as i64;
            nelements *= usize::try_from(ne[i])?;
        }

        let tensor_name = read_string(&mut reader, key_length as usize)?;

        // if tensor_name.contains("emb.weight") || tensor_name.contains("head.weight") {
        //     nelements = mulf!(nelements, ggml_rwkv::type_sizef(wtype))
        // }

        let Some(tensor) = model.tensors().get(&tensor_name)
            else {
                return Err(LoadError::UnknownTensor { tensor_name, path: main_path.to_owned() });
            };

        //println!("n_dims:{}, ne[0]:{}, ne[1]:{}, tensor_name:{}, nelements: {}, nbytes: {}", n_dims, ne[0], ne[1], tensor_name, tensor.nelements(), tensor.nbytes());

        if tensor.nelements() != nelements {
            return Err(LoadError::TensorWrongSize {
                tensor_name,
                path: main_path.to_owned(),
            });
        }

        let slice = unsafe {
            let data = tensor.data();
            std::slice::from_raw_parts_mut(data as *mut u8, tensor.nbytes())
        };
        reader.read_exact(slice);
        n_tensors = n_tensors + 1;
        total_size += tensor.nbytes();
    }

    println!(
        "Loaded tensor_count:{}, total_bytes:{}",
        n_tensors, total_size
    );

    load_progress_callback(LoadProgress::PartTensorLoaded {
        file: main_path,
        current_tensor: n_tensors.try_into()?,
        tensor_count: model.tensors().len(),
    });

    load_progress_callback(LoadProgress::PartLoaded {
        file: main_path,
        byte_size: total_size,
        tensor_count: n_tensors.try_into()?,
    });

    Ok(model)
}

pub fn read_bytes<const N: usize>(reader: &mut impl BufRead) -> Result<[u8; N], LoadError> {
    let mut bytes = [0u8; N];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LoadError::ReadExactFailed {
            source: e,
            bytes: N,
        })?;
    Ok(bytes)
}

pub fn read_bytes_with_len(reader: &mut impl BufRead, len: usize) -> Result<Vec<u8>, LoadError> {
    let mut bytes = vec![0u8; len];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LoadError::ReadExactFailed {
            source: e,
            bytes: len,
        })?;
    Ok(bytes)
}

pub fn read_i32(reader: &mut impl BufRead) -> Result<i32, LoadError> {
    Ok(i32::from_le_bytes(read_bytes::<4>(reader)?))
}

pub fn read_u32(reader: &mut impl BufRead) -> Result<u32, LoadError> {
    Ok(u32::from_le_bytes(read_bytes::<4>(reader)?))
}

pub fn read_f32(reader: &mut impl BufRead) -> Result<f32, LoadError> {
    Ok(f32::from_le_bytes(read_bytes::<4>(reader)?))
}

/// Helper function. Reads a string from the buffer and returns it.
pub fn read_string(reader: &mut impl BufRead, len: usize) -> Result<String, LoadError> {
    Ok(String::from_utf8(read_bytes_with_len(reader, len)?)?)
}

#[test]
fn test() {
    let model = load(
        "D:/AI/rwkv/RWKV-4-Raven-3B-v7-ChnEng-Q4_1.bin",
        |progress| {},
    )
    .expect("TODO: panic message");
    let mut session = model.start_session(InferenceSessionParameters::default());
    let input_tokenss = vec![12i32];
    session.evaluate(&model, input_tokenss.as_slice());
    println!("{}", 1)
}
