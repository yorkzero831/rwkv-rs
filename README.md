# RWKV, but in Rust with ggml

A project which reimplementation [llama.cpp](https://github.com/saharNooby/rwkv.cpp) in [rustformers/llm](https://github.com/rustformers/llm) style.

## Current situation
Performance issue: current version only has about 30% performance of rwkv.cpp

## Acknowledgments
This library was published under MIT/Apache-2.0 license. However, we strongly recommend you to cite our work/our dependencies work if you wish to reuse the code from this library.

### Models/Inferencing tools dependencies

-   RWKV models: [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
-   llm: [rustformers/llm](https://github.com/rustformers/llm)
-   rwkv.cpp: [saharNooby/rwkv.cpp](https://github.com/saharNooby/rwkv.cpp)

### Some source code comes from
-   interface design: [rustformers/llm](https://github.com/rustformers/llm)
