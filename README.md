# NanoServe

NanoServe — built a minimal LLM inference engine from scratch with custom Triton kernels (fused softmax, LayerNorm, FlashAttention), paged KV cache and continuous batching; served a 1B-parameter model at X tokens/sec, benchmarked against Hugging Face and vLLM.
