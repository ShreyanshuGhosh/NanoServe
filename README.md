# NanoServe

A minimal LLM inference engine built from scratch on custom Triton GPU kernels, to learn how language models actually run on GPUs.

Every component is written by hand, benchmarked against PyTorch, and checked for correctness before it is optimised. All results are on a free Colab **NVIDIA T4** (16 GB, ~320 GB/s peak memory bandwidth, 65 TFLOPS fp16 tensor-core peak).

---

## v0.1 — Fused kernels

### The idea

Operations like softmax and LayerNorm do very little arithmetic per byte of data, so they are **memory-bound**: their speed is set by how fast data moves between the GPU's large, slow global memory (HBM) and its compute units, not by how fast it can do maths.

Naive softmax in PyTorch runs as several separate kernels (max, subtract, exp, sum, divide), and each one reads and often writes the whole matrix in HBM — roughly 5 full reads and 3 full writes. A **fused** kernel loads each row once, does all the work in fast on-chip memory, and writes the result once.

### Results

Effective bandwidth = minimum bytes the operation must move ÷ measured time.

| Kernel | Best effective bandwidth | % of T4 peak (320 GB/s) | Speedup vs PyTorch baseline |
|---|---|---|---|
| Fused softmax | 230.0 GB/s | 71.9% | 3.8× |
| Online softmax | 215.4 GB/s | 67.3% | 3.5× |
| LayerNorm (forward) | 232.9 GB/s | 72.8% | 1.4× |

Softmax speedups are against the naive multi-kernel PyTorch implementation; LayerNorm is against `torch.nn.functional.layer_norm`.

![softmax bandwidth](assets/softmax.png)

### Online softmax

The fused kernel needs a whole row to fit in one block of on-chip memory. **Online softmax** removes that limit: it walks the row in chunks, keeping a running max `m` and running sum `d`. When a new chunk raises the max, the old sum is corrected:

```
m_new = max(m, max(chunk))
d     = d · exp(m − m_new) + Σ exp(chunk − m_new)
```

It reads the input twice, so it is slightly slower on short rows, but it handles rows of any length. This running-max trick is the foundation of FlashAttention (v0.2).

### Notes

- Padding lanes are loaded as `-inf` for softmax (so they never win the max and add `exp(-inf) = 0` to the sum), and masked back to zero before computing LayerNorm's variance.
- Statistics are accumulated in fp32.
- The PyTorch profiler confirms the difference: naive softmax launches several kernels per call; the fused version launches one.

---

## v0.2 — FlashAttention forward kernel

### The problem

Standard attention, `softmax(QKᵀ / √d) V`, builds an **N × N** score matrix in GPU memory, where N is the sequence length. Doubling N quadruples memory.

### The idea

Never build that matrix. Each Triton program takes a block of 64 query rows and loops over 64-row blocks of K and V, computing a small 64 × 64 tile of scores at a time in on-chip memory. Week 1's online softmax is extended with a third running value — the output accumulator — which is rescaled by the same factor as the running sum whenever the max changes:

```
m_new = max(m, rowmax(S))
α     = exp(m − m_new)
P     = exp(S − m_new)
l     = α · l + rowsum(P)
acc   = α · acc + P · V
out   = acc / l          # after the last block
```

The result is mathematically identical to standard attention, and memory drops from O(N²) to O(N).

**Causal masking** (GPT-style models): scores where a key comes after its query are masked inside each block, and key blocks lying entirely in the future are skipped without being loaded.

Correctness is tested against `torch.nn.functional.scaled_dot_product_attention` on awkward sizes (N not a multiple of the block size), with and without the causal mask.

### Results — memory

Batch 2, 8 heads, head dimension 64, fp16. Extra GPU memory needed beyond the inputs:

| Sequence length | Naive attention | My kernel | Reduction |
|---|---|---|---|
| 8,192 | 10,240 MB | 16 MB | 640× |
| 16,384 | **out of memory** | 32 MB | — |

My kernel's memory grows linearly with N (only the output tensor), and it runs at 16,384 tokens where naive attention fails on the T4.

![attention memory](assets/attention_memory.png)

### Results — speed (known issue)

| N = 8,192, non-causal | TFLOPS |
|---|---|
| PyTorch SDPA | 12.7 |
| Naive attention | 2.1 |
| My kernel | 0.5 |
| My kernel, causal | 0.1 |

**This is a performance bug, not a hardware limit, and I'm debugging it.** Two signs: causal attention skips about half the work, so it should never be slower than non-causal; and PyTorch reaches 12.7 TFLOPS on the same GPU, so the T4's tensor cores work fine. The leading suspect is the attention-times-V matmul running in fp32 (bypassing the tensor cores) instead of fp16. I'll update this table once it's fixed.

## What I learned

- **Measure before optimising.** GPU calls are asynchronous; timing without synchronising measures how long it takes to *queue* work, not do it. Every number here comes from `triton.testing.do_bench`.
- **Report against the hardware limit**, not just in milliseconds: "72% of peak bandwidth" says how much room is left.
- **Correct first, fast second.** Every kernel is checked against PyTorch before it is benchmarked.
- **Results that look wrong usually are.** A causal kernel slower than a non-causal one is a bug to find, not a hardware quirk to explain away.

## References

- Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) — vector add, fused softmax, layer norm, fused attention
- Milakov & Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (2018)
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022)
- Sasha Rush, [Triton-Puzzles](https://github.com/srush/Triton-Puzzles)
