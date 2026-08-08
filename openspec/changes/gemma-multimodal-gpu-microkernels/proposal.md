# Change Proposal: Gemma Multimodal Integration & CUDA GPU Micro-Kernel Engine

> **Integrating Google Gemma Edge Models (Gemma-2B/9B, PaliGemma) and GPU CUDA / Triton Micro-Kernel Execution**

---

## 1. Executive Summary

This change introduces two core architectural enhancements to the BSM-RLI framework:
1. **Gemma Multimodal Edge Model Integration**: Full support for Google's Gemma model family (`Gemma-2B`, `Gemma-2-9B`, `PaliGemma-3B`), aligning SentencePiece `<start_of_turn>` tokens with BSM-RLI trigger delimiters.
2. **GPU CUDA / Triton Micro-Kernel Engine**: Transitioning heavy micro-kernel execution from host CPU to GPU VRAM using custom CUDA / Triton kernels, eliminating PCIe host-device transfer overhead and enabling massive GPU parallelism.

---

## 2. Problem Statement & Motivation

- **PCIe Latency Penalty**: Executing micro-kernels on host CPU requires copying tensor parameters from GPU VRAM to Host RAM over PCIe, introducing a ~1.5–3.0 µs transfer latency penalty.
- **Scale Limits of Host CPU**: Complex micro-kernels (e.g. 10,000-node graph searches, 1024-point FFT audio transforms, or matrix operations) saturate single-threaded CPU execution.
- **Edge Deployment Standardization**: Google's Gemma models represent the dominant open-weights edge model family for local Linux, Jetson Orin, and mobile NPU deployments.

---

## 3. Proposed Solutions

### A. CUDA VRAM Direct-Tensor Micro-Kernels
- Implement custom CUDA micro-kernels in C++/CUDA (`kernels/gpu_kernels.cu`) compiled via PyTorch CUDA extensions / Triton (`@triton.jit`).
- Parameters are read directly from GPU KV-cache tensor memory with **zero host memory copies**.

### B. Dual-Engine Hybrid Dispatcher
- **Host CPU Engine (SIMD AVX-512)**: Dispatches scalar string and character micro-kernels (`COUNT_CHAR`, `REVERSE_STR`) executing in < 100 ns.
- **GPU CUDA Engine**: Dispatches math, graph, matrix, and signal processing micro-kernels (`GPU_GRAPH_DIJKSTRA`, `GPU_FFT_AUDIO`, `GPU_EVAL_EXPR`) executing in parallel CUDA warps.

### C. Gemma Model Alignment
- Add token mapping for Gemma's `<start_of_turn>user` / `<start_of_turn>model` headers and special trigger tokens `<|jit_start|>` / `<|jit_end|>`.

---

## 4. Architectural Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     GPU DIRECT TENSOR MICRO-KERNEL DISPATCH ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  Gemma / Llama Model Logits (GPU VRAM)                                                           │
│        │                                                                                         │
│        ▼                                                                                         │
│  Logit Masking & Trigger Detector                                                                │
│        │                                                                                         │
│        ├────────────────────────────────────────┬────────────────────────────────────────┐       │
│        ▼                                        ▼                                        ▼       │
│  Scalar Micro-Kernels                   Tensor / Math Micro-Kernels              Signal / Vision │
│  (Host CPU AVX-512)                     (GPU CUDA / Triton)                      (GPU CUDA)      │
│  • COUNT_CHAR (< 50 ns)                 • GPU_EVAL_EXPR (CUDA Warp)              • GPU_FFT_AUDIO │
│  • REVERSE_STR (< 60 ns)                • GPU_DIJKSTRA (CUDA Shared Mem)         • GPU_CALC_IOU  │
│        │                                        │                                        │       │
│        └────────────────────────────────────────┴────────────────────────────────────────┘       │
│                                                 │                                                │
│                                                 ▼                                                │
│                              Zero-Copy GPU KV-Cache Insertion                                    │
│                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
