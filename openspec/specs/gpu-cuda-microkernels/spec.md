# Capability Specification: GPU CUDA & Triton Micro-Kernel Acceleration

> **Direct GPU VRAM Tensor Execution, Zero-Copy PCIe Overhead, and Persistent CUDA Stream Dispatch**

---

## 1. Specification Overview

This specification defines the execution architecture for running BSM-RLI micro-kernels directly on NVIDIA CUDA / Triton GPU hardware.

---

## 2. Requirements & Guarantees

### Requirement 1: Zero-Copy VRAM Memory Access
- **MUST** extract micro-kernel input arguments directly from GPU model logits and KV-cache tensors in GPU memory.
- **MUST NOT** trigger PCIe host-to-device (`cudaMemcpyHostToDevice`) or device-to-host (`cudaMemcpyDeviceToHost`) transfers for tensor micro-kernels.

### Requirement 2: Sub-Microsecond GPU Execution Latency
- Mathematical expressions (`GPU_EVAL_EXPR`), parallel graph searches (`GPU_GRAPH_DIJKSTRA`), and SAT solvers (`GPU_SOLVE_SAT`) **MUST** execute within `< 3.0 microseconds` on NVIDIA CUDA hardware (RTX 4070 Ti / H100 / Jetson Orin).

### Requirement 3: CUDA Graph & Persistent Stream Dispatch
- **MUST** maintain dedicated persistent CUDA streams (`cudaStream_t`) and pre-recorded CUDA Graphs to eliminate driver kernel launch overhead (`cudaLaunchKernel` < 0.5 µs).

---

## 3. Micro-Kernel GPU Implementation API

```cpp
// Direct CUDA Kernel Execution Interface
__global__ void bsm_rli_eval_expr_kernel(
    const float* __restrict__ input_tokens,
    float* __restrict__ output_results,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        // Direct warp-level parallel evaluation
        output_results[idx] = evaluate_symbolic_expr_device(input_tokens[idx]);
    }
}
```

---

## 4. CUDA vs Host CPU Performance Targets

| Micro-Kernel Operation | Target Hardware | Execution Latency | PCIe Memory Overhead | Precision Guarantee |
| :--- | :--- | :--- | :--- | :--- |
| **`GPU_EVAL_EXPR` (Math)** | CUDA Core Warp | **`0.45 µs`** | **`0 bytes (Zero-Copy)`** | **`100.0% Exact IEEE 754`** |
| **`GPU_GRAPH_DIJKSTRA` (Graph)** | CUDA Shared Mem | **`1.12 µs`** | **`0 bytes (Zero-Copy)`** | **`100.0% Exact`** |
| **`GPU_FFT_AUDIO` (Audio)** | cuFFT / Triton | **`2.15 µs`** | **`0 bytes (Zero-Copy)`** | **`100.0% Exact`** |
| **`GPU_CALC_IOU` (Vision)** | CUDA SIMD | **`0.08 µs`** | **`0 bytes (Zero-Copy)`** | **`100.0% Exact`** |
