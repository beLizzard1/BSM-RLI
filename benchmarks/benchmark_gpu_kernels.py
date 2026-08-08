"""
BSM-RLI GPU CUDA vs Host CPU Micro-Kernel Benchmark Harness
Measures nanosecond & microsecond p50, p95, p99 latencies for GPU VRAM execution.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import torch
import numpy as np
from kernels.gpu_microkernels import GPUMicroKernelEngine

def benchmark_gpu_microkernels(iterations=1000):
    print("=================================================================")
    print("  BSM-RLI GPU CUDA MICRO-KERNEL LATENCY BENCHMARK MATRIX         ")
    print(f"  Iterations: {iterations} | Target Device: NVIDIA GeForce RTX 4070 Ti")
    print("=================================================================")

    engine = GPUMicroKernelEngine(device="cuda")
    
    # 1. GPU IoU Bounding Box Benchmark
    b1 = torch.rand((100, 4), device=engine.device) * 100.0
    b2 = torch.rand((100, 4), device=engine.device) * 100.0
    
    # Warmup GPU
    for _ in range(50):
        engine.gpu_calc_iou(b1, b2)
    torch.cuda.synchronize()

    iou_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        _, _ = engine.gpu_calc_iou(b1, b2)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        iou_latencies.append((t1 - t0) / 1000.0)

    # 2. GPU Math Sum Evaluation Benchmark
    vec = torch.rand((10000,), device=engine.device)
    for _ in range(50):
        engine.gpu_eval_expr(vec, None, op="sum")
    torch.cuda.synchronize()

    sum_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        _, _ = engine.gpu_eval_expr(vec, None, op="sum")
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        sum_latencies.append((t1 - t0) / 1000.0)

    # 3. GPU Fast Fourier Transform (FFT) Benchmark
    audio_pcm = torch.rand((2048,), device=engine.device)
    for _ in range(50):
        engine.gpu_fft_audio(audio_pcm)
    torch.cuda.synchronize()

    fft_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        _, _ = engine.gpu_fft_audio(audio_pcm)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        fft_latencies.append((t1 - t0) / 1000.0)

    results = {
        "GPU_CALC_IOU": {
            "p50_us": round(float(np.percentile(iou_latencies, 50)), 3),
            "p95_us": round(float(np.percentile(iou_latencies, 95)), 3),
            "p99_us": round(float(np.percentile(iou_latencies, 99)), 3),
        },
        "GPU_EVAL_SUM": {
            "p50_us": round(float(np.percentile(sum_latencies, 50)), 3),
            "p95_us": round(float(np.percentile(sum_latencies, 95)), 3),
            "p99_us": round(float(np.percentile(sum_latencies, 99)), 3),
        },
        "GPU_FFT_AUDIO": {
            "p50_us": round(float(np.percentile(fft_latencies, 50)), 3),
            "p95_us": round(float(np.percentile(fft_latencies, 95)), 3),
            "p99_us": round(float(np.percentile(fft_latencies, 99)), 3),
        }
    }

    print("\n-------------------------------------------------------------")
    print(f"{'GPU Micro-Kernel':<20} | {'p50 (µs)':<10} | {'p95 (µs)':<10} | {'p99 (µs)':<10}")
    print("-------------------------------------------------------------")
    for k, v in results.items():
        print(f"{k:<20} | {v['p50_us']:<10} | {v['p95_us']:<10} | {v['p99_us']:<10}")
    print("-------------------------------------------------------------")

    with open("benchmarks/gpu_kernel_latencies.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nGPU micro-kernel benchmark complete! Saved benchmarks/gpu_kernel_latencies.json")

if __name__ == "__main__":
    benchmark_gpu_microkernels(iterations=1000)
