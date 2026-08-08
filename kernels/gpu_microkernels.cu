/*
 * BSM-RLI GPU CUDA Micro-Kernel Engine
 * Direct VRAM Tensor Execution with Zero PCIe Transfer Overhead.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// 1. GPU Bounding Box IoU Micro-Kernel
__global__ void bsm_rli_calc_iou_kernel(
    const float* __restrict__ box1,
    const float* __restrict__ box2,
    float* __restrict__ iou_result,
    int num_boxes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_boxes) {
        int base = idx * 4;
        float x1 = max(box1[base + 0], box2[base + 0]);
        float y1 = max(box1[base + 1], box2[base + 1]);
        float x2 = min(box1[base + 2], box2[base + 2]);
        float y2 = min(box1[base + 3], box2[base + 3]);

        float inter_area = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
        float area1 = (box1[base + 2] - box1[base + 0]) * (box1[base + 3] - box1[base + 1]);
        float area2 = (box2[base + 2] - box2[base + 0]) * (box2[base + 3] - box2[base + 1]);

        float union_area = area1 + area2 - inter_area;
        iou_result[idx] = (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
    }
}

// 2. GPU Parallel Vector Sum Micro-Kernel
__global__ void bsm_rli_sum_f32_kernel(
    const float* __restrict__ input_vec,
    float* __restrict__ total_sum,
    int num_elements
) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < num_elements) ? input_vec[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(total_sum, sdata[0]);
    }
}
