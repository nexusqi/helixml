//! 🌀 HelixML CUDA Kernels
//! 
//! High-performance CUDA kernels for tensor operations

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Element-wise operations
__global__ void elementwise_add_kernel(
    const float* a,
    const float* b,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void elementwise_sub_kernel(
    const float* a,
    const float* b,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] - b[idx];
    }
}

__global__ void elementwise_mul_kernel(
    const float* a,
    const float* b,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void elementwise_div_kernel(
    const float* a,
    const float* b,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] / b[idx];
    }
}

__global__ void elementwise_max_kernel(
    const float* a,
    const float* b,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fmaxf(a[idx], b[idx]);
    }
}

__global__ void elementwise_min_kernel(
    const float* a,
    const float* b,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fminf(a[idx], b[idx]);
    }
}

// Mathematical functions
__global__ void elementwise_pow_kernel(
    const float* input,
    float* result,
    float exponent,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = powf(input[idx], exponent);
    }
}

__global__ void elementwise_sqrt_kernel(
    const float* input,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = sqrtf(input[idx]);
    }
}

__global__ void elementwise_exp_kernel(
    const float* input,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = expf(input[idx]);
    }
}

__global__ void elementwise_log_kernel(
    const float* input,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = logf(input[idx]);
    }
}

__global__ void elementwise_sin_kernel(
    const float* input,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = sinf(input[idx]);
    }
}

__global__ void elementwise_cos_kernel(
    const float* input,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = cosf(input[idx]);
    }
}

__global__ void elementwise_tan_kernel(
    const float* input,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = tanf(input[idx]);
    }
}

__global__ void elementwise_abs_kernel(
    const float* input,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fabsf(input[idx]);
    }
}

__global__ void elementwise_sign_kernel(
    const float* input,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = (input[idx] > 0.0f) ? 1.0f : ((input[idx] < 0.0f) ? -1.0f : 0.0f);
    }
}

__global__ void elementwise_clamp_kernel(
    const float* input,
    float* result,
    float min_val,
    float max_val,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fminf(fmaxf(input[idx], min_val), max_val);
    }
}

// Activation functions
__global__ void elementwise_relu_kernel(
    const float* input,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void elementwise_gelu_kernel(
    const float* input,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
        result[idx] = x * cdf;
    }
}

__global__ void elementwise_silu_kernel(
    const float* input,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        result[idx] = x / (1.0f + expf(-x));
    }
}

__global__ void elementwise_sigmoid_kernel(
    const float* input,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void elementwise_tanh_kernel(
    const float* input,
    float* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = tanhf(input[idx]);
    }
}

__global__ void elementwise_leaky_relu_kernel(
    const float* input,
    float* result,
    float negative_slope,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        result[idx] = (x >= 0.0f) ? x : negative_slope * x;
    }
}

// Matrix operations
__global__ void transpose_2d_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        int row = idx / cols;
        int col = idx % cols;
        int transposed_idx = col * rows + row;
        output[transposed_idx] = input[idx];
    }
}

// Reduction operations
__global__ void sum_reduce_kernel(
    const float* input,
    float* output,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void max_reduce_kernel(
    const float* input,
    float* output,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : -INFINITY;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void min_reduce_kernel(
    const float* input,
    float* output,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : INFINITY;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
