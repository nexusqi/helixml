//! 🌀 HelixML CUDA Backend
//! 
//! High-performance CUDA backend with cuBLAS optimization for SSM/Hyena architectures.

#![recursion_limit = "1024"]

use tensor_core::{Tensor, Shape, DType, Device, Result, TensorError};
use tensor_core::tensor::{TensorOps, TensorReduce, TensorStats, TensorActivation, TensorRandom, TensorBroadcast, TensorMixedPrecision};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use cudarc::cublas::CudaBlas;

#[cfg(feature = "cuda")]
mod kernels;
#[cfg(feature = "cuda")]
use kernels::{CudaKernels, configs};

mod tensor_impl;
mod ops_impl;
mod traits_impl;

/// CUDA tensor implementation using cudarc
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CudaTensor {
    data: CudaSlice<f32>,
    shape: Shape,
    dtype: DType,
    device: Device,
    cuda_device: Arc<CudaDevice>,
    cublas: Arc<CudaBlas>,
}

/// Placeholder CUDA tensor for systems without CUDA
#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CudaTensor {
    shape: Shape,
    dtype: DType,
    device: Device,
}

#[cfg(feature = "cuda")]
impl CudaTensor {
    /// Create a new CUDA tensor
    pub fn new(data: CudaSlice<f32>, shape: Shape, dtype: DType, cuda_device: Arc<CudaDevice>) -> Result<Self> {
        let cublas = Arc::new(CudaBlas::new(cuda_device.clone())?);
        
        Ok(Self {
            data,
            shape,
            dtype,
            device: Device::cuda(0),
            cuda_device,
            cublas,
        })
    }
    
    /// Get the underlying CUDA device
    pub fn cuda_device(&self) -> &Arc<CudaDevice> {
        &self.cuda_device
    }
    
    /// Get the cuBLAS handle
    pub fn cublas(&self) -> &Arc<CudaBlas> {
        &self.cublas
    }
}

#[cfg(not(feature = "cuda"))]
impl CudaTensor {
    /// Create a placeholder CUDA tensor
    pub fn new(_data: (), shape: Shape, dtype: DType, _cuda_device: ()) -> Result<Self> {
        Ok(Self {
            shape,
            dtype,
            device: Device::cuda(0),
        })
    }
}