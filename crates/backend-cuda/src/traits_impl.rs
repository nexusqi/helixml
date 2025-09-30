//! CUDA Tensor trait implementations for all other traits

use tensor_core::{Tensor, Shape, DType, Device, Result, TensorError};
use tensor_core::tensor::{TensorOps, TensorReduce, TensorStats, TensorActivation, TensorRandom, TensorBroadcast, TensorMixedPrecision};
use super::CudaTensor;

// Implement TensorReduce
impl TensorReduce for CudaTensor {
    fn sum(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "sum reduction not implemented".to_string(),
        })
    }
    
    fn mean(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "mean reduction not implemented".to_string(),
        })
    }
    
    fn max_reduce(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "max reduction not implemented".to_string(),
        })
    }
    
    fn min_reduce(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "min reduction not implemented".to_string(),
        })
    }
    
    fn var(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "variance reduction not implemented".to_string(),
        })
    }
    
    fn std(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self> {
        let var = self.var(dims, keepdim)?;
        var.sqrt()
    }
}

// Implement TensorStats
impl TensorStats for CudaTensor {
    fn argmax(&self, dim: Option<usize>) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "argmax not implemented".to_string(),
        })
    }
    
    fn argmin(&self, dim: Option<usize>) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "argmin not implemented".to_string(),
        })
    }
    
    fn softmax(&self, dim: usize) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "softmax not implemented".to_string(),
        })
    }
    
    fn log_softmax(&self, dim: usize) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "log_softmax not implemented".to_string(),
        })
    }
    
    fn topk(&self, k: usize, dim: usize, largest: bool) -> Result<(Self, Self)> {
        Err(TensorError::UnsupportedOperation {
            op: "topk not implemented".to_string(),
        })
    }
}

// Implement TensorActivation
impl TensorActivation for CudaTensor {
    fn relu(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch ReLU kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_relu").unwrap();
            unsafe {
                kernel.launch(
                    config,
                    (
                        &self.data,
                        &mut result_data,
                        self.shape.numel() as u32,
                    ),
                )?;
            }
            
            Ok(Self {
                data: result_data,
                shape: self.shape.clone(),
                dtype: self.dtype,
                device: self.device.clone(),
                cuda_device: self.cuda_device.clone(),
                cublas: self.cublas.clone(),
            })
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            Err(TensorError::UnsupportedOperation {
                op: "CUDA operations not available without CUDA feature".to_string(),
            })
        }
    }
    
    fn gelu(&self) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "GELU not implemented".to_string(),
        })
    }
    
    fn silu(&self) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "SiLU not implemented".to_string(),
        })
    }
    
    fn sigmoid(&self) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "Sigmoid not implemented".to_string(),
        })
    }
    
    fn tanh(&self) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "Tanh not implemented".to_string(),
        })
    }
    
    fn leaky_relu(&self, negative_slope: f32) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "LeakyReLU not implemented".to_string(),
        })
    }
}

// Implement TensorRandom
impl TensorRandom for CudaTensor {
    fn random_normal(_shape: Shape, _mean: f32, _std: f32, device: &Device) -> Result<Self> {
        if !device.is_cuda() {
            return Err(TensorError::UnsupportedOperation {
                op: format!("random_normal on device {}", device),
            });
        }
        
        Err(TensorError::UnsupportedOperation {
            op: "CUDA random generation not implemented".to_string(),
        })
    }
    
    fn random_uniform(_shape: Shape, _min: f32, _max: f32, device: &Device) -> Result<Self> {
        if !device.is_cuda() {
            return Err(TensorError::UnsupportedOperation {
                op: format!("random_uniform on device {}", device),
            });
        }
        
        Err(TensorError::UnsupportedOperation {
            op: "CUDA random generation not implemented".to_string(),
        })
    }
    
    fn random_bernoulli(_shape: Shape, _p: f32, _device: &Device) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "CUDA random generation not implemented".to_string(),
        })
    }
    
    fn zeros(_shape: Shape, _dtype: DType, _device: &Device) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "CUDA tensor creation not implemented".to_string(),
        })
    }
    
    fn ones(_shape: Shape, _dtype: DType, _device: &Device) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "CUDA tensor creation not implemented".to_string(),
        })
    }
    
    fn eye(_n: usize, _dtype: DType, _device: &Device) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "CUDA tensor creation not implemented".to_string(),
        })
    }
    
    fn arange(_start: f32, _end: f32, _step: f32, _dtype: DType, _device: &Device) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "CUDA tensor creation not implemented".to_string(),
        })
    }
}

// Implement TensorBroadcast
impl TensorBroadcast for CudaTensor {
    fn broadcast_to(&self, shape: Shape) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "CUDA broadcasting not implemented".to_string(),
        })
    }
    
    fn expand(&self, shape: Shape) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "CUDA expansion not implemented".to_string(),
        })
    }
    
    fn unsqueeze(&self, dim: usize) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "CUDA unsqueeze not implemented".to_string(),
        })
    }
    
    fn squeeze(&self, dim: Option<usize>) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "CUDA squeeze not implemented".to_string(),
        })
    }
}

// Implement TensorMixedPrecision
impl TensorMixedPrecision for CudaTensor {
    fn to_f16(&self) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "FP16 conversion not implemented".to_string(),
        })
    }
    
    fn to_f32(&self) -> Result<Self> {
        Ok(self.clone())
    }
    
    fn to_f64(&self) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "FP64 conversion not implemented".to_string(),
        })
    }
    
    fn cast(&self, _dtype: DType) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "Type casting not implemented".to_string(),
        })
    }
    
    fn quantize_int8(&self, _scale: f32, _zero_point: i8) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "INT8 quantization not implemented".to_string(),
        })
    }
    
    fn dequantize_int8(&self, _scale: f32, _zero_point: i8) -> Result<Self> {
        Err(TensorError::UnsupportedOperation {
            op: "INT8 dequantization not implemented".to_string(),
        })
    }
}
