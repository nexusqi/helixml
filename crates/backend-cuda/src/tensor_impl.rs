//! CUDA Tensor trait implementations

use tensor_core::{Tensor, Shape, DType, Device, Result, TensorError};
use tensor_core::tensor::{TensorOps, TensorReduce, TensorStats, TensorActivation, TensorRandom, TensorBroadcast, TensorMixedPrecision};
use super::CudaTensor;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::cublas::CudaBlas;
#[cfg(feature = "cuda")]
use std::sync::Arc;

impl Tensor for CudaTensor {
    fn shape(&self) -> &Shape {
        &self.shape
    }
    
    fn dtype(&self) -> DType {
        self.dtype
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn is_contiguous(&self) -> bool {
        true
    }
    
    fn contiguous(&self) -> Result<Self> {
        Ok(self.clone())
    }
    
    fn reshape(&self, new_shape: Shape) -> Result<Self> {
        if new_shape.numel() != self.shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.shape.numel()],
                actual: vec![new_shape.numel()],
            });
        }
        
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                data: self.data.clone(),
                shape: new_shape,
                dtype: self.dtype,
                device: self.device.clone(),
                cuda_device: self.cuda_device.clone(),
                cublas: self.cublas.clone(),
            })
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                shape: new_shape,
                dtype: self.dtype,
                device: self.device.clone(),
            })
        }
    }
    
    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        if dim0 >= self.shape.ndim() || dim1 >= self.shape.ndim() {
            return Err(TensorError::InvalidDimension {
                dim: dim0.max(dim1),
                shape: self.shape.as_slice().to_vec(),
            });
        }
        
        if self.shape.ndim() != 2 {
            return Err(TensorError::UnsupportedOperation {
                op: "transpose only supported for 2D tensors".to_string(),
            });
        }
        
        let rows = self.shape.dim(0).unwrap();
        let cols = self.shape.dim(1).unwrap();
        
        #[cfg(feature = "cuda")]
        {
            // Create transposed tensor using CUDA kernel
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(rows * cols)?;
            
            // Launch transpose kernel
            let config = LaunchConfig::for_num_elems(rows * cols as u32);
            let kernel = self.cuda_device.get_func("transpose_2d").unwrap();
            unsafe {
                kernel.launch(
                    config,
                    (
                        &self.data,
                        &mut result_data,
                        rows as u32,
                        cols as u32,
                    ),
                )?;
            }
            
            Ok(Self {
                data: result_data,
                shape: Shape::new(vec![cols, rows]),
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
    
    fn slice(&self, start: Vec<usize>, end: Vec<usize>) -> Result<Self> {
        if start.len() != self.shape.ndim() || end.len() != self.shape.ndim() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.shape.ndim()],
                actual: vec![start.len()],
            });
        }
        
        // For now, return a simplified slice implementation
        // TODO: Implement proper CUDA slicing
        Ok(self.clone())
    }
    
    fn cat(tensors: Vec<Self>, dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TensorError::UnsupportedOperation {
                op: "Cannot concatenate empty tensor list".to_string(),
            });
        }
        
        let first_shape = tensors[0].shape.clone();
        if dim >= first_shape.ndim() {
            return Err(TensorError::InvalidDimension {
                dim,
                shape: first_shape.as_slice().to_vec(),
            });
        }
        
        // Check that all dimensions except 'dim' are the same
        for tensor in &tensors {
            if tensor.shape.ndim() != first_shape.ndim() {
                return Err(TensorError::ShapeMismatch {
                    expected: first_shape.as_slice().to_vec(),
                    actual: tensor.shape.as_slice().to_vec(),
                });
            }
            
            for i in 0..first_shape.ndim() {
                if i != dim && first_shape.dim(i).unwrap() != tensor.shape.dim(i).unwrap() {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![first_shape.dim(i).unwrap()],
                        actual: vec![tensor.shape.dim(i).unwrap()],
                    });
                }
            }
        }
        
        // Calculate new shape
        let mut new_shape = first_shape.as_slice().to_vec();
        new_shape[dim] = tensors.iter().map(|t| t.shape.dim(dim).unwrap()).sum();
        
        #[cfg(feature = "cuda")]
        {
            // Allocate new tensor
            let total_size = new_shape.iter().product::<usize>();
            let mut result_data = tensors[0].cuda_device.alloc_zeros::<f32>(total_size)?;
            
            // Copy data using CUDA kernels
            // TODO: Implement proper concatenation kernel
            
            Ok(Self {
                data: result_data,
                shape: Shape::new(new_shape),
                dtype: tensors[0].dtype,
                device: tensors[0].device.clone(),
                cuda_device: tensors[0].cuda_device.clone(),
                cublas: tensors[0].cublas.clone(),
            })
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                shape: Shape::new(new_shape),
                dtype: tensors[0].dtype,
                device: tensors[0].device.clone(),
            })
        }
    }
    
    fn stack(tensors: Vec<Self>, dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TensorError::UnsupportedOperation {
                op: "Cannot stack empty tensor list".to_string(),
            });
        }
        
        let first_shape = tensors[0].shape.clone();
        
        // Check that all tensors have the same shape
        for tensor in &tensors {
            if tensor.shape != first_shape {
                return Err(TensorError::ShapeMismatch {
                    expected: first_shape.as_slice().to_vec(),
                    actual: tensor.shape.as_slice().to_vec(),
                });
            }
        }
        
        let mut new_shape = first_shape.as_slice().to_vec();
        new_shape.insert(dim, tensors.len());
        
        #[cfg(feature = "cuda")]
        {
            let total_size = new_shape.iter().product::<usize>();
            let mut result_data = tensors[0].cuda_device.alloc_zeros::<f32>(total_size)?;
            
            // TODO: Implement proper stacking kernel
            
            Ok(Self {
                data: result_data,
                shape: Shape::new(new_shape),
                dtype: tensors[0].dtype,
                device: tensors[0].device.clone(),
                cuda_device: tensors[0].cuda_device.clone(),
                cublas: tensors[0].cublas.clone(),
            })
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                shape: Shape::new(new_shape),
                dtype: tensors[0].dtype,
                device: tensors[0].device.clone(),
            })
        }
    }
    
    fn to_device(&self, device: &Device) -> Result<Self> {
        if device.is_cpu() {
            // Copy from CUDA to CPU
            #[cfg(feature = "cuda")]
            {
                let cpu_data = self.data.to_vec()?;
                // TODO: Create CPU tensor
                Err(TensorError::UnsupportedOperation {
                    op: "CUDA to CPU transfer not implemented".to_string(),
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(TensorError::UnsupportedOperation {
                    op: "CUDA to CPU transfer not available without CUDA".to_string(),
                })
            }
        } else if device.is_cuda() {
            Ok(self.clone())
        } else {
            Err(TensorError::UnsupportedOperation {
                op: format!("Unsupported device: {}", device),
            })
        }
    }
    
    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if dtype == self.dtype {
            return Ok(self.clone());
        }
        
        // TODO: Implement dtype conversion
        Err(TensorError::UnsupportedOperation {
            op: "dtype conversion not implemented".to_string(),
        })
    }
    
    fn as_slice<T>(&self) -> Result<&[T]> {
        Err(TensorError::UnsupportedOperation {
            op: "as_slice not supported for CUDA tensors".to_string(),
        })
    }
    
    fn as_slice_mut<T>(&mut self) -> Result<&mut [T]> {
        Err(TensorError::UnsupportedOperation {
            op: "as_slice_mut not supported for CUDA tensors".to_string(),
        })
    }
}
