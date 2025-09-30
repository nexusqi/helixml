//! CUDA Tensor operations implementations

use tensor_core::{Tensor, Shape, DType, Device, Result, TensorError};
use tensor_core::tensor::{TensorOps, TensorReduce, TensorStats, TensorActivation, TensorRandom, TensorBroadcast, TensorMixedPrecision};
use super::CudaTensor;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::cublas::CudaBlas;
#[cfg(feature = "cuda")]
use std::sync::Arc;

impl TensorOps for CudaTensor {
    fn add(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().as_slice().to_vec(),
                actual: other.shape().as_slice().to_vec(),
            });
        }
        
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch element-wise addition kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_add").unwrap();
            unsafe {
                kernel.launch(
                    config,
                    (
                        &self.data,
                        &other.data,
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
    
    fn sub(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().as_slice().to_vec(),
                actual: other.shape().as_slice().to_vec(),
            });
        }
        
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch element-wise subtraction kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_sub").unwrap();
            unsafe {
                kernel.launch(
                    config,
                    (
                        &self.data,
                        &other.data,
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
    
    fn mul(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().as_slice().to_vec(),
                actual: other.shape().as_slice().to_vec(),
            });
        }
        
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch element-wise multiplication kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_mul").unwrap();
            unsafe {
                kernel.launch(
                    config,
                    (
                        &self.data,
                        &other.data,
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
    
    fn div(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().as_slice().to_vec(),
                actual: other.shape().as_slice().to_vec(),
            });
        }
        
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch element-wise division kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_div").unwrap();
            unsafe {
                kernel.launch(
                    config,
                    (
                        &self.data,
                        &other.data,
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
    
    fn matmul(&self, other: &Self) -> Result<Self> {
        if self.shape().ndim() == 0 || other.shape().ndim() == 0 {
            return Err(TensorError::UnsupportedOperation {
                op: "matrix multiplication requires at least 1D tensors".to_string(),
            });
        }
        
        match (self.shape().ndim(), other.shape().ndim()) {
            // 2D x 2D: Use cuBLAS for optimized matrix multiplication
            (2, 2) => {
                let a_rows = self.shape().dim(0).unwrap();
                let a_cols = self.shape().dim(1).unwrap();
                let b_rows = other.shape().dim(0).unwrap();
                let b_cols = other.shape().dim(1).unwrap();
                
                if a_cols != b_rows {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![a_cols],
                        actual: vec![b_rows],
                    });
                }
                
                #[cfg(feature = "cuda")]
                {
                    let mut result_data = self.cuda_device.alloc_zeros::<f32>(a_rows * b_cols)?;
                    
                    // Use cuBLAS for matrix multiplication
                    self.cublas.gemm(
                        false, // transpose A
                        false, // transpose B
                        a_rows as i32,
                        b_cols as i32,
                        a_cols as i32,
                        1.0,
                        &self.data,
                        a_rows as i32,
                        &other.data,
                        a_cols as i32,
                        0.0,
                        &mut result_data,
                        a_rows as i32,
                    )?;
                    
                    Ok(Self {
                        data: result_data,
                        shape: Shape::new(vec![a_rows, b_cols]),
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
            
            // Other cases: TODO implement
            _ => Err(TensorError::UnsupportedOperation {
                op: "matrix multiplication for this tensor shape not implemented".to_string(),
            })
        }
    }
    
    fn pow(&self, exponent: f32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch power kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_pow").unwrap();
            unsafe {
                kernel.launch(
                    config,
                    (
                        &self.data,
                        &mut result_data,
                        exponent,
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
    
    fn sqrt(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch sqrt kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_sqrt").unwrap();
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
    
    fn exp(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch exp kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_exp").unwrap();
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
    
    fn log(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch log kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_log").unwrap();
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
    
    fn sin(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch sin kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_sin").unwrap();
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
    
    fn cos(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch cos kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_cos").unwrap();
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
    
    fn tan(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch tan kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_tan").unwrap();
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
    
    fn abs(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch abs kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_abs").unwrap();
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
    
    fn sign(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch sign kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_sign").unwrap();
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
    
    fn max(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().as_slice().to_vec(),
                actual: other.shape().as_slice().to_vec(),
            });
        }
        
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch max kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_max").unwrap();
            unsafe {
                kernel.launch(
                    config,
                    (
                        &self.data,
                        &other.data,
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
    
    fn min(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().as_slice().to_vec(),
                actual: other.shape().as_slice().to_vec(),
            });
        }
        
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch min kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_min").unwrap();
            unsafe {
                kernel.launch(
                    config,
                    (
                        &self.data,
                        &other.data,
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
    
    fn clamp(&self, min: f32, max: f32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut result_data = self.cuda_device.alloc_zeros::<f32>(self.shape.numel())?;
            
            // Launch clamp kernel
            let config = LaunchConfig::for_num_elems(self.shape.numel() as u32);
            let kernel = self.cuda_device.get_func("elementwise_clamp").unwrap();
            unsafe {
                kernel.launch(
                    config,
                    (
                        &self.data,
                        &mut result_data,
                        min,
                        max,
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
}
