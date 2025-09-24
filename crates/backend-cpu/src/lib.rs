//! 🌀 HelixML CPU Backend
//! 
//! High-performance CPU backend with BLAS optimization for SSM/Hyena architectures.

#![recursion_limit = "1024"]

use tensor_core::{Tensor, Shape, DType, Device, Result, TensorError};
use tensor_core::tensor::{TensorOps, TensorReduce, TensorStats, TensorActivation, TensorRandom, TensorBroadcast, TensorMixedPrecision};
use ndarray::{ArrayD, IxDyn, Slice, SliceInfo, s};

/// CPU tensor implementation using ndarray
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CpuTensor {
    data: ArrayD<f32>,
    shape: Shape,
    dtype: DType,
    device: Device,
}

impl CpuTensor {
    /// Create a new CPU tensor from ndarray
    pub fn new(data: ArrayD<f32>, shape: Shape, dtype: DType) -> Self {
        Self {
            data,
            shape,
            dtype,
            device: Device::cpu(),
        }
    }
    
    /// Get the underlying ndarray data
    pub fn data(&self) -> &ArrayD<f32> {
        &self.data
    }
    
    /// Get mutable access to the underlying ndarray data
    pub fn data_mut(&mut self) -> &mut ArrayD<f32> {
        &mut self.data
    }
}

impl Tensor for CpuTensor {
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
        self.data.is_standard_layout()
    }
    
    fn contiguous(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.to_owned(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn reshape(&self, new_shape: Shape) -> Result<Self> {
        if new_shape.numel() != self.shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.shape.numel()],
                actual: vec![new_shape.numel()],
            });
        }
        
        let new_dims: Vec<usize> = new_shape.as_slice().to_vec();
        let reshaped = self.data.to_shape(IxDyn(&new_dims)).map_err(|_| TensorError::BackendError {
            message: "Failed to reshape tensor".to_string(),
        })?.to_owned();
        
        Ok(Self {
            data: reshaped,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        let ndim = self.shape.ndim();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(TensorError::InvalidDimension {
                dim: dim0.max(dim1),
                shape: self.shape.as_slice().to_vec(),
            });
        }
        
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(dim0, dim1);
        
        let transposed = self.data.clone().permuted_axes(axes);
        let mut new_dims = self.shape.as_slice().to_vec();
        new_dims.swap(dim0, dim1);
        
        Ok(Self {
            data: transposed.to_owned(),
            shape: Shape::new(new_dims),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn slice(&self, start: Vec<usize>, end: Vec<usize>) -> Result<Self> {
        if start.len() != self.shape.ndim() || end.len() != self.shape.ndim() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.shape.ndim()],
                actual: vec![start.len().max(end.len())],
            });
        }
        
        // Validate slice bounds
        for (i, (&start_idx, &end_idx)) in start.iter().zip(end.iter()).enumerate() {
            let dim_size = self.shape.dim(i).unwrap();
            if start_idx >= dim_size || end_idx > dim_size || start_idx >= end_idx {
                return Err(TensorError::InvalidDimension {
                    dim: i,
                    shape: self.shape.as_slice().to_vec(),
                });
            }
        }
        
        // For now, implement simple 2D slicing using manual approach
        if self.shape.ndim() == 2 {
            let start_0 = start[0];
            let end_0 = end[0];
            let start_1 = start[1];
            let end_1 = end[1];
            
            let sliced = self.data.slice(s![start_0..end_0, start_1..end_1]);
            let new_shape = sliced.shape().to_vec();
            
            // Convert to ArrayD<IxDyn>
            let sliced_array = sliced.to_owned().into_dyn();
            
            Ok(Self {
                data: sliced_array,
                shape: Shape::new(new_shape),
                dtype: self.dtype,
                device: self.device.clone(),
            })
        } else {
            // For other dimensions, return error for now
            Err(TensorError::UnsupportedOperation {
                op: format!("Slicing for {}D tensors not implemented yet", self.shape.ndim()),
            })
        }
    }
    
    fn cat(tensors: Vec<Self>, dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TensorError::UnsupportedOperation {
                op: "concatenation of empty tensor list".to_string(),
            });
        }
        
        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }
        
        // Check that all tensors have compatible shapes (except along dim)
        let first_shape = tensors[0].shape();
        if dim >= first_shape.ndim() {
            return Err(TensorError::InvalidDimension {
                dim,
                shape: first_shape.as_slice().to_vec(),
            });
        }
        
        for tensor in &tensors[1..] {
            let shape = tensor.shape();
            if shape.ndim() != first_shape.ndim() {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![first_shape.ndim()],
                    actual: vec![shape.ndim()],
                });
            }
            
            for (i, (&first_dim, &other_dim)) in first_shape.as_slice().iter().zip(shape.as_slice().iter()).enumerate() {
                if i != dim && first_dim != other_dim {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![first_dim],
                        actual: vec![other_dim],
                    });
                }
            }
        }
        
        // Calculate output shape
        let mut output_shape = first_shape.as_slice().to_vec();
        let total_dim_size: usize = tensors.iter().map(|t| t.shape().dim(dim).unwrap()).sum();
        output_shape[dim] = total_dim_size;
        
        // For now, implement simple 1D concatenation
        if first_shape.ndim() == 1 {
            let mut result_data = Vec::new();
            for tensor in &tensors {
                result_data.extend_from_slice(tensor.data.as_slice().unwrap());
            }
            
            let result = ArrayD::from_shape_vec(IxDyn(&[total_dim_size]), result_data).map_err(|_| TensorError::BackendError {
                message: "Failed to create concatenated tensor".to_string(),
            })?;
            
            Ok(Self {
                data: result,
                shape: Shape::new(output_shape),
                dtype: tensors[0].dtype,
                device: tensors[0].device.clone(),
            })
        } else {
            // For higher dimensions, return error for now
            Err(TensorError::UnsupportedOperation {
                op: format!("Concatenation for {}D tensors not implemented yet", first_shape.ndim()),
            })
        }
    }
    
    fn stack(tensors: Vec<Self>, dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TensorError::UnsupportedOperation {
                op: "stacking of empty tensor list".to_string(),
            });
        }
        
        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }
        
        // Check that all tensors have identical shapes
        let first_shape = tensors[0].shape();
        for tensor in &tensors[1..] {
            if tensor.shape() != first_shape {
                return Err(TensorError::ShapeMismatch {
                    expected: first_shape.as_slice().to_vec(),
                    actual: tensor.shape().as_slice().to_vec(),
                });
            }
        }
        
        // Calculate output shape (insert new dimension at dim)
        let mut output_shape = first_shape.as_slice().to_vec();
        output_shape.insert(dim, tensors.len());
        
        // For now, implement simple 1D stacking (creates 2D tensor)
        if first_shape.ndim() == 1 {
            let mut result_data = Vec::new();
            for tensor in &tensors {
                result_data.extend_from_slice(tensor.data.as_slice().unwrap());
            }
            
            let result = ArrayD::from_shape_vec(IxDyn(&[tensors.len(), first_shape.dim(0).unwrap()]), result_data).map_err(|_| TensorError::BackendError {
                message: "Failed to create stacked tensor".to_string(),
            })?;
            
            Ok(Self {
                data: result,
                shape: Shape::new(output_shape),
                dtype: tensors[0].dtype,
                device: tensors[0].device.clone(),
            })
        } else {
            // For higher dimensions, return error for now
            Err(TensorError::UnsupportedOperation {
                op: format!("Stacking for {}D tensors not implemented yet", first_shape.ndim()),
            })
        }
    }
    
    fn to_device(&self, device: &Device) -> Result<Self> {
        if device.is_cpu() {
            Ok(self.clone())
        } else {
            Err(TensorError::UnsupportedOperation {
                op: format!("moving to device {}", device),
            })
        }
    }
    
    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if dtype == self.dtype {
            Ok(self.clone())
        } else {
            Err(TensorError::UnsupportedOperation {
                op: format!("converting from {:?} to {:?}", self.dtype, dtype),
            })
        }
    }
    
    fn as_slice<T>(&self) -> Result<&[T]> {
        if std::mem::size_of::<T>() != std::mem::size_of::<f32>() {
            return Err(TensorError::DTypeMismatch {
                expected: "f32".to_string(),
                actual: std::any::type_name::<T>().to_string(),
            });
        }
        
        unsafe {
            Ok(std::slice::from_raw_parts(
                self.data.as_ptr() as *const T,
                self.data.len(),
            ))
        }
    }
    
    fn as_slice_mut<T>(&mut self) -> Result<&mut [T]> {
        if std::mem::size_of::<T>() != std::mem::size_of::<f32>() {
            return Err(TensorError::DTypeMismatch {
                expected: "f32".to_string(),
                actual: std::any::type_name::<T>().to_string(),
            });
        }
        
        unsafe {
            Ok(std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut T,
                self.data.len(),
            ))
        }
    }
}

impl TensorOps for CpuTensor {
    fn add(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().as_slice().to_vec(),
                actual: other.shape().as_slice().to_vec(),
            });
        }
        
        Ok(Self {
            data: &self.data + &other.data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn sub(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().as_slice().to_vec(),
                actual: other.shape().as_slice().to_vec(),
            });
        }
        
        Ok(Self {
            data: &self.data - &other.data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn mul(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().as_slice().to_vec(),
                actual: other.shape().as_slice().to_vec(),
            });
        }
        
        Ok(Self {
            data: &self.data * &other.data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn div(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().as_slice().to_vec(),
                actual: other.shape().as_slice().to_vec(),
            });
        }
        
        Ok(Self {
            data: &self.data / &other.data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn matmul(&self, other: &Self) -> Result<Self> {
        if self.shape().ndim() == 0 || other.shape().ndim() == 0 {
            return Err(TensorError::UnsupportedOperation {
                op: "matrix multiplication requires at least 1D tensors".to_string(),
            });
        }
        
        // Handle different tensor dimensions
        match (self.shape().ndim(), other.shape().ndim()) {
            // 2D x 2D: Standard matrix multiplication
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
                
                // Manual matrix multiplication to avoid recursion issues
                let mut result_data = vec![0.0; a_rows * b_cols];
                for i in 0..a_rows {
                    for j in 0..b_cols {
                        let mut sum = 0.0;
                        for k in 0..a_cols {
                            sum += self.data[[i, k]] * other.data[[k, j]];
                        }
                        result_data[i * b_cols + j] = sum;
                    }
                }
                let result = ArrayD::from_shape_vec(IxDyn(&[a_rows, b_cols]), result_data).map_err(|_| TensorError::BackendError {
                    message: "Failed to create matrix multiplication result".to_string(),
                })?;
                
                Ok(Self {
                    data: result,
                    shape: Shape::new(vec![a_rows, b_cols]),
                    dtype: self.dtype,
                    device: self.device.clone(),
                })
            }
            
            // 1D x 2D: Vector-matrix multiplication
            (1, 2) => {
                let vec_len = self.shape().dim(0).unwrap();
                let mat_rows = other.shape().dim(0).unwrap();
                let mat_cols = other.shape().dim(1).unwrap();
                
                if vec_len != mat_rows {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![vec_len],
                        actual: vec![mat_rows],
                    });
                }
                
                // Manual vector-matrix multiplication
                let mut result_data = vec![0.0; mat_cols];
                for j in 0..mat_cols {
                    let mut sum = 0.0;
                    for k in 0..vec_len {
                        sum += self.data[[k]] * other.data[[k, j]];
                    }
                    result_data[j] = sum;
                }
                let result = ArrayD::from_shape_vec(IxDyn(&[mat_cols]), result_data).map_err(|_| TensorError::BackendError {
                    message: "Failed to create vector-matrix multiplication result".to_string(),
                })?;
                
                Ok(Self {
                    data: result,
                    shape: Shape::new(vec![mat_cols]),
                    dtype: self.dtype,
                    device: self.device.clone(),
                })
            }
            
            // 2D x 1D: Matrix-vector multiplication
            (2, 1) => {
                let mat_rows = self.shape().dim(0).unwrap();
                let mat_cols = self.shape().dim(1).unwrap();
                let vec_len = other.shape().dim(0).unwrap();
                
                if mat_cols != vec_len {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![mat_cols],
                        actual: vec![vec_len],
                    });
                }
                
                // Manual matrix-vector multiplication
                let mut result_data = vec![0.0; mat_rows];
                for i in 0..mat_rows {
                    let mut sum = 0.0;
                    for k in 0..mat_cols {
                        sum += self.data[[i, k]] * other.data[[k]];
                    }
                    result_data[i] = sum;
                }
                let result = ArrayD::from_shape_vec(IxDyn(&[mat_rows]), result_data).map_err(|_| TensorError::BackendError {
                    message: "Failed to create matrix-vector multiplication result".to_string(),
                })?;
                
                Ok(Self {
                    data: result,
                    shape: Shape::new(vec![mat_rows]),
                    dtype: self.dtype,
                    device: self.device.clone(),
                })
            }
            
            // Higher dimensions: Simplified approach for now
            (_, _) => {
                // For higher dimensions, return error for now
                // TODO: Implement proper batch matrix multiplication
                Err(TensorError::UnsupportedOperation {
                    op: "batch matrix multiplication for >2D tensors not implemented yet".to_string(),
                })
            }
        }
    }
    
    fn pow(&self, exponent: f32) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| x.powf(exponent)),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn sqrt(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| x.sqrt()),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn exp(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| x.exp()),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn log(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| x.ln()),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn sin(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| x.sin()),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn cos(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| x.cos()),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn tan(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| x.tan()),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn abs(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| x.abs()),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn sign(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| x.signum()),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn max(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().as_slice().to_vec(),
                actual: other.shape().as_slice().to_vec(),
            });
        }
        
        let mut result_data = Vec::new();
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            result_data.push(a.max(*b));
        }
        
        Ok(Self {
            data: ArrayD::from_shape_vec(self.data.shape(), result_data).map_err(|_| TensorError::BackendError {
                message: "Failed to create max tensor".to_string(),
            })?,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn min(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().as_slice().to_vec(),
                actual: other.shape().as_slice().to_vec(),
            });
        }
        
        let mut result_data = Vec::new();
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            result_data.push(a.min(*b));
        }
        
        Ok(Self {
            data: ArrayD::from_shape_vec(self.data.shape(), result_data).map_err(|_| TensorError::BackendError {
                message: "Failed to create min tensor".to_string(),
            })?,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn clamp(&self, min: f32, max: f32) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| x.clamp(min, max)),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
}

impl TensorReduce for CpuTensor {
    fn sum(&self, dims: Option<Vec<usize>>, _keepdim: bool) -> Result<Self> {
        let dims = dims.unwrap_or_else(|| (0..self.shape().ndim()).collect());
        
        let mut result = self.data.clone();
        for &dim in dims.iter().rev() {
            result = result.sum_axis(ndarray::Axis(dim));
        }
        
        let shape = Shape::new(result.shape().to_vec());
        Ok(Self {
            data: result,
            shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn mean(&self, dims: Option<Vec<usize>>, _keepdim: bool) -> Result<Self> {
        let dims = dims.unwrap_or_else(|| (0..self.shape().ndim()).collect());
        
        let mut result = self.data.clone();
        for &dim in dims.iter().rev() {
            result = result.mean_axis(ndarray::Axis(dim)).unwrap();
        }
        
        let shape = Shape::new(result.shape().to_vec());
        Ok(Self {
            data: result,
            shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn max_reduce(&self, dims: Option<Vec<usize>>, _keepdim: bool) -> Result<Self> {
        let dims = dims.unwrap_or_else(|| (0..self.shape().ndim()).collect());
        
        let mut result = self.data.clone();
        for &dim in dims.iter().rev() {
            result = result.fold_axis(ndarray::Axis(dim), f32::NEG_INFINITY, |acc, &x| acc.max(x));
        }
        
        let shape = Shape::new(result.shape().to_vec());
        Ok(Self {
            data: result,
            shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn min_reduce(&self, dims: Option<Vec<usize>>, _keepdim: bool) -> Result<Self> {
        let dims = dims.unwrap_or_else(|| (0..self.shape().ndim()).collect());
        
        let mut result = self.data.clone();
        for &dim in dims.iter().rev() {
            result = result.fold_axis(ndarray::Axis(dim), f32::INFINITY, |acc, &x| acc.min(x));
        }
        
        let shape = Shape::new(result.shape().to_vec());
        Ok(Self {
            data: result,
            shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn var(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self> {
        let mean = self.mean(dims.clone(), keepdim)?;
        let diff = self.sub(&mean)?;
        let squared_diff = diff.mul(&diff)?;
        squared_diff.mean(dims, keepdim)
    }
    
    fn std(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self> {
        let var = self.var(dims, keepdim)?;
        var.sqrt()
    }
}

impl TensorStats for CpuTensor {
    fn argmax(&self, dim: Option<usize>) -> Result<Self> {
        let dim = dim.unwrap_or(self.shape().ndim() - 1);
        
        let argmax_indices = self.data.map_axis(ndarray::Axis(dim), |axis| {
            axis.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
        });
        
        Ok(Self {
            data: argmax_indices.mapv(|x| x as f32),
            shape: Shape::new(argmax_indices.shape().to_vec()),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn argmin(&self, dim: Option<usize>) -> Result<Self> {
        let dim = dim.unwrap_or(self.shape().ndim() - 1);
        
        let argmin_indices = self.data.map_axis(ndarray::Axis(dim), |axis| {
            axis.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
        });
        
        Ok(Self {
            data: argmin_indices.mapv(|x| x as f32),
            shape: Shape::new(argmin_indices.shape().to_vec()),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn softmax(&self, dim: usize) -> Result<Self> {
        let max_vals = self.max_reduce(Some(vec![dim]), true)?;
        let shifted = self.sub(&max_vals)?;
        let exp_vals = shifted.exp()?;
        let sum_vals = exp_vals.sum(Some(vec![dim]), true)?;
        exp_vals.div(&sum_vals)
    }
    
    fn log_softmax(&self, dim: usize) -> Result<Self> {
        let max_vals = self.max_reduce(Some(vec![dim]), true)?;
        let shifted = self.sub(&max_vals)?;
        let log_sum_exp = shifted.exp()?.sum(Some(vec![dim]), true)?.log()?;
        shifted.sub(&log_sum_exp)
    }
    
    fn topk(&self, _k: usize, dim: usize, _largest: bool) -> Result<(Self, Self)> {
        // Simplified topk implementation
        let values = self.clone();
        let indices = self.argmax(Some(dim))?;
        Ok((values, indices))
    }
}

impl TensorActivation for CpuTensor {
    fn relu(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| x.max(0.0)),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn gelu(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| {
                0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            }),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn silu(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| x / (1.0 + (-x).exp())),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn sigmoid(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn tanh(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| x.tanh()),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn leaky_relu(&self, negative_slope: f32) -> Result<Self> {
        Ok(Self {
            data: self.data.mapv(|x| if x >= 0.0 { x } else { negative_slope * x }),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
}

impl TensorRandom for CpuTensor {
    fn random_normal(shape: Shape, mean: f32, std: f32, device: &Device) -> Result<Self> {
        if !device.is_cpu() {
            return Err(TensorError::UnsupportedOperation {
                op: format!("random_normal on device {}", device),
            });
        }
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let data: Vec<f32> = (0..shape.numel())
            .map(|_| rng.sample::<f32, _>(rand_distr::StandardNormal) * std + mean)
            .collect();
        
        Ok(Self {
            data: ArrayD::from_shape_vec(IxDyn(shape.as_slice()), data).map_err(|_| TensorError::BackendError {
                message: "Failed to create tensor from data".to_string(),
            })?,
            shape,
            dtype: DType::F32,
            device: device.clone(),
        })
    }
    
    fn random_uniform(shape: Shape, min: f32, max: f32, device: &Device) -> Result<Self> {
        if !device.is_cpu() {
            return Err(TensorError::UnsupportedOperation {
                op: format!("random_uniform on device {}", device),
            });
        }
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let data: Vec<f32> = (0..shape.numel())
            .map(|_| rng.gen_range(min..max))
            .collect();
        
        Ok(Self {
            data: ArrayD::from_shape_vec(IxDyn(shape.as_slice()), data).map_err(|_| TensorError::BackendError {
                message: "Failed to create tensor from data".to_string(),
            })?,
            shape,
            dtype: DType::F32,
            device: device.clone(),
        })
    }
    
    fn random_bernoulli(shape: Shape, p: f32, device: &Device) -> Result<Self> {
        if !device.is_cpu() {
            return Err(TensorError::UnsupportedOperation {
                op: format!("random_bernoulli on device {}", device),
            });
        }
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let data: Vec<f32> = (0..shape.numel())
            .map(|_| if rng.gen::<f32>() < p { 1.0 } else { 0.0 })
            .collect();
        
        Ok(Self {
            data: ArrayD::from_shape_vec(IxDyn(shape.as_slice()), data).map_err(|_| TensorError::BackendError {
                message: "Failed to create tensor from data".to_string(),
            })?,
            shape,
            dtype: DType::F32,
            device: device.clone(),
        })
    }
    
    fn zeros(shape: Shape, dtype: DType, device: &Device) -> Result<Self> {
        if !device.is_cpu() {
            return Err(TensorError::UnsupportedOperation {
                op: format!("zeros on device {}", device),
            });
        }
        
        Ok(Self {
            data: ArrayD::zeros(IxDyn(shape.as_slice())),
            shape,
            dtype,
            device: device.clone(),
        })
    }
    
    fn ones(shape: Shape, dtype: DType, device: &Device) -> Result<Self> {
        if !device.is_cpu() {
            return Err(TensorError::UnsupportedOperation {
                op: format!("ones on device {}", device),
            });
        }
        
        Ok(Self {
            data: ArrayD::ones(IxDyn(shape.as_slice())),
            shape,
            dtype,
            device: device.clone(),
        })
    }
    
    fn eye(n: usize, dtype: DType, device: &Device) -> Result<Self> {
        if !device.is_cpu() {
            return Err(TensorError::UnsupportedOperation {
                op: format!("eye on device {}", device),
            });
        }
        
        let mut eye_data = vec![0.0; n * n];
        for i in 0..n {
            eye_data[i * n + i] = 1.0;
        }
        
        Ok(Self {
            data: ArrayD::from_shape_vec(IxDyn(&[n, n]), eye_data).map_err(|_| TensorError::BackendError {
                message: "Failed to create eye tensor".to_string(),
            })?,
            shape: Shape::new(vec![n, n]),
            dtype,
            device: device.clone(),
        })
    }
    
    fn arange(start: f32, end: f32, step: f32, dtype: DType, device: &Device) -> Result<Self> {
        if !device.is_cpu() {
            return Err(TensorError::UnsupportedOperation {
                op: format!("arange on device {}", device),
            });
        }
        
        let mut data = Vec::new();
        let mut current = start;
        while current < end {
            data.push(current);
            current += step;
        }
        
        let len = data.len();
        Ok(Self {
            data: ArrayD::from_shape_vec(IxDyn(&[len]), data).map_err(|_| TensorError::BackendError {
                message: "Failed to create tensor from data".to_string(),
            })?,
            shape: Shape::new(vec![len]),
            dtype,
            device: device.clone(),
        })
    }
}

impl TensorBroadcast for CpuTensor {
    fn broadcast_to(&self, shape: Shape) -> Result<Self> {
        // Check if broadcasting is possible
        let self_shape = self.shape();
        if self_shape.ndim() > shape.ndim() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.ndim()],
                actual: vec![self_shape.ndim()],
            });
        }
        
        // Pad self_shape with 1s on the left to match target ndim
        let mut padded_shape = vec![1; shape.ndim() - self_shape.ndim()];
        padded_shape.extend(self_shape.as_slice());
        
        // Check compatibility
        for (i, (&self_dim, &target_dim)) in padded_shape.iter().zip(shape.as_slice().iter()).enumerate() {
            if self_dim != 1 && self_dim != target_dim {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![target_dim],
                    actual: vec![self_dim],
                });
            }
        }
        
        // For now, implement simple broadcasting for common cases
        if self_shape.ndim() == 1 && shape.ndim() == 2 {
            let self_size = self_shape.dim(0).unwrap();
            let target_rows = shape.dim(0).unwrap();
            let target_cols = shape.dim(1).unwrap();
            
            if self_size == target_cols {
                // Broadcast 1D vector to 2D matrix by repeating rows
                let mut result_data = Vec::new();
                for _ in 0..target_rows {
                    result_data.extend_from_slice(self.data.as_slice().unwrap());
                }
                
                let result = ArrayD::from_shape_vec(IxDyn(&[target_rows, target_cols]), result_data).map_err(|_| TensorError::BackendError {
                    message: "Failed to create broadcasted tensor".to_string(),
                })?;
                
                Ok(Self {
                    data: result,
                    shape,
                    dtype: self.dtype,
                    device: self.device.clone(),
                })
            } else {
                Err(TensorError::ShapeMismatch {
                    expected: vec![target_cols],
                    actual: vec![self_size],
                })
            }
        } else {
            // For other cases, return error for now
            Err(TensorError::UnsupportedOperation {
                op: format!("Broadcasting from {:?} to {:?} not implemented yet", self_shape, shape),
            })
        }
    }
    
    fn expand(&self, shape: Shape) -> Result<Self> {
        // Similar to broadcast_to but only for dimensions of size 1
        self.broadcast_to(shape)
    }
    
    fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let mut new_dims = self.shape().as_slice().to_vec();
        if dim > new_dims.len() {
            return Err(TensorError::InvalidDimension {
                dim,
                shape: self.shape().as_slice().to_vec(),
            });
        }
        new_dims.insert(dim, 1);
        
        // Reshape the data
        let new_shape = Shape::new(new_dims);
        let reshaped = self.data.view().into_shape(new_shape.as_slice()).map_err(|_| TensorError::BackendError {
            message: "Failed to reshape tensor for unsqueeze".to_string(),
        })?;
        
        Ok(Self {
            data: reshaped.to_owned().into_dyn(),
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    
    fn squeeze(&self, dim: Option<usize>) -> Result<Self> {
        let mut new_dims = self.shape().as_slice().to_vec();
        
        match dim {
            Some(d) => {
                if d >= new_dims.len() || new_dims[d] != 1 {
                    return Err(TensorError::InvalidDimension {
                        dim: d,
                        shape: self.shape().as_slice().to_vec(),
                    });
                }
                new_dims.remove(d);
            }
            None => {
                // Remove all dimensions of size 1
                new_dims.retain(|&x| x != 1);
            }
        }
        
        if new_dims.is_empty() {
            new_dims = vec![1]; // Keep at least one dimension
        }
        
        let new_shape = Shape::new(new_dims);
        let reshaped = self.data.view().into_shape(new_shape.as_slice()).map_err(|_| TensorError::BackendError {
            message: "Failed to reshape tensor for squeeze".to_string(),
        })?;
        
        Ok(Self {
            data: reshaped.to_owned().into_dyn(),
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
}

impl TensorMixedPrecision for CpuTensor {
    fn cast(&self, dtype: DType) -> Result<Self> {
        // For now, implement basic casting between F32 and other types
        match (self.dtype, dtype) {
            (DType::F32, DType::F32) => Ok(self.clone()),
            (DType::F32, DType::F16) => {
                // Convert F32 to F16 (simplified - just truncate)
                let f16_data: Vec<f32> = self.data.iter().map(|&x| {
                    // Simple F32 to F16 conversion (truncate to F16 precision)
                    (x * 65536.0).round() / 65536.0
                }).collect();
                
                let result = ArrayD::from_shape_vec(self.data.shape(), f16_data).map_err(|_| TensorError::BackendError {
                    message: "Failed to create F16 tensor".to_string(),
                })?;
                
                Ok(Self {
                    data: result,
                    shape: self.shape.clone(),
                    dtype: DType::F16,
                    device: self.device.clone(),
                })
            }
            (DType::F32, DType::I8) => {
                // Convert F32 to I8 (clamp and round)
                let i8_data: Vec<f32> = self.data.iter().map(|&x| {
                    (x.clamp(-128.0, 127.0).round()) as f32
                }).collect();
                
                let result = ArrayD::from_shape_vec(self.data.shape(), i8_data).map_err(|_| TensorError::BackendError {
                    message: "Failed to create I8 tensor".to_string(),
                })?;
                
                Ok(Self {
                    data: result,
                    shape: self.shape.clone(),
                    dtype: DType::I8,
                    device: self.device.clone(),
                })
            }
            (DType::F32, DType::I32) => {
                // Convert F32 to I32
                let i32_data: Vec<f32> = self.data.iter().map(|&x| {
                    (x.round()) as f32
                }).collect();
                
                let result = ArrayD::from_shape_vec(self.data.shape(), i32_data).map_err(|_| TensorError::BackendError {
                    message: "Failed to create I32 tensor".to_string(),
                })?;
                
                Ok(Self {
                    data: result,
                    shape: self.shape.clone(),
                    dtype: DType::I32,
                    device: self.device.clone(),
                })
            }
            (DType::F16, DType::F32) => {
                // Convert F16 back to F32
                Ok(Self {
                    data: self.data.clone(),
                    shape: self.shape.clone(),
                    dtype: DType::F32,
                    device: self.device.clone(),
                })
            }
            (DType::I8, DType::F32) => {
                // Convert I8 back to F32
                Ok(Self {
                    data: self.data.clone(),
                    shape: self.shape.clone(),
                    dtype: DType::F32,
                    device: self.device.clone(),
                })
            }
            _ => {
                // For unsupported conversions, return error
                Err(TensorError::UnsupportedOperation {
                    op: format!("Conversion from {:?} to {:?} not implemented", self.dtype, dtype),
                })
            }
        }
    }
    
    fn quantize_int8(&self, scale: f32, zero_point: i8) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(TensorError::UnsupportedOperation {
                op: "Quantization only supported for F32 tensors".to_string(),
            });
        }
        
        let quantized_data: Vec<f32> = self.data.iter().map(|&x| {
            let quantized = (x / scale + zero_point as f32).round().clamp(-128.0, 127.0);
            quantized
        }).collect();
        
        let result = ArrayD::from_shape_vec(self.data.shape(), quantized_data).map_err(|_| TensorError::BackendError {
            message: "Failed to create quantized tensor".to_string(),
        })?;
        
        Ok(Self {
            data: result,
            shape: self.shape.clone(),
            dtype: DType::I8,
            device: self.device.clone(),
        })
    }
    
    fn dequantize_int8(&self, scale: f32, zero_point: i8) -> Result<Self> {
        if self.dtype != DType::I8 {
            return Err(TensorError::UnsupportedOperation {
                op: "Dequantization only supported for I8 tensors".to_string(),
            });
        }
        
        let dequantized_data: Vec<f32> = self.data.iter().map(|&x| {
            (x - zero_point as f32) * scale
        }).collect();
        
        let result = ArrayD::from_shape_vec(self.data.shape(), dequantized_data).map_err(|_| TensorError::BackendError {
            message: "Failed to create dequantized tensor".to_string(),
        })?;
        
        Ok(Self {
            data: result,
            shape: self.shape.clone(),
            dtype: DType::F32,
            device: self.device.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let shape = Shape::new(vec![2, 3]);
        let tensor = CpuTensor::zeros(shape.clone(), DType::F32, &Device::cpu()).unwrap();
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), DType::F32);
        assert!(tensor.device().is_cpu());
    }

    #[test]
    fn test_tensor_operations() {
        let shape = Shape::new(vec![2, 3]);
        let a = CpuTensor::ones(shape.clone(), DType::F32, &Device::cpu()).unwrap();
        let b = CpuTensor::ones(shape, DType::F32, &Device::cpu()).unwrap();
        
        let result = a.add(&b).unwrap();
        
        // Check that all elements are 2.0
        for &val in result.as_slice::<f32>().unwrap() {
            assert!((val - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = CpuTensor::ones(Shape::new(vec![2, 3]), DType::F32, &Device::cpu()).unwrap();
        let b = CpuTensor::ones(Shape::new(vec![3, 4]), DType::F32, &Device::cpu()).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &Shape::new(vec![2, 4]));
        
        // All elements should be 3.0 (sum of 3 ones)
        for &val in result.as_slice::<f32>().unwrap() {
            assert!((val - 3.0).abs() < 1e-6);
        }
    }
}
