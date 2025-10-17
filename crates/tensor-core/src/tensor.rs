//! Core tensor trait and operations

use crate::{DType, Device, Shape, Result, TensorError};
use serde::{Deserialize, Serialize};

/// Core tensor trait for all tensor implementations
pub trait Tensor: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> {
    /// Get the shape of the tensor
    fn shape(&self) -> &Shape;
    
    /// Get the data type of the tensor
    fn dtype(&self) -> DType;
    
    /// Get the device of the tensor
    fn device(&self) -> &Device;
    
    /// Get the number of elements
    fn numel(&self) -> usize {
        self.shape().numel()
    }
    
    /// Get the number of dimensions
    fn ndim(&self) -> usize {
        self.shape().ndim()
    }
    
    /// Check if tensor is contiguous in memory
    fn is_contiguous(&self) -> bool;
    
    /// Create a contiguous copy of the tensor
    fn contiguous(&self) -> Result<Self>;
    
    /// Reshape the tensor to new shape
    fn reshape(&self, new_shape: Shape) -> Result<Self>;
    
    /// Transpose the tensor
    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self>;
    
    /// Get a slice of the tensor
    fn slice(&self, start: Vec<usize>, end: Vec<usize>) -> Result<Self>;
    
    /// Concatenate tensors along a dimension
    fn cat(tensors: Vec<Self>, dim: usize) -> Result<Self>;
    
    /// Stack tensors along a new dimension
    fn stack(tensors: Vec<Self>, dim: usize) -> Result<Self>;
    
    /// Move tensor to a different device
    fn to_device(&self, device: &Device) -> Result<Self>;
    
    /// Convert tensor to different data type
    fn to_dtype(&self, dtype: DType) -> Result<Self>;
    
    /// Get tensor data as a slice (for CPU tensors)
    fn as_slice<T>(&self) -> Result<&[T]>;
    
    /// Get mutable tensor data as a slice (for CPU tensors)
    fn as_slice_mut<T>(&mut self) -> Result<&mut [T]>;
}

/// Basic tensor operations
pub trait TensorOps: Tensor {
    /// Element-wise addition
    fn add(&self, other: &Self) -> Result<Self>;
    
    /// Element-wise subtraction
    fn sub(&self, other: &Self) -> Result<Self>;
    
    /// Element-wise multiplication
    fn mul(&self, other: &Self) -> Result<Self>;
    
    /// Element-wise division
    fn div(&self, other: &Self) -> Result<Self>;
    
    /// Matrix multiplication
    fn matmul(&self, other: &Self) -> Result<Self>;
    
    /// Element-wise power
    fn pow(&self, exponent: f32) -> Result<Self>;
    
    /// Square root
    fn sqrt(&self) -> Result<Self>;
    
    /// Exponential
    fn exp(&self) -> Result<Self>;
    
    /// Natural logarithm
    fn log(&self) -> Result<Self>;
    
    /// Sine
    fn sin(&self) -> Result<Self>;
    
    /// Cosine
    fn cos(&self) -> Result<Self>;
    
    /// Tangent
    fn tan(&self) -> Result<Self>;
    
    /// Absolute value
    fn abs(&self) -> Result<Self>;
    
    /// Maximum of two tensors
    fn max(&self, other: &Self) -> Result<Self>;
    
    /// Minimum of two tensors
    fn min(&self, other: &Self) -> Result<Self>;
    
    /// Clamp values between min and max
    fn clamp(&self, min: f32, max: f32) -> Result<Self>;
}

/// Reduction operations
pub trait TensorReduce: Tensor {
    /// Sum over specified dimensions
    fn sum(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self>;
    
    /// Mean over specified dimensions
    fn mean(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self>;
    
    /// Maximum over specified dimensions
    fn max_reduce(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self>;
    
    /// Minimum over specified dimensions
    fn min_reduce(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self>;
    
    /// Variance over specified dimensions
    fn var(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self>;
    
    /// Standard deviation over specified dimensions
    fn std(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Result<Self>;
}

/// Statistical operations
pub trait TensorStats: Tensor {
    /// Argmax along specified dimension
    fn argmax(&self, dim: Option<usize>) -> Result<Self>;
    
    /// Argmin along specified dimension
    fn argmin(&self, dim: Option<usize>) -> Result<Self>;
    
    /// Softmax along specified dimension
    fn softmax(&self, dim: usize) -> Result<Self>;
    
    /// Log softmax along specified dimension
    fn log_softmax(&self, dim: usize) -> Result<Self>;
    
    /// Top-k values and indices
    fn topk(&self, k: usize, dim: usize, largest: bool) -> Result<(Self, Self)>;
}

/// Activation functions
pub trait TensorActivation: Tensor {
    /// ReLU activation
    fn relu(&self) -> Result<Self>;
    
    /// GELU activation
    fn gelu(&self) -> Result<Self>;
    
    /// SiLU/Swish activation
    fn silu(&self) -> Result<Self>;
    
    /// Sigmoid activation
    fn sigmoid(&self) -> Result<Self>;
    
    /// Tanh activation
    fn tanh(&self) -> Result<Self>;
    
    /// Leaky ReLU activation
    fn leaky_relu(&self, negative_slope: f32) -> Result<Self>;
    /// Negate tensor (element-wise)
    fn neg(&self) -> Result<Self>;
    
    /// Create tensor from scalar value
    fn from_scalar(value: f32, shape: Shape, dtype: DType, device: &Device) -> Result<Self>;
    
    /// Add scalar to all elements
    fn add_scalar(&mut self, scalar: f32) -> Result<()>;
    
    /// Multiply all elements by scalar
    fn mul_scalar(&mut self, scalar: f32) -> Result<()>;
    
    /// Greater than comparison (element-wise)
    fn gt(&self, other: &Self) -> Result<Self>;
    
    /// Greater than scalar comparison
    fn gt_scalar(&mut self, scalar: f32) -> Result<Self>;
    
    /// Extract scalar value from 0-dimensional or single-element tensor
    fn to_scalar(&self) -> Result<f32>;

}

/// Random tensor generation
pub trait TensorRandom: Tensor {
    /// Create tensor with random values from normal distribution
    fn random_normal(shape: Shape, mean: f32, std: f32, device: &Device) -> Result<Self>;
    
    /// Create tensor with random values from uniform distribution
    fn random_uniform(shape: Shape, min: f32, max: f32, device: &Device) -> Result<Self>;
    
    /// Create tensor with random values from Bernoulli distribution
    fn random_bernoulli(shape: Shape, p: f32, device: &Device) -> Result<Self>;
    
    /// Create tensor with zeros
    fn zeros(shape: Shape, dtype: DType, device: &Device) -> Result<Self>;
    
    /// Create tensor with ones
    fn ones(shape: Shape, dtype: DType, device: &Device) -> Result<Self>;
    
    /// Create tensor with identity matrix
    fn eye(n: usize, dtype: DType, device: &Device) -> Result<Self>;
    
    /// Create tensor with range of values
    fn arange(start: f32, end: f32, step: f32, dtype: DType, device: &Device) -> Result<Self>;
}
