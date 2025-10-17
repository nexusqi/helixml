//! 🌀 Gradient Functions
//! 
//! Specialized gradient computation functions for complex operations

use tensor_core::{Tensor, Result, TensorError, Shape, DType, Device};
use tensor_core::tensor::{TensorOps, TensorReduce, TensorStats, TensorBroadcast, TensorActivation};
use std::collections::HashMap;

/// Gradient function registry
pub struct GradientRegistry<T: Tensor> {
    functions: HashMap<String, Box<dyn GradientFunction<T> + Send + Sync>>,
}

impl<T: Tensor> GradientRegistry<T> {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }
    
    /// Register a gradient function
    pub fn register<F>(&mut self, name: &str, function: F)
    where
        F: GradientFunction<T> + Send + Sync + 'static,
    {
        self.functions.insert(name.to_string(), Box::new(function));
    }
    
    /// Get a gradient function
    pub fn get(&self, name: &str) -> Option<&(dyn GradientFunction<T> + Send + Sync)> {
        self.functions.get(name).map(|f| f.as_ref())
    }
}

/// Trait for gradient computation functions
pub trait GradientFunction<T: Tensor> {
    /// Compute gradient for the operation
    fn compute_gradient(
        &self,
        inputs: &[&T],
        output_grad: &T,
        operation_params: &OperationParams,
    ) -> Result<Vec<T>>;
}

/// Parameters for gradient computation
#[derive(Debug, Clone)]
pub struct OperationParams {
    pub dims: Option<Vec<usize>>,
    pub keep_dim: bool,
    pub alpha: f32,
    pub beta: f32,
    pub epsilon: f32,
}

impl Default for OperationParams {
    fn default() -> Self {
        Self {
            dims: None,
            keep_dim: false,
            alpha: 1.0,
            beta: 1.0,
            epsilon: 1e-8,
        }
    }
}

/// Gradient function for matrix multiplication
pub struct MatMulGradient<T: Tensor> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> MatMulGradient<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor + TensorOps> GradientFunction<T> for MatMulGradient<T> {
    fn compute_gradient(
        &self,
        inputs: &[&T],
        output_grad: &T,
        _params: &OperationParams,
    ) -> Result<Vec<T>> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInput {
                message: "MatMul requires exactly 2 inputs".to_string(),
            });
        }
        
        let left = inputs[0];
        let right = inputs[1];
        
        // For matrix transpose, we transpose the last two dimensions
        let ndim = right.shape().ndim();
        let dim0 = if ndim >= 2 { ndim - 2 } else { 0 };
        let dim1 = if ndim >= 2 { ndim - 1 } else { 0 };
        
        // Gradient w.r.t. left input: output_grad @ right.T
        let left_grad = output_grad.matmul(&right.transpose(dim0, dim1)?)?;
        
        // Gradient w.r.t. right input: left.T @ output_grad
        let left_ndim = left.shape().ndim();
        let left_dim0 = if left_ndim >= 2 { left_ndim - 2 } else { 0 };
        let left_dim1 = if left_ndim >= 2 { left_ndim - 1 } else { 0 };
        let right_grad = left.transpose(left_dim0, left_dim1)?.matmul(output_grad)?;
        
        Ok(vec![left_grad, right_grad])
    }
}

/// Gradient function for convolution
pub struct Conv2dGradient<T: Tensor> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> Conv2dGradient<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor + TensorOps> GradientFunction<T> for Conv2dGradient<T> {
    fn compute_gradient(
        &self,
        inputs: &[&T],
        output_grad: &T,
        _params: &OperationParams,
    ) -> Result<Vec<T>> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInput {
                message: "Conv2d requires exactly 2 inputs".to_string(),
            });
        }
        
        let input = inputs[0];
        let weight = inputs[1];
        
        // TODO: Implement proper convolution gradient computation
        // This is a placeholder implementation
        let input_grad = input.clone();
        let weight_grad = weight.clone();
        
        Ok(vec![input_grad, weight_grad])
    }
}

/// Gradient function for batch normalization
pub struct BatchNormGradient<T: Tensor> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> BatchNormGradient<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor + TensorOps + TensorReduce + TensorStats> GradientFunction<T> for BatchNormGradient<T> {
    fn compute_gradient(
        &self,
        inputs: &[&T],
        output_grad: &T,
        params: &OperationParams,
    ) -> Result<Vec<T>> {
        if inputs.len() != 3 {
            return Err(TensorError::InvalidInput {
                message: "BatchNorm requires exactly 3 inputs (input, weight, bias)".to_string(),
            });
        }
        
        let input = inputs[0];
        let weight = inputs[1];
        let bias = inputs[2];
        
        // Compute batch statistics
        let mean = input.mean(None, false)?;
        let var = input.var(None, false)?;
        let std = var.add_scalar(params.epsilon)?.sqrt()?;
        
        // Normalize input
        let normalized = input.sub(&mean)?.div(&std)?;
        
        // Gradient w.r.t. input
        let input_grad = output_grad.mul(weight)?;
        
        // Gradient w.r.t. weight
        let weight_grad = output_grad.mul(&normalized)?;
        
        // Gradient w.r.t. bias
        let bias_grad = output_grad.sum(None, false)?;
        
        Ok(vec![input_grad, weight_grad, bias_grad])
    }
}

/// Gradient function for attention
pub struct AttentionGradient<T: Tensor> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> AttentionGradient<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor + TensorOps + TensorReduce + TensorStats> GradientFunction<T> for AttentionGradient<T> {
    fn compute_gradient(
        &self,
        inputs: &[&T],
        output_grad: &T,
        _params: &OperationParams,
    ) -> Result<Vec<T>> {
        if inputs.len() != 3 {
            return Err(TensorError::InvalidInput {
                message: "Attention requires exactly 3 inputs (Q, K, V)".to_string(),
            });
        }
        
        let q = inputs[0];
        let k = inputs[1];
        let v = inputs[2];
        
        // TODO: Implement proper attention gradient computation
        // This is a placeholder implementation
        let q_grad = q.clone();
        let k_grad = k.clone();
        let v_grad = v.clone();
        
        Ok(vec![q_grad, k_grad, v_grad])
    }
}

/// Gradient function for LSTM
pub struct LSTMGradient<T: Tensor> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> LSTMGradient<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor + TensorOps + TensorReduce + TensorStats> GradientFunction<T> for LSTMGradient<T> {
    fn compute_gradient(
        &self,
        inputs: &[&T],
        output_grad: &T,
        _params: &OperationParams,
    ) -> Result<Vec<T>> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInput {
                message: "LSTM requires exactly 2 inputs (input, hidden)".to_string(),
            });
        }
        
        let input = inputs[0];
        let hidden = inputs[1];
        
        // TODO: Implement proper LSTM gradient computation
        // This is a placeholder implementation
        let input_grad = input.clone();
        let hidden_grad = hidden.clone();
        
        Ok(vec![input_grad, hidden_grad])
    }
}

/// Gradient function for SSM (State Space Model)
pub struct SSMGradient<T: Tensor> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> SSMGradient<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor + TensorOps + TensorReduce + TensorStats> GradientFunction<T> for SSMGradient<T> {
    fn compute_gradient(
        &self,
        inputs: &[&T],
        output_grad: &T,
        _params: &OperationParams,
    ) -> Result<Vec<T>> {
        if inputs.len() != 4 {
            return Err(TensorError::InvalidInput {
                message: "SSM requires exactly 4 inputs (A, B, C, D)".to_string(),
            });
        }
        
        let a = inputs[0];
        let b = inputs[1];
        let c = inputs[2];
        let d = inputs[3];
        
        // TODO: Implement proper SSM gradient computation
        // This is a placeholder implementation
        let a_grad = a.clone();
        let b_grad = b.clone();
        let c_grad = c.clone();
        let d_grad = d.clone();
        
        Ok(vec![a_grad, b_grad, c_grad, d_grad])
    }
}

/// Gradient function for Hyena
pub struct HyenaGradient<T: Tensor> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> HyenaGradient<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor + TensorOps + TensorReduce + TensorStats> GradientFunction<T> for HyenaGradient<T> {
    fn compute_gradient(
        &self,
        inputs: &[&T],
        output_grad: &T,
        _params: &OperationParams,
    ) -> Result<Vec<T>> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInput {
                message: "Hyena requires exactly 2 inputs (input, filter)".to_string(),
            });
        }
        
        let input = inputs[0];
        let filter = inputs[1];
        
        // TODO: Implement proper Hyena gradient computation
        // This is a placeholder implementation
        let input_grad = input.clone();
        let filter_grad = filter.clone();
        
        Ok(vec![input_grad, filter_grad])
    }
}

/// Create default gradient registry
pub fn create_default_registry<T: Tensor + TensorOps + TensorReduce + TensorStats + 'static>() -> GradientRegistry<T> {
    let mut registry = GradientRegistry::new();
    
    registry.register("matmul", MatMulGradient::new());
    registry.register("conv2d", Conv2dGradient::new());
    registry.register("batch_norm", BatchNormGradient::new());
    registry.register("attention", AttentionGradient::new());
    registry.register("lstm", LSTMGradient::new());
    registry.register("ssm", SSMGradient::new());
    registry.register("hyena", HyenaGradient::new());
    
    registry
}
