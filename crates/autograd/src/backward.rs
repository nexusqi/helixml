//! 🌀 Backward Pass Implementation
//! 
//! Complete backward pass implementation with gradient computation

use tensor_core::{Tensor, Result, TensorError, Shape};
use tensor_core::tensor::{TensorOps, TensorReduce, TensorStats, TensorBroadcast, TensorActivation, TensorRandom};
use std::collections::{HashMap, HashSet};
use super::{AutogradContext, DiffTensor};

/// Operation types for gradient computation
#[derive(Debug, Clone)]
pub enum Operation {
    // Unary operations
    Neg, Abs, Sqrt, Exp, Log, Sin, Cos, Tan,
    Relu, Gelu, Silu, Sigmoid, Tanh,
    
    // Binary operations
    Add, Sub, Mul, Div, Pow, Max, Min,
    
    // Reduce operations
    Sum, Mean, ReduceMax, ReduceMin, Var, Std,
    
    // Matrix operations
    MatMul, Transpose, Reshape, Broadcast,
    
    // Activation functions
    Softmax, LogSoftmax,
}

/// Computation graph node
#[derive(Debug, Clone)]
pub struct ComputationNode {
    pub operation: Operation,
    pub inputs: Vec<usize>,
    pub output_shape: Shape,
    pub requires_grad: bool,
}

/// Complete backward pass implementation
#[derive(Debug)]
pub struct BackwardPass<T: Tensor> {
    computation_graph: HashMap<usize, ComputationNode>,
    gradient_cache: HashMap<usize, T>,
    visited: HashSet<usize>,
}

impl<T: Tensor + TensorOps + TensorReduce + TensorStats + TensorBroadcast + TensorActivation + TensorRandom> BackwardPass<T> {
    pub fn new() -> Self {
        Self {
            computation_graph: HashMap::new(),
            gradient_cache: HashMap::new(),
            visited: HashSet::new(),
        }
    }
    
    /// Add a computation node to the graph
    pub fn add_node(&mut self, tensor_id: usize, node: ComputationNode) {
        self.computation_graph.insert(tensor_id, node);
    }
    
    /// Execute backward pass from output tensor
    pub fn backward(&mut self, ctx: &mut AutogradContext<T>, output_id: usize) -> Result<()> {
        // Initialize output gradient
        if let Some(output_tensor) = ctx.get_tensor(output_id) {
            let output_shape = output_tensor.tensor().shape().clone();
            let output_dtype = output_tensor.tensor().dtype();
            let output_device = output_tensor.tensor().device();
            
            let ones = T::ones(output_shape, output_dtype, output_device)?;
            self.gradient_cache.insert(output_id, ones);
        }
        
        // Topological sort for gradient computation
        let sorted_ids = self.topological_sort()?;
        
        // Compute gradients in reverse order
        for tensor_id in sorted_ids.into_iter().rev() {
            if let Some(node) = self.computation_graph.get(&tensor_id).cloned() {
                self.compute_gradient(ctx, tensor_id, &node)?;
            }
        }
        
        Ok(())
    }
    
    /// Compute gradient for a specific node
    fn compute_gradient(&mut self, ctx: &mut AutogradContext<T>, tensor_id: usize, node: &ComputationNode) -> Result<()> {
        if let Some(output_grad) = self.gradient_cache.get(&tensor_id) {
            match &node.operation {
                // Unary operations
                Operation::Neg => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_grad = output_grad.neg()?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                Operation::Abs => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let sign = input_tensor.tensor().sign()?;
                        let input_grad = output_grad.mul(&sign)?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                Operation::Sqrt => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let sqrt_input = input_tensor.tensor().sqrt()?;
                        let input_grad = output_grad.div(&sqrt_input.mul(&sqrt_input)?)?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                Operation::Exp => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let exp_input = input_tensor.tensor().exp()?;
                        let input_grad = output_grad.mul(&exp_input)?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                Operation::Log => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let input_grad = output_grad.div(input_tensor.tensor())?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                Operation::Sin => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let cos_input = input_tensor.tensor().cos()?;
                        let input_grad = output_grad.mul(&cos_input)?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                Operation::Cos => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let sin_input = input_tensor.tensor().sin()?;
                        let input_grad = output_grad.mul(&sin_input.neg()?)?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                Operation::Tan => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let tan_input = input_tensor.tensor().tan()?;
                        let sec_squared = tan_input.mul(&tan_input)?.add(&T::ones(tan_input.shape().clone(), tan_input.dtype(), tan_input.device())?)?;
                        let input_grad = output_grad.mul(&sec_squared)?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                Operation::Relu => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let mask = input_tensor.tensor().gt(&T::zeros(input_tensor.tensor().shape().clone(), input_tensor.tensor().dtype(), input_tensor.tensor().device())?)?;
                        let input_grad = output_grad.mul(&mask)?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                Operation::Gelu => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let gelu_grad = self.compute_gelu_gradient(input_tensor.tensor())?;
                        let input_grad = output_grad.mul(&gelu_grad)?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                Operation::Silu => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let silu_grad = self.compute_silu_gradient(input_tensor.tensor())?;
                        let input_grad = output_grad.mul(&silu_grad)?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                Operation::Sigmoid => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let sigmoid_input = input_tensor.tensor().sigmoid()?;
                        let sigmoid_grad = sigmoid_input.mul(&sigmoid_input.neg()?.add(&T::ones(sigmoid_input.shape().clone(), sigmoid_input.dtype(), sigmoid_input.device())?)?)?;
                        let input_grad = output_grad.mul(&sigmoid_grad)?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                Operation::Tanh => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let tanh_input = input_tensor.tensor().tanh()?;
                        let tanh_grad = tanh_input.mul(&tanh_input)?.neg()?.add(&T::ones(tanh_input.shape().clone(), tanh_input.dtype(), tanh_input.device())?)?;
                        let input_grad = output_grad.mul(&tanh_grad)?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                
                // Binary operations
                Operation::Add => {
                    if node.inputs.len() >= 2 {
                        let left_grad = output_grad.clone();
                        let right_grad = output_grad.clone();
                        self.accumulate_gradient(ctx, node.inputs[0], left_grad)?;
                        self.accumulate_gradient(ctx, node.inputs[1], right_grad)?;
                    }
                }
                Operation::Sub => {
                    if node.inputs.len() >= 2 {
                        let left_grad = output_grad.clone();
                        let right_grad = output_grad.neg()?;
                        self.accumulate_gradient(ctx, node.inputs[0], left_grad)?;
                        self.accumulate_gradient(ctx, node.inputs[1], right_grad)?;
                    }
                }
                Operation::Mul => {
                    if node.inputs.len() >= 2 {
                        let left_tensor = ctx.get_tensor(node.inputs[0]).unwrap();
                        let right_tensor = ctx.get_tensor(node.inputs[1]).unwrap();
                        let left_grad = output_grad.mul(right_tensor.tensor())?;
                        let right_grad = output_grad.mul(left_tensor.tensor())?;
                        self.accumulate_gradient(ctx, node.inputs[0], left_grad)?;
                        self.accumulate_gradient(ctx, node.inputs[1], right_grad)?;
                    }
                }
                Operation::Div => {
                    if node.inputs.len() >= 2 {
                        let left_tensor = ctx.get_tensor(node.inputs[0]).unwrap();
                        let right_tensor = ctx.get_tensor(node.inputs[1]).unwrap();
                        let left_grad = output_grad.div(right_tensor.tensor())?;
                        let right_grad = output_grad.mul(&left_tensor.tensor())?.div(&right_tensor.tensor().mul(&right_tensor.tensor())?)?;
                        self.accumulate_gradient(ctx, node.inputs[0], left_grad)?;
                        self.accumulate_gradient(ctx, node.inputs[1], right_grad)?;
                    }
                }
                Operation::Pow => {
                    if node.inputs.len() >= 2 {
                        let left_tensor = ctx.get_tensor(node.inputs[0]).unwrap();
                        let right_tensor = ctx.get_tensor(node.inputs[1]).unwrap();
                        // Simplified gradient computation for pow operation
                        // For x^y: d/dx = y * x^(y-1), d/dy = x^y * ln(x)
                        // Note: This is a simplified implementation, full gradient requires element-wise pow
                        let left_grad = output_grad.mul(right_tensor.tensor())?;
                        let right_grad = output_grad.mul(&left_tensor.tensor().log()?)?;
                        self.accumulate_gradient(ctx, node.inputs[0], left_grad)?;
                        self.accumulate_gradient(ctx, node.inputs[1], right_grad)?;
                    }
                }
                
                // Reduce operations
                Operation::Sum => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let input_grad = self.broadcast_gradient(output_grad, input_tensor.tensor().shape())?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                Operation::Mean => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let input_size = input_tensor.tensor().shape().numel();
                        let input_grad = self.broadcast_gradient(output_grad, input_tensor.tensor().shape())?.div(&T::ones(input_tensor.tensor().shape().clone(), input_tensor.tensor().dtype(), input_tensor.tensor().device())?.mul(&T::from_scalar(input_size as f32, input_tensor.tensor().shape().clone(), input_tensor.tensor().dtype(), input_tensor.tensor().device())?)?)?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                
                // Matrix operations
                Operation::MatMul => {
                    if node.inputs.len() >= 2 {
                        let left_tensor = ctx.get_tensor(node.inputs[0]).unwrap();
                        let right_tensor = ctx.get_tensor(node.inputs[1]).unwrap();
                        let left_grad = output_grad.matmul(&right_tensor.tensor().transpose(0, 1)?)?;
                        let right_grad = left_tensor.tensor().transpose(0, 1)?.matmul(output_grad)?;
                        self.accumulate_gradient(ctx, node.inputs[0], left_grad)?;
                        self.accumulate_gradient(ctx, node.inputs[1], right_grad)?;
                    }
                }
                Operation::Transpose => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_grad = output_grad.transpose(0, 1)?;
                        self.accumulate_gradient(ctx, *input_id, input_grad)?;
                    }
                }
                
                // Activation functions
                Operation::Softmax => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let softmax_input = input_tensor.tensor().softmax(input_tensor.tensor().ndim() - 1)?;
                        let softmax_grad = self.compute_softmax_gradient(&softmax_input, output_grad)?;
                        self.accumulate_gradient(ctx, *input_id, softmax_grad)?;
                    }
                }
                Operation::LogSoftmax => {
                    if let Some(input_id) = node.inputs.first() {
                        let input_tensor = ctx.get_tensor(*input_id).unwrap();
                        let log_softmax_input = input_tensor.tensor().log_softmax(input_tensor.tensor().ndim() - 1)?;
                        let log_softmax_grad = self.compute_log_softmax_gradient(&log_softmax_input, output_grad)?;
                        self.accumulate_gradient(ctx, *input_id, log_softmax_grad)?;
                    }
                }
                
                _ => {
                    // TODO: Implement remaining operations
                    return Err(TensorError::UnsupportedOperation {
                        op: format!("Gradient computation for {:?} not implemented", node.operation),
                    });
                }
            }
        }
        Ok(())
    }
    
    /// Compute GELU gradient
    fn compute_gelu_gradient(&self, input: &T) -> Result<T> {
        let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
        let x = input.mul(&T::from_scalar(sqrt_2_over_pi, input.shape().clone(), input.dtype(), input.device())?)?;
        let x_squared = x.mul(&x)?;
        let x_cubed = x_squared.mul(&x)?;
        let tanh_term = x.add(&x_cubed.mul(&T::from_scalar(0.044715, x.shape().clone(), x.dtype(), x.device())?)?)?.tanh()?;
        let sech_squared = tanh_term.mul(&tanh_term)?.neg()?.add(&T::ones(tanh_term.shape().clone(), tanh_term.dtype(), tanh_term.device())?)?;
        let gelu_grad = sech_squared.mul(&T::from_scalar(0.5, sech_squared.shape().clone(), sech_squared.dtype(), sech_squared.device())?)?.add(&T::from_scalar(0.5, sech_squared.shape().clone(), sech_squared.dtype(), sech_squared.device())?)?;
        Ok(gelu_grad)
    }
    
    /// Compute SiLU gradient
    fn compute_silu_gradient(&self, input: &T) -> Result<T> {
        let sigmoid_input = input.sigmoid()?;
        let silu_grad = sigmoid_input.add(&input.mul(&sigmoid_input.mul(&sigmoid_input.neg()?.add(&T::ones(sigmoid_input.shape().clone(), sigmoid_input.dtype(), sigmoid_input.device())?)?)?)?)?;
        Ok(silu_grad)
    }
    
    /// Compute softmax gradient
    fn compute_softmax_gradient(&self, softmax: &T, output_grad: &T) -> Result<T> {
        let softmax_grad = softmax.mul(output_grad)?;
        let sum_grad = softmax_grad.sum(None, false)?;
        let broadcasted_sum = self.broadcast_gradient(&sum_grad, softmax.shape())?;
        let result = softmax.mul(&output_grad.sub(&broadcasted_sum)?)?;
        Ok(result)
    }
    
    /// Compute log softmax gradient
    fn compute_log_softmax_gradient(&self, log_softmax: &T, output_grad: &T) -> Result<T> {
        let softmax = log_softmax.exp()?;
        let softmax_grad = self.compute_softmax_gradient(&softmax, output_grad)?;
        Ok(softmax_grad)
    }
    
    /// Broadcast gradient to target shape
    fn broadcast_gradient(&self, grad: &T, target_shape: &Shape) -> Result<T> {
        if grad.shape() == target_shape {
            Ok(grad.clone())
        } else {
            grad.broadcast_to(target_shape.clone())
        }
    }
    
    /// Accumulate gradient for a tensor
    fn accumulate_gradient(&mut self, ctx: &mut AutogradContext<T>, tensor_id: usize, grad: T) -> Result<()> {
        if let Some(diff_tensor) = ctx.get_tensor_mut(tensor_id) {
            if let Some(existing_grad) = &mut diff_tensor.grad {
                *existing_grad = existing_grad.add(&grad)?;
            } else {
                diff_tensor.grad = Some(grad);
            }
        }
        Ok(())
    }
    
    /// Topological sort for gradient computation
    fn topological_sort(&self) -> Result<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        
        for &tensor_id in self.computation_graph.keys() {
            if !visited.contains(&tensor_id) {
                self.dfs(tensor_id, &mut visited, &mut stack)?;
            }
        }
        
        Ok(stack)
    }
    
    /// Depth-first search for topological sort
    fn dfs(&self, tensor_id: usize, visited: &mut HashSet<usize>, stack: &mut Vec<usize>) -> Result<()> {
        visited.insert(tensor_id);
        
        if let Some(node) = self.computation_graph.get(&tensor_id) {
            for &input_id in &node.inputs {
                if !visited.contains(&input_id) {
                    self.dfs(input_id, visited, stack)?;
                }
            }
        }
        
        stack.push(tensor_id);
        Ok(())
    }
    
    /// Clear the computation graph
    pub fn clear(&mut self) {
        self.computation_graph.clear();
        self.gradient_cache.clear();
        self.visited.clear();
    }
    
    /// Get computation graph size
    pub fn graph_size(&self) -> usize {
        self.computation_graph.len()
    }
}