//! 🌀 Complete Autograd Example
//! 
//! Demonstrates the full autograd system with backward pass, gradient optimization, and training

use helix_ml::*;
use tensor_core::{Tensor, Shape, DType, Device};
use tensor_core::tensor::TensorRandom;
use backend_cpu::CpuTensor;
use autograd::*;
use nn::*;
use optim::*;
use training::*;
use anyhow::Result;
use tracing::{info, warn, error};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🚀 Starting Complete Autograd Example");
    
    // Create device
    let device = Device::cpu();
    
    // Example 1: Basic autograd operations
    basic_autograd_example(&device)?;
    
    // Example 2: Advanced gradient optimization
    gradient_optimization_example(&device)?;
    
    // Example 3: Training with autograd
    training_example(&device)?;
    
    // Example 4: Gradient flow analysis
    gradient_flow_example(&device)?;
    
    info!("✅ Complete Autograd Example finished successfully!");
    Ok(())
}

/// Basic autograd operations example
fn basic_autograd_example(device: &Device) -> Result<()> {
    info!("📚 Running Basic Autograd Example");
    
    // Create autograd operations
    let mut autograd_ops = AutogradOps::<CpuTensor>::new();
    
    // Create input tensors
    let x = CpuTensor::random_uniform(Shape::new(vec![2, 3]), 0.0, 1.0, device)?;
    let y = CpuTensor::random_uniform(Shape::new(vec![2, 3]), 0.0, 1.0, device)?;
    
    let x_id = autograd_ops.tensor(x, true);
    let y_id = autograd_ops.tensor(y, true);
    
    // Perform operations
    let add_id = autograd_ops.add(x_id, y_id)?;
    let mul_id = autograd_ops.mul(add_id, x_id)?;
    let relu_id = autograd_ops.relu(mul_id)?;
    let sum_id = autograd_ops.sum(relu_id, None, false)?;
    
    // Execute backward pass
    autograd_ops.backward(sum_id)?;
    
    // Check gradients
    if let Some(x_tensor) = autograd_ops.get_tensor(x_id) {
        if let Some(grad) = x_tensor.grad() {
            info!("✅ X gradient computed: shape {:?}", grad.shape());
        }
    }
    
    if let Some(y_tensor) = autograd_ops.get_tensor(y_id) {
        if let Some(grad) = y_tensor.grad() {
            info!("✅ Y gradient computed: shape {:?}", grad.shape());
        }
    }
    
    info!("✅ Basic Autograd Example completed");
    Ok(())
}

/// Gradient optimization example
fn gradient_optimization_example(device: &Device) -> Result<()> {
    info!("🔧 Running Gradient Optimization Example");
    
    // Create gradient optimizer with mixed precision
    let optimization = GradientOptimization::MixedPrecision { loss_scale: 16.0 };
    let mut optimizer = GradientOptimizer::<CpuTensor>::new(optimization);
    
    // Create autograd context
    let mut ctx = AutogradContext::<CpuTensor>::new();
    
    // Create tensors
    let x = CpuTensor::random_uniform(Shape::new(vec![4, 4]), 0.0, 1.0, device)?;
    let y = CpuTensor::random_uniform(Shape::new(vec![4, 4]), 0.0, 1.0, device)?;
    
    let x_id = ctx.tensor(x, true);
    let y_id = ctx.tensor(y, true);
    
    // Perform computation
    let result = ctx.get_tensor(x_id).unwrap().tensor().add(ctx.get_tensor(y_id).unwrap().tensor())?;
    let result_id = ctx.tensor(result, true);
    
    // Set gradients
    if let Some(result_tensor) = ctx.get_tensor_mut(result_id) {
        let grad = CpuTensor::ones(Shape::new(vec![4, 4]), DType::F32, device)?;
        result_tensor.set_grad(grad);
    }
    
    // Optimize gradients
    optimizer.optimize_gradients(&mut ctx)?;
    
    // Get optimization statistics
    let stats = optimizer.get_stats();
    info!("📊 Optimization Stats: {:?}", stats);
    
    info!("✅ Gradient Optimization Example completed");
    Ok(())
}

/// Training example with autograd
fn training_example(device: &Device) -> Result<()> {
    info!("🏋️ Running Training Example");
    
    // Create training state
    let training_state = TrainingState::<CpuTensor>::new(
        4, // accumulation steps
        1.0, // max grad norm
        true, // mixed precision
        CheckpointStrategy::EveryN(2), // checkpoint every 2 layers
    );
    
    // Create autograd context
    let mut ctx = AutogradContext::<CpuTensor>::new();
    
    // Create model parameters
    let weight = CpuTensor::random_normal(Shape::new(vec![10, 5]), 0.0, 0.1, device)?;
    let bias = CpuTensor::zeros(Shape::new(vec![5]), DType::F32, device)?;
    
    let weight_id = ctx.tensor(weight, true);
    let bias_id = ctx.tensor(bias, true);
    
    // Create input data
    let input = CpuTensor::random_uniform(Shape::new(vec![32, 10]), 0.0, 1.0, device)?;
    let target = CpuTensor::random_uniform(Shape::new(vec![32, 5]), 0.0, 1.0, device)?;
    
    let input_id = ctx.tensor(input, false);
    let target_id = ctx.tensor(target, false);
    
    // Forward pass
    let weight_tensor = ctx.get_tensor(weight_id).unwrap();
    let bias_tensor = ctx.get_tensor(bias_id).unwrap();
    let input_tensor = ctx.get_tensor(input_id).unwrap();
    
    let output = input_tensor.tensor().matmul(weight_tensor.tensor())?.add(bias_tensor.tensor())?;
    let output_id = ctx.tensor(output, true);
    
    // Compute loss
    let target_tensor = ctx.get_tensor(target_id).unwrap();
    let loss = output.sub(target_tensor.tensor())?.pow_scalar(2.0)?.mean(None, false)?;
    let loss_id = ctx.tensor(loss, true);
    
    // Set loss gradient
    if let Some(loss_tensor) = ctx.get_tensor_mut(loss_id) {
        let grad = CpuTensor::ones(Shape::new(vec![]), DType::F32, device)?;
        loss_tensor.set_grad(grad);
    }
    
    // Execute training step
    let result = training_state.training_step(
        &mut ctx,
        |_ctx| Ok(loss_id), // forward function
        |_ctx, _loss_id| Ok(()), // backward function (placeholder)
    )?;
    
    info!("📈 Training Step Result: {:?}", result);
    
    info!("✅ Training Example completed");
    Ok(())
}

/// Gradient flow analysis example
fn gradient_flow_example(device: &Device) -> Result<()> {
    info!("📊 Running Gradient Flow Analysis Example");
    
    // Create gradient flow analyzer
    let mut analyzer = GradientFlowAnalyzer::<CpuTensor>::new();
    
    // Create autograd context
    let mut ctx = AutogradContext::<CpuTensor>::new();
    
    // Create a simple network
    let w1 = CpuTensor::random_normal(Shape::new(vec![10, 5]), 0.0, 0.1, device)?;
    let w2 = CpuTensor::random_normal(Shape::new(vec![5, 1]), 0.0, 0.1, device)?;
    
    let w1_id = ctx.tensor(w1, true);
    let w2_id = ctx.tensor(w2, true);
    
    // Create input
    let input = CpuTensor::random_uniform(Shape::new(vec![32, 10]), 0.0, 1.0, device)?;
    let input_id = ctx.tensor(input, false);
    
    // Forward pass
    let w1_tensor = ctx.get_tensor(w1_id).unwrap();
    let w2_tensor = ctx.get_tensor(w2_id).unwrap();
    let input_tensor = ctx.get_tensor(input_id).unwrap();
    
    let h1 = input_tensor.tensor().matmul(w1_tensor.tensor())?.relu()?;
    let h1_id = ctx.tensor(h1, true);
    
    let h1_tensor = ctx.get_tensor(h1_id).unwrap();
    let output = h1_tensor.tensor().matmul(w2_tensor.tensor())?;
    let output_id = ctx.tensor(output, true);
    
    // Compute loss
    let target = CpuTensor::random_uniform(Shape::new(vec![32, 1]), 0.0, 1.0, device)?;
    let target_id = ctx.tensor(target, false);
    
    let target_tensor = ctx.get_tensor(target_id).unwrap();
    let loss = output.sub(target_tensor.tensor())?.pow_scalar(2.0)?.mean(None, false)?;
    let loss_id = ctx.tensor(loss, true);
    
    // Set gradients
    if let Some(loss_tensor) = ctx.get_tensor_mut(loss_id) {
        let grad = CpuTensor::ones(Shape::new(vec![]), DType::F32, device)?;
        loss_tensor.set_grad(grad);
    }
    
    // Analyze gradient flow
    let report = analyzer.analyze_flow(&ctx)?;
    
    info!("📊 Gradient Flow Report:");
    info!("  - Vanishing gradients: {:?}", report.vanishing_gradients);
    info!("  - Exploding gradients: {:?}", report.exploding_gradients);
    info!("  - Gradient norms: {:?}", report.gradient_norms);
    
    if !report.vanishing_gradients.is_empty() {
        warn!("⚠️  Detected vanishing gradients in tensors: {:?}", report.vanishing_gradients);
    }
    
    if !report.exploding_gradients.is_empty() {
        warn!("⚠️  Detected exploding gradients in tensors: {:?}", report.exploding_gradients);
    }
    
    info!("✅ Gradient Flow Analysis Example completed");
    Ok(())
}
