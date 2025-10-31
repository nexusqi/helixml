//! 🌀 HelixML Optimizers
//! 
//! Advanced optimizers with adaptive learning rates and momentum.

use tensor_core::tensor::{Tensor, TensorOps};
use std::collections::HashMap;
use std::ptr;
use anyhow::Result as AnyResult;

/// Trait for optimizers
pub trait Optimizer<T: Tensor + TensorOps>: Send + Sync {
    /// Update parameters using gradients
    /// Takes pairs of (parameter, gradient) tensors
    fn step(&mut self, param_grad_pairs: &mut [(&mut T, &T)]) -> AnyResult<()>;
    
    /// Zero gradients (clears internal state)
    fn zero_grad(&mut self) -> AnyResult<()>;
    
    /// Get optimizer name
    fn name(&self) -> &str;
    
    /// Get optimizer parameters
    fn parameters(&self) -> HashMap<String, f64>;
    
    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f64);
    
    /// Get learning rate
    fn get_learning_rate(&self) -> f64;
}

/// Adam optimizer
pub struct Adam<T: Tensor> {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    first_moments: HashMap<usize, T>,
    second_moments: HashMap<usize, T>,
    step_count: usize,
}

impl<T: Tensor + TensorOps> Adam<T> {
    pub fn new(learning_rate: f64) -> Self {
        Self::with_params(learning_rate, 0.9, 0.999, 1e-8)
    }
    
    pub fn with_params(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            step_count: 0,
        }
    }
}

impl<T: Tensor + TensorOps> Optimizer<T> for Adam<T> {
    fn step(&mut self, param_grad_pairs: &mut [(&mut T, &T)]) -> AnyResult<()> {
        // Adam algorithm:
        // m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        // v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        // m_hat = m_t / (1 - beta1^t)
        // v_hat = v_t / (1 - beta2^t)
        // param = param - lr * m_hat / (sqrt(v_hat) + eps)
        
        self.step_count += 1;
        let t = self.step_count as f64;
        let beta1_t = 1.0 - self.beta1.powi(t as i32);
        let beta2_t = 1.0 - self.beta2.powi(t as i32);
        
        for (idx, (param, grad_ref)) in param_grad_pairs.iter_mut().enumerate() {
            // Apply weight decay if needed
            let grad: T = if self.weight_decay > 0.0 {
                // grad = grad + weight_decay * param
                let decay = (*param).mul_scalar(self.weight_decay as f32)?;
                (*grad_ref).add(&decay)?
            } else {
                (*grad_ref).clone()
            };
            
            // Get or initialize moments
            let grad_shape = grad.shape().clone();
            let grad_dtype = grad.dtype();
            let grad_device = grad.device();
            
            let m = self.first_moments.entry(idx)
                .or_insert_with(|| {
                    // Initialize with zeros same shape as gradient
                    T::from_scalar(0.0, grad_shape.clone(), grad_dtype, grad_device)
                        .unwrap_or_else(|_| grad.clone())
                });
            let m = m.clone();
            let v = self.second_moments.entry(idx)
                .or_insert_with(|| {
                    T::from_scalar(0.0, grad_shape.clone(), grad_dtype, grad_device)
                        .unwrap_or_else(|_| grad.clone())
                });
            let v = v.clone();
            
            // Update biased first moment estimate
            // m = beta1 * m + (1 - beta1) * grad
            let m_update = m.mul_scalar(self.beta1 as f32)?;
            let grad_scaled = grad.mul_scalar((1.0 - self.beta1) as f32)?;
            let m_new = m_update.add(&grad_scaled)?;
            self.first_moments.insert(idx, m_new.clone());
            
            // Update biased second raw moment estimate
            // v = beta2 * v + (1 - beta2) * grad^2
            let grad_squared = grad.mul(&grad)?;
            let v_update = v.mul_scalar(self.beta2 as f32)?;
            let grad_sq_scaled = grad_squared.mul_scalar((1.0 - self.beta2) as f32)?;
            let v_new = v_update.add(&grad_sq_scaled)?;
            self.second_moments.insert(idx, v_new.clone());
            
            // Compute bias-corrected moment estimates
            // m_hat = m / (1 - beta1^t)
            // v_hat = v / (1 - beta2^t)
            let m_hat = m_new.mul_scalar((1.0 / beta1_t) as f32)?;
            let v_hat = v_new.mul_scalar((1.0 / beta2_t) as f32)?;
            
            // Compute update
            // update = lr * m_hat / (sqrt(v_hat) + eps)
            let v_hat_sqrt = v_hat.sqrt()?;
            let denominator = v_hat_sqrt.add_scalar(self.epsilon as f32)?;
            let update = m_hat.div(&denominator)?;
            let scaled_update = update.mul_scalar(self.learning_rate as f32)?;
            
            // Update parameter: param = param - update
            let current_param = (*param).clone();
            let new_param = current_param.sub(&scaled_update)?;
            unsafe {
                ptr::write(*param, new_param);
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self) -> AnyResult<()> {
        // Clear moment estimates (optional - can also keep them)
        // For now, we keep them for momentum
        Ok(())
    }
    
    fn name(&self) -> &str {
        "Adam"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), self.learning_rate);
        params.insert("beta1".to_string(), self.beta1);
        params.insert("beta2".to_string(), self.beta2);
        params
    }
    
    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
    
    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
pub struct AdamW<T: Tensor> {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    first_moments: HashMap<usize, T>,
    second_moments: HashMap<usize, T>,
    step_count: usize,
}

impl<T: Tensor + TensorOps> AdamW<T> {
    pub fn new(learning_rate: f64, weight_decay: f64) -> Self {
        Self::with_params(learning_rate, 0.9, 0.999, weight_decay, 1e-8)
    }
    
    pub fn with_params(learning_rate: f64, beta1: f64, beta2: f64, weight_decay: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            step_count: 0,
        }
    }
}

impl<T: Tensor + TensorOps> Optimizer<T> for AdamW<T> {
    fn step(&mut self, param_grad_pairs: &mut [(&mut T, &T)]) -> AnyResult<()> {
        // AdamW algorithm (same as Adam but weight decay is applied to param, not grad):
        // m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        // v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        // m_hat = m_t / (1 - beta1^t)
        // v_hat = v_t / (1 - beta2^t)
        // param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
        
        self.step_count += 1;
        let t = self.step_count as f64;
        let beta1_t = 1.0 - self.beta1.powi(t as i32);
        let beta2_t = 1.0 - self.beta2.powi(t as i32);
        
        for (idx, (param, grad_ref)) in param_grad_pairs.iter_mut().enumerate() {
            let grad = (*grad_ref).clone();
            
            // Get or initialize moments
            let grad_shape = grad.shape().clone();
            let grad_dtype = grad.dtype();
            let grad_device = grad.device();
            
            let m = self.first_moments.entry(idx)
                .or_insert_with(|| {
                    T::from_scalar(0.0, grad_shape.clone(), grad_dtype, grad_device)
                        .unwrap_or_else(|_| grad.clone())
                });
            let m = m.clone();
            let v = self.second_moments.entry(idx)
                .or_insert_with(|| {
                    T::from_scalar(0.0, grad_shape.clone(), grad_dtype, grad_device)
                        .unwrap_or_else(|_| grad.clone())
                });
            let v = v.clone();
            
            // Update biased first moment estimate
            let m_update = m.mul_scalar(self.beta1 as f32)?;
            let grad_scaled = grad.mul_scalar((1.0 - self.beta1) as f32)?;
            let m_new = m_update.add(&grad_scaled)?;
            self.first_moments.insert(idx, m_new.clone());
            
            // Update biased second raw moment estimate
            let grad_squared = grad.mul(&grad)?;
            let v_update = v.mul_scalar(self.beta2 as f32)?;
            let grad_sq_scaled = grad_squared.mul_scalar((1.0 - self.beta2) as f32)?;
            let v_new = v_update.add(&grad_sq_scaled)?;
            self.second_moments.insert(idx, v_new.clone());
            
            // Compute bias-corrected moment estimates
            let m_hat = m_new.mul_scalar((1.0 / beta1_t) as f32)?;
            let v_hat = v_new.mul_scalar((1.0 / beta2_t) as f32)?;
            
            // Compute update (without weight decay in denominator)
            let v_hat_sqrt = v_hat.sqrt()?;
            let denominator = v_hat_sqrt.add_scalar(self.epsilon as f32)?;
            let update = m_hat.div(&denominator)?;
            let scaled_update = update.mul_scalar(self.learning_rate as f32)?;
            
            // Apply weight decay to parameter (decoupled)
            let weight_decay_update = if self.weight_decay > 0.0 {
                (*param).mul_scalar((self.learning_rate * self.weight_decay) as f32)?
            } else {
                (*param).mul_scalar(0.0)?
            };
            
            // Update parameter: param = param - lr * update - lr * weight_decay * param
            let current_param = (*param).clone();
            let total_update = scaled_update.add(&weight_decay_update)?;
            let new_param = current_param.sub(&total_update)?;
            unsafe {
                ptr::write(*param, new_param);
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self) -> AnyResult<()> {
        Ok(())
    }
    
    fn name(&self) -> &str {
        "AdamW"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), self.learning_rate);
        params.insert("weight_decay".to_string(), self.weight_decay);
        params
    }
    
    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
    
    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

/// SGD optimizer
pub struct SGD<T: Tensor> {
    learning_rate: f64,
    momentum: f64,
    dampening: f64,
    weight_decay: f64,
    nesterov: bool,
    momentum_buffer: HashMap<usize, T>,
}

impl<T: Tensor + TensorOps> SGD<T> {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            momentum_buffer: HashMap::new(),
        }
    }
    
    pub fn with_momentum(learning_rate: f64, momentum: f64) -> Self {
        Self::with_nesterov(learning_rate, momentum, false)
    }
    
    pub fn with_nesterov(learning_rate: f64, momentum: f64, nesterov: bool) -> Self {
        Self {
            learning_rate,
            momentum,
            dampening: momentum,
            weight_decay: 0.0,
            nesterov,
            momentum_buffer: HashMap::new(),
        }
    }
}

impl<T: Tensor + TensorOps> Optimizer<T> for SGD<T> {
    fn step(&mut self, param_grad_pairs: &mut [(&mut T, &T)]) -> AnyResult<()> {
        // SGD algorithm:
        // if momentum > 0:
        //   buf = momentum * buf + grad
        //   if nesterov:
        //     grad = grad + momentum * buf
        //   else:
        //     grad = buf
        // param = param - lr * (grad + weight_decay * param)
        
        for (idx, (param, grad_ref)) in param_grad_pairs.iter_mut().enumerate() {
            // Apply weight decay
            let grad: T = if self.weight_decay > 0.0 {
                let decay = (*param).mul_scalar(self.weight_decay as f32)?;
                (*grad_ref).add(&decay)?
            } else {
                (*grad_ref).clone()
            };
            
            // Handle momentum
            let grad_shape = grad.shape().clone();
            let grad_dtype = grad.dtype();
            let grad_device = grad.device();
            
            let update: T = if self.momentum > 0.0 {
                let buf = self.momentum_buffer.entry(idx)
                    .or_insert_with(|| {
                        T::from_scalar(0.0, grad_shape.clone(), grad_dtype, grad_device)
                            .unwrap_or_else(|_| grad.clone())
                    }).clone();
                
                // buf = momentum * buf + (1 - dampening) * grad
                let buf_scaled = buf.mul_scalar(self.momentum as f32)?;
                let grad_scaled = grad.mul_scalar((1.0 - self.dampening) as f32)?;
                let buf_new = buf_scaled.add(&grad_scaled)?;
                self.momentum_buffer.insert(idx, buf_new.clone());
                
                if self.nesterov {
                    // Nesterov: grad = grad + momentum * buf_new
                    grad.add(&buf_new.mul_scalar(self.momentum as f32)?)?
                } else {
                    buf_new
                }
            } else {
                grad.clone()
            };
            
            // Update parameter: param = param - lr * update
            let scaled_update = update.mul_scalar(self.learning_rate as f32)?;
            let current_param = (*param).clone();
            let new_param = current_param.sub(&scaled_update)?;
            unsafe {
                ptr::write(*param, new_param);
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self) -> AnyResult<()> {
        // Optionally clear momentum buffer
        // For now, keep it for momentum continuity
        Ok(())
    }
    
    fn name(&self) -> &str {
        "SGD"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), self.learning_rate);
        params.insert("momentum".to_string(), self.momentum);
        params
    }
    
    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
    
    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

/// RMSprop optimizer
pub struct RMSprop<T: Tensor> {
    learning_rate: f64,
    alpha: f64,
    epsilon: f64,
    weight_decay: f64,
    momentum: f64,
    centered: bool,
    velocity: HashMap<usize, T>,
    square_avg: HashMap<usize, T>,
    grad_avg: HashMap<usize, T>,
}

impl<T: Tensor + TensorOps> RMSprop<T> {
    pub fn new(learning_rate: f64) -> Self {
        Self::with_params(learning_rate, 0.99, 1e-8)
    }
    
    pub fn with_params(learning_rate: f64, alpha: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            alpha,
            epsilon,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            velocity: HashMap::new(),
            square_avg: HashMap::new(),
            grad_avg: HashMap::new(),
        }
    }
}

impl<T: Tensor + TensorOps> Optimizer<T> for RMSprop<T> {
    fn step(&mut self, param_grad_pairs: &mut [(&mut T, &T)]) -> AnyResult<()> {
        // RMSprop algorithm:
        // square_avg = alpha * square_avg + (1 - alpha) * grad^2
        // if centered:
        //   grad_avg = alpha * grad_avg + (1 - alpha) * grad
        //   avg = sqrt(square_avg - grad_avg^2 + eps)
        // else:
        //   avg = sqrt(square_avg + eps)
        // if momentum > 0:
        //   buf = momentum * buf + grad / avg
        //   param = param - lr * buf
        // else:
        //   param = param - lr * grad / avg
        
        for (idx, (param, grad_ref)) in param_grad_pairs.iter_mut().enumerate() {
            // Apply weight decay
            let grad: T = if self.weight_decay > 0.0 {
                let decay = (*param).mul_scalar(self.weight_decay as f32)?;
                (*grad_ref).add(&decay)?
            } else {
                (*grad_ref).clone()
            };
            
            // Get or initialize square average
            let grad_shape = grad.shape().clone();
            let grad_dtype = grad.dtype();
            let grad_device = grad.device();
            
            let square_avg = self.square_avg.entry(idx)
                .or_insert_with(|| {
                    T::from_scalar(0.0, grad_shape.clone(), grad_dtype, grad_device)
                        .unwrap_or_else(|_| grad.clone())
                });
            let square_avg = square_avg.clone();
            
            // Update square average
            // square_avg = alpha * square_avg + (1 - alpha) * grad^2
            let grad_squared = grad.mul(&grad)?;
            let square_avg_scaled = square_avg.mul_scalar(self.alpha as f32)?;
            let grad_sq_scaled = grad_squared.mul_scalar((1.0 - self.alpha) as f32)?;
            let square_avg_new = square_avg_scaled.add(&grad_sq_scaled)?;
            self.square_avg.insert(idx, square_avg_new.clone());
            
            // Compute average
            let avg: T = if self.centered {
                // Get or initialize grad average
                let grad_avg = self.grad_avg.entry(idx)
                    .or_insert_with(|| {
                        T::from_scalar(0.0, grad_shape.clone(), grad_dtype, grad_device)
                            .unwrap_or_else(|_| grad.clone())
                    }).clone();
                
                // Update grad average
                let grad_avg_scaled = grad_avg.mul_scalar(self.alpha as f32)?;
                let grad_scaled = grad.mul_scalar((1.0 - self.alpha) as f32)?;
                let grad_avg_new = grad_avg_scaled.add(&grad_scaled)?;
                self.grad_avg.insert(idx, grad_avg_new.clone());
                
                // avg = sqrt(square_avg - grad_avg^2 + eps)
                let grad_avg_sq = grad_avg_new.mul(&grad_avg_new)?;
                let diff = square_avg_new.sub(&grad_avg_sq)?;
                let diff_eps = diff.add_scalar(self.epsilon as f32)?;
                diff_eps.sqrt()?
            } else {
                // avg = sqrt(square_avg + eps)
                let square_avg_eps = square_avg_new.add_scalar(self.epsilon as f32)?;
                square_avg_eps.sqrt()?
            };
            
            // Compute update
            let update: T = if self.momentum > 0.0 {
                let buf = self.velocity.entry(idx)
                    .or_insert_with(|| {
                        T::from_scalar(0.0, grad.shape().clone(), grad.dtype(), grad.device())
                            .unwrap_or_else(|_| grad.clone())
                    }).clone();
                
                // buf = momentum * buf + grad / avg
                let grad_normalized = grad.div(&avg)?;
                let buf_scaled = buf.mul_scalar(self.momentum as f32)?;
                let buf_new = buf_scaled.add(&grad_normalized)?;
                self.velocity.insert(idx, buf_new.clone());
                buf_new
            } else {
                // update = grad / avg
                grad.div(&avg)?
            };
            
            // Update parameter: param = param - lr * update
            let scaled_update = update.mul_scalar(self.learning_rate as f32)?;
            let current_param = (*param).clone();
            let new_param = current_param.sub(&scaled_update)?;
            unsafe {
                ptr::write(*param, new_param);
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self) -> AnyResult<()> {
        Ok(())
    }
    
    fn name(&self) -> &str {
        "RMSprop"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), self.learning_rate);
        params.insert("alpha".to_string(), self.alpha);
        params
    }
    
    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
    
    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend_cpu::CpuTensor;
    use tensor_core::{Shape, DType, Device};
    use tensor_core::tensor::{TensorRandom, TensorReduce};
    
    fn create_test_param(shape: Shape, device: &Device) -> CpuTensor {
        CpuTensor::random_uniform(shape, -0.1, 0.1, device).unwrap()
    }
    
    fn create_test_grad(shape: Shape, device: &Device) -> CpuTensor {
        CpuTensor::random_uniform(shape, -0.05, 0.05, device).unwrap()
    }
    
    #[test]
    fn test_adam_step() {
        let device = Device::cpu();
        let mut optimizer: Adam<CpuTensor> = Adam::with_params(0.001, 0.9, 0.999, 1e-8);
        
        let mut param = create_test_param(Shape::new(vec![10, 10]), &device);
        let grad = create_test_grad(Shape::new(vec![10, 10]), &device);
        let param_before = param.clone();
        
        let mut param_grad_pairs = vec![(&mut param, &grad)];
        optimizer.step(&mut param_grad_pairs).unwrap();
        
        // Parameter should have changed
        assert_ne!(param.shape().numel(), 0);
        // Check that parameter was updated (values should differ)
        let diff = param.sub(&param_before).unwrap();
        let diff_norm = diff.mul(&diff).unwrap().sum(None, false).unwrap().to_scalar().unwrap();
        assert!(diff_norm > 0.0, "Parameter should be updated by optimizer");
    }
    
    #[test]
    fn test_adamw_step() {
        let device = Device::cpu();
        let mut optimizer: AdamW<CpuTensor> = AdamW::with_params(0.001, 0.9, 0.999, 0.01, 1e-8);
        
        let mut param = create_test_param(Shape::new(vec![5, 5]), &device);
        let grad = create_test_grad(Shape::new(vec![5, 5]), &device);
        
        let mut param_grad_pairs = vec![(&mut param, &grad)];
        optimizer.step(&mut param_grad_pairs).unwrap();
        
        // Should not panic and parameter shape should be preserved
        assert_eq!(param.shape().as_slice(), &[5, 5]);
    }
    
    #[test]
    fn test_sgd_step() {
        let device = Device::cpu();
        let mut optimizer: SGD<CpuTensor> = SGD::with_nesterov(0.01, 0.9, true);
        
        let mut param = create_test_param(Shape::new(vec![8, 8]), &device);
        let grad = create_test_grad(Shape::new(vec![8, 8]), &device);
        
        let mut param_grad_pairs = vec![(&mut param, &grad)];
        optimizer.step(&mut param_grad_pairs).unwrap();
        
        // Should not panic and parameter shape should be preserved
        assert_eq!(param.shape().as_slice(), &[8, 8]);
    }
    
    #[test]
    fn test_rmsprop_step() {
        let device = Device::cpu();
        let mut optimizer: RMSprop<CpuTensor> = RMSprop::with_params(0.001, 0.99, 1e-8);
        
        let mut param = create_test_param(Shape::new(vec![6, 6]), &device);
        let grad = create_test_grad(Shape::new(vec![6, 6]), &device);
        
        let mut param_grad_pairs = vec![(&mut param, &grad)];
        optimizer.step(&mut param_grad_pairs).unwrap();
        
        // Should not panic and parameter shape should be preserved
        assert_eq!(param.shape().as_slice(), &[6, 6]);
    }
    
    #[test]
    fn test_optimizer_zero_grad() {
        let mut optimizer: Adam<CpuTensor> = Adam::new(0.001);
        
        // Should not panic
        optimizer.zero_grad().unwrap();
    }
    
    #[test]
    fn test_optimizer_learning_rate() {
        let mut optimizer: Adam<CpuTensor> = Adam::new(0.001);
        
        assert_eq!(optimizer.get_learning_rate(), 0.001);
        
        optimizer.set_learning_rate(0.0005);
        assert_eq!(optimizer.get_learning_rate(), 0.0005);
    }
    
    #[test]
    fn test_optimizer_parameters() {
        let optimizer: AdamW<CpuTensor> = AdamW::new(0.001, 0.01);
        let params = optimizer.parameters();
        
        assert!(params.contains_key("learning_rate"));
        assert!(params.contains_key("weight_decay"));
    }
}
