//! 🌀 HelixML Optimizers
//! 
//! Optimization algorithms for SSM/Hyena architectures.

use tensor_core::{Tensor, Device, Result, TensorError, Shape, DType};
use tensor_core::tensor::{TensorOps, TensorRandom};
use std::collections::HashMap;

/// Base trait for all optimizers
pub trait Optimizer<T: Tensor> {
    fn step(&mut self, parameters: &mut [&mut T], gradients: &[&T]) -> Result<()>;
    fn zero_grad(&mut self);
    fn lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}

/// AdamW optimizer
#[derive(Debug, Clone)]
pub struct AdamW<T: Tensor> {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: usize,
    m: HashMap<usize, T>, // First moment estimates
    v: HashMap<usize, T>, // Second moment estimates
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom> AdamW<T> {
    pub fn new(lr: f32, device: &Device) -> Self {
        Self::new_with_params(lr, 0.9, 0.999, 1e-8, 0.01, device)
    }
    
    pub fn new_with_params(
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        device: &Device,
    ) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            device: device.clone(),
        }
    }
    
    pub fn beta1(&self) -> f32 {
        self.beta1
    }
    
    pub fn beta2(&self) -> f32 {
        self.beta2
    }
    
    pub fn eps(&self) -> f32 {
        self.eps
    }
    
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }
    
    pub fn step_count(&self) -> usize {
        self.step
    }
}

impl<T: Tensor + TensorOps + TensorRandom> Optimizer<T> for AdamW<T> {
    fn step(&mut self, parameters: &mut [&mut T], gradients: &[&T]) -> Result<()> {
        if parameters.len() != gradients.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![parameters.len()],
                actual: vec![gradients.len()],
            });
        }
        
        self.step += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);
        
        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            // Apply weight decay
            let grad_with_decay = if self.weight_decay > 0.0 {
                let decay_term = param.mul_scalar(self.weight_decay)?;
                grad.add(&decay_term)?
            } else {
                (*grad).clone()
            };
            
            // Initialize moment estimates if not present
            if !self.m.contains_key(&i) {
                let m_init = T::zeros(grad.shape().clone(), grad.dtype(), grad.device())?;
                let v_init = T::zeros(grad.shape().clone(), grad.dtype(), grad.device())?;
                self.m.insert(i, m_init);
                self.v.insert(i, v_init);
            }
            
            // Get moment estimates
            let m = self.m.get_mut(&i).unwrap();
            let v = self.v.get_mut(&i).unwrap();
            
            // Update biased first moment estimate
            let m_new = m.mul_scalar(self.beta1)?
                .add(&grad_with_decay.mul_scalar(1.0 - self.beta1)?)?;
            *m = m_new;
            
            // Update biased second raw moment estimate
            let v_new = v.mul_scalar(self.beta2)?
                .add(&grad_with_decay.mul(&grad_with_decay)?.mul_scalar(1.0 - self.beta2)?)?;
            *v = v_new;
            
            // Compute bias-corrected first moment estimate
            let m_hat = m.mul_scalar(1.0 / bias_correction1)?;
            
            // Compute bias-corrected second raw moment estimate
            let v_hat = v.mul_scalar(1.0 / bias_correction2)?;
            
            // Update parameters
            let eps_tensor = T::from_scalar(self.eps, v_hat.shape().clone(), DType::F32, &v_hat.device())?;
            let update = m_hat.div(&v_hat.sqrt()?.add(&eps_tensor)?)?;
            let scaled_update = update.mul_scalar(-self.lr)?;
            let new_param = param.add(&scaled_update)?;
            **param = new_param;
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        self.m.clear();
        self.v.clear();
        self.step = 0;
    }
    
    fn lr(&self) -> f32 {
        self.lr
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Lion optimizer (EvoLved Sign Momentum)
#[derive(Debug, Clone)]
pub struct Lion<T: Tensor> {
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    m: HashMap<usize, T>, // Momentum
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom> Lion<T> {
    pub fn new(lr: f32, device: &Device) -> Self {
        Self::new_with_params(lr, 0.9, 0.99, 0.01, device)
    }
    
    pub fn new_with_params(
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        device: &Device,
    ) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            weight_decay,
            m: HashMap::new(),
            device: device.clone(),
        }
    }
    
    pub fn beta1(&self) -> f32 {
        self.beta1
    }
    
    pub fn beta2(&self) -> f32 {
        self.beta2
    }
    
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }
}

impl<T: Tensor + TensorOps + TensorRandom> Optimizer<T> for Lion<T> {
    fn step(&mut self, parameters: &mut [&mut T], gradients: &[&T]) -> Result<()> {
        if parameters.len() != gradients.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![parameters.len()],
                actual: vec![gradients.len()],
            });
        }
        
        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            // Apply weight decay
            let grad_with_decay = if self.weight_decay > 0.0 {
                let decay_term = param.mul_scalar(self.weight_decay)?;
                grad.add(&decay_term)?
            } else {
                (*grad).clone()
            };
            
            // Initialize momentum if not present
            if !self.m.contains_key(&i) {
                let m_init = T::zeros(grad.shape().clone(), grad.dtype(), grad.device())?;
                self.m.insert(i, m_init);
            }
            
            // Get momentum
            let m = self.m.get_mut(&i).unwrap();
            
            // Update momentum
            let m_new = m.mul_scalar(self.beta1)?
                .add(&grad_with_decay.mul_scalar(1.0 - self.beta1)?)?;
            *m = m_new;
            
            // Compute update direction using sign
            let update_dir = m.sign()?;
            
            // Update parameters
            let update = update_dir.mul_scalar(-self.lr)?;
            let new_param = param.add(&update)?;
            **param = new_param;
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        self.m.clear();
    }
    
    fn lr(&self) -> f32 {
        self.lr
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// SGD optimizer
#[derive(Debug, Clone)]
pub struct SGD<T: Tensor> {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    m: HashMap<usize, T>, // Momentum
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom> SGD<T> {
    pub fn new(lr: f32, device: &Device) -> Self {
        Self::new_with_params(lr, 0.0, 0.0, device)
    }
    
    pub fn new_with_momentum(lr: f32, momentum: f32, device: &Device) -> Self {
        Self::new_with_params(lr, momentum, 0.0, device)
    }
    
    pub fn new_with_params(
        lr: f32,
        momentum: f32,
        weight_decay: f32,
        device: &Device,
    ) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            m: HashMap::new(),
            device: device.clone(),
        }
    }
    
    pub fn momentum(&self) -> f32 {
        self.momentum
    }
    
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }
}

impl<T: Tensor + TensorOps + TensorRandom> Optimizer<T> for SGD<T> {
    fn step(&mut self, parameters: &mut [&mut T], gradients: &[&T]) -> Result<()> {
        if parameters.len() != gradients.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![parameters.len()],
                actual: vec![gradients.len()],
            });
        }
        
        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            // Apply weight decay
            let grad_with_decay = if self.weight_decay > 0.0 {
                let decay_term = param.mul_scalar(self.weight_decay)?;
                grad.add(&decay_term)?
            } else {
                (*grad).clone()
            };
            
            let update = if self.momentum > 0.0 {
                // Initialize momentum if not present
                if !self.m.contains_key(&i) {
                    let m_init = T::zeros(grad.shape().clone(), grad.dtype(), grad.device())?;
                    self.m.insert(i, m_init);
                }
                
                // Get momentum
                let m = self.m.get_mut(&i).unwrap();
                
                // Update momentum
                let m_new = m.mul_scalar(self.momentum)?
                    .add(&grad_with_decay)?;
                *m = m_new.clone();
                
                // Use momentum for update
                m_new.mul_scalar(-self.lr)?
            } else {
                // No momentum
                grad_with_decay.mul_scalar(-self.lr)?
            };
            
            // Update parameters
            let new_param = param.add(&update)?;
            **param = new_param;
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        self.m.clear();
    }
    
    fn lr(&self) -> f32 {
        self.lr
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Learning rate scheduler
pub trait LRScheduler {
    fn step(&mut self, epoch: usize);
    fn get_lr(&self) -> f32;
}

/// Step learning rate scheduler
#[derive(Debug, Clone)]
pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
    current_lr: f32,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_lr: initial_lr,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, epoch: usize) {
        if epoch > 0 && epoch % self.step_size == 0 {
            self.current_lr *= self.gamma;
        }
    }
    
    fn get_lr(&self) -> f32 {
        self.current_lr
    }
}

/// Exponential learning rate scheduler
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    initial_lr: f32,
    gamma: f32,
    current_lr: f32,
}

impl ExponentialLR {
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self {
            initial_lr,
            gamma,
            current_lr: initial_lr,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self, epoch: usize) {
        self.current_lr = self.initial_lr * self.gamma.powi(epoch as i32);
    }
    
    fn get_lr(&self) -> f32 {
        self.current_lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend_cpu::CpuTensor;
    
    #[test]
    fn test_adamw_optimizer() {
        let device = Device::cpu();
        let mut optimizer = AdamW::<CpuTensor>::new(0.001, &device);
        
        let mut param = CpuTensor::random_uniform(Shape::new(vec![2, 3]), 0.0, 1.0, &device).unwrap();
        let grad = CpuTensor::random_uniform(Shape::new(vec![2, 3]), -1.0, 1.0, &device).unwrap();
        
        let mut params = vec![&mut param];
        let grads = vec![&grad];
        
        // Should not panic
        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(optimizer.step_count(), 1);
    }
    
    #[test]
    fn test_sgd_optimizer() {
        let device = Device::cpu();
        let mut optimizer = SGD::<CpuTensor>::new_with_momentum(0.01, 0.9, &device);
        
        let mut param = CpuTensor::random_uniform(Shape::new(vec![2, 3]), 0.0, 1.0, &device).unwrap();
        let grad = CpuTensor::random_uniform(Shape::new(vec![2, 3]), -1.0, 1.0, &device).unwrap();
        
        let mut params = vec![&mut param];
        let grads = vec![&grad];
        
        // Should not panic
        optimizer.step(&mut params, &grads).unwrap();
    }
    
    #[test]
    fn test_lr_scheduler() {
        let mut scheduler = StepLR::new(0.1, 10, 0.5);
        
        assert_eq!(scheduler.get_lr(), 0.1);
        
        scheduler.step(10);
        assert_eq!(scheduler.get_lr(), 0.05);
        
        scheduler.step(20);
        assert_eq!(scheduler.get_lr(), 0.025);
    }
}