//! 🌀 HelixML Learning Rate Schedulers
//! 
//! Advanced learning rate scheduling for optimal training.

use tensor_core::tensor::Tensor;
use std::collections::HashMap;
use anyhow::Result as AnyResult;

/// Trait for learning rate schedulers
pub trait Scheduler<T: Tensor>: Send + Sync {
    /// Step the scheduler
    fn step(&mut self) -> AnyResult<()>;
    
    /// Get current learning rate
    fn get_learning_rate(&self) -> f64;
    
    /// Get scheduler name
    fn name(&self) -> &str;
    
    /// Get scheduler parameters
    fn parameters(&self) -> HashMap<String, f64>;
}

/// Constant learning rate scheduler (no decay)
#[derive(Debug, Clone)]
pub struct ConstantScheduler<T: Tensor> {
    learning_rate: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> ConstantScheduler<T> {
    pub fn new(learning_rate: f64) -> Self {
        Self { 
            learning_rate,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor> Scheduler<T> for ConstantScheduler<T> {
    fn step(&mut self) -> AnyResult<()> {
        Ok(())
    }
    
    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
    
    fn name(&self) -> &str {
        "Constant"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), self.learning_rate);
        params
    }
}

/// Linear learning rate scheduler
#[derive(Debug, Clone)]
pub struct LinearScheduler<T: Tensor> {
    initial_lr: f64,
    final_lr: f64,
    total_steps: usize,
    current_step: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> LinearScheduler<T> {
    pub fn new(initial_lr: f64, final_lr: f64, total_steps: usize) -> Self {
        Self {
            initial_lr,
            final_lr,
            total_steps,
            current_step: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor> Scheduler<T> for LinearScheduler<T> {
    fn step(&mut self) -> AnyResult<()> {
        self.current_step = (self.current_step + 1).min(self.total_steps);
        Ok(())
    }
    
    fn get_learning_rate(&self) -> f64 {
        let progress = self.current_step as f64 / self.total_steps as f64;
        self.initial_lr + (self.final_lr - self.initial_lr) * progress
    }
    
    fn name(&self) -> &str {
        "Linear"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("initial_lr".to_string(), self.initial_lr);
        params.insert("final_lr".to_string(), self.final_lr);
        params.insert("total_steps".to_string(), self.total_steps as f64);
        params
    }
}

/// Exponential learning rate scheduler
#[derive(Debug, Clone)]
pub struct ExponentialScheduler<T: Tensor> {
    initial_lr: f64,
    gamma: f64,
    current_step: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> ExponentialScheduler<T> {
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        Self {
            initial_lr,
            gamma,
            current_step: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor> Scheduler<T> for ExponentialScheduler<T> {
    fn step(&mut self) -> AnyResult<()> {
        self.current_step += 1;
        Ok(())
    }
    
    fn get_learning_rate(&self) -> f64 {
        self.initial_lr * self.gamma.powi(self.current_step as i32)
    }
    
    fn name(&self) -> &str {
        "Exponential"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("initial_lr".to_string(), self.initial_lr);
        params.insert("gamma".to_string(), self.gamma);
        params
    }
}

/// Cosine annealing learning rate scheduler
#[derive(Debug, Clone)]
pub struct CosineAnnealingScheduler<T: Tensor> {
    initial_lr: f64,
    min_lr: f64,
    total_steps: usize,
    current_step: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> CosineAnnealingScheduler<T> {
    pub fn new(initial_lr: f64, min_lr: f64, total_steps: usize) -> Self {
        Self {
            initial_lr,
            min_lr,
            total_steps,
            current_step: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor> Scheduler<T> for CosineAnnealingScheduler<T> {
    fn step(&mut self) -> AnyResult<()> {
        self.current_step = (self.current_step + 1).min(self.total_steps);
        Ok(())
    }
    
    fn get_learning_rate(&self) -> f64 {
        let progress = self.current_step as f64 / self.total_steps as f64;
        let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        self.min_lr + (self.initial_lr - self.min_lr) * cosine
    }
    
    fn name(&self) -> &str {
        "CosineAnnealing"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("initial_lr".to_string(), self.initial_lr);
        params.insert("min_lr".to_string(), self.min_lr);
        params.insert("total_steps".to_string(), self.total_steps as f64);
        params
    }
}
