//! 🌀 HelixML Optimizers
//! 
//! Advanced optimizers with adaptive learning rates and momentum.

use tensor_core::tensor::Tensor;
use hal::{Result, HalError};
use std::collections::HashMap;
use anyhow::Result as AnyResult;

/// Trait for optimizers
pub trait Optimizer: Send + Sync {
    /// Update parameters
    fn step(&self, gradients: &[Tensor]) -> AnyResult<()>;
    
    /// Zero gradients
    fn zero_grad(&self) -> AnyResult<()>;
    
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
pub struct Adam {
    /// Learning rate
    learning_rate: f64,
    /// Beta1 parameter
    beta1: f64,
    /// Beta2 parameter
    beta2: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
    /// Weight decay
    weight_decay: f64,
    /// First moment estimates
    first_moments: HashMap<usize, Tensor>,
    /// Second moment estimates
    second_moments: HashMap<usize, Tensor>,
    /// Step counter
    step: usize,
}

impl Adam {
    /// Create new Adam optimizer
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            step: 0,
        }
    }
    
    /// Set beta1 parameter
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }
    
    /// Set beta2 parameter
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }
    
    /// Set epsilon
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
    
    /// Set weight decay
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Adam {
    fn step(&self, gradients: &[Tensor]) -> AnyResult<()> {
        self.step += 1;
        
        for (i, grad) in gradients.iter().enumerate() {
            // Apply weight decay
            let grad = if self.weight_decay > 0.0 {
                grad + self.weight_decay * grad
            } else {
                grad.clone()
            };
            
            // Update first moment estimate
            let first_moment = if let Some(moment) = self.first_moments.get(&i) {
                self.beta1 * moment + (1.0 - self.beta1) * grad
            } else {
                (1.0 - self.beta1) * grad
            };
            
            // Update second moment estimate
            let second_moment = if let Some(moment) = self.second_moments.get(&i) {
                self.beta2 * moment + (1.0 - self.beta2) * (grad * grad)
            } else {
                (1.0 - self.beta2) * (grad * grad)
            };
            
            // Bias correction
            let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);
            
            let first_moment_corrected = first_moment / bias_correction1;
            let second_moment_corrected = second_moment / bias_correction2;
            
            // Update parameters
            let update = first_moment_corrected / (second_moment_corrected.sqrt() + self.epsilon);
            // TODO: Apply update to parameters
        }
        
        Ok(())
    }
    
    fn zero_grad(&self) -> AnyResult<()> {
        // TODO: Clear gradients
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
        params.insert("epsilon".to_string(), self.epsilon);
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

/// AdamW optimizer (Adam with decoupled weight decay)
pub struct AdamW {
    /// Learning rate
    learning_rate: f64,
    /// Beta1 parameter
    beta1: f64,
    /// Beta2 parameter
    beta2: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
    /// Weight decay
    weight_decay: f64,
    /// First moment estimates
    first_moments: HashMap<usize, Tensor>,
    /// Second moment estimates
    second_moments: HashMap<usize, Tensor>,
    /// Step counter
    step: usize,
}

impl AdamW {
    /// Create new AdamW optimizer
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            step: 0,
        }
    }
    
    /// Set weight decay
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&self, gradients: &[Tensor]) -> AnyResult<()> {
        self.step += 1;
        
        for (i, grad) in gradients.iter().enumerate() {
            // Update first moment estimate
            let first_moment = if let Some(moment) = self.first_moments.get(&i) {
                self.beta1 * moment + (1.0 - self.beta1) * grad
            } else {
                (1.0 - self.beta1) * grad
            };
            
            // Update second moment estimate
            let second_moment = if let Some(moment) = self.second_moments.get(&i) {
                self.beta2 * moment + (1.0 - self.beta2) * (grad * grad)
            } else {
                (1.0 - self.beta2) * (grad * grad)
            };
            
            // Bias correction
            let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);
            
            let first_moment_corrected = first_moment / bias_correction1;
            let second_moment_corrected = second_moment / bias_correction2;
            
            // Update parameters with weight decay
            let update = first_moment_corrected / (second_moment_corrected.sqrt() + self.epsilon);
            // TODO: Apply update to parameters with weight decay
        }
        
        Ok(())
    }
    
    fn zero_grad(&self) -> AnyResult<()> {
        // TODO: Clear gradients
        Ok(())
    }
    
    fn name(&self) -> &str {
        "AdamW"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), self.learning_rate);
        params.insert("beta1".to_string(), self.beta1);
        params.insert("beta2".to_string(), self.beta2);
        params.insert("epsilon".to_string(), self.epsilon);
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
pub struct SGD {
    /// Learning rate
    learning_rate: f64,
    /// Momentum
    momentum: f64,
    /// Weight decay
    weight_decay: f64,
    /// Dampening
    dampening: f64,
    /// Nesterov momentum
    nesterov: bool,
    /// Velocity
    velocity: HashMap<usize, Tensor>,
}

impl SGD {
    /// Create new SGD optimizer
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocity: HashMap::new(),
        }
    }
    
    /// Set momentum
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }
    
    /// Set weight decay
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    
    /// Set Nesterov momentum
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for SGD {
    fn step(&self, gradients: &[Tensor]) -> AnyResult<()> {
        for (i, grad) in gradients.iter().enumerate() {
            // Apply weight decay
            let grad = if self.weight_decay > 0.0 {
                grad + self.weight_decay * grad
            } else {
                grad.clone()
            };
            
            // Update velocity
            let velocity = if let Some(vel) = self.velocity.get(&i) {
                self.momentum * vel + (1.0 - self.dampening) * grad
            } else {
                grad.clone()
            };
            
            // Update parameters
            let update = if self.nesterov {
                grad + self.momentum * velocity
            } else {
                velocity
            };
            
            // TODO: Apply update to parameters
        }
        
        Ok(())
    }
    
    fn zero_grad(&self) -> AnyResult<()> {
        // TODO: Clear gradients
        Ok(())
    }
    
    fn name(&self) -> &str {
        "SGD"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), self.learning_rate);
        params.insert("momentum".to_string(), self.momentum);
        params.insert("weight_decay".to_string(), self.weight_decay);
        params.insert("dampening".to_string(), self.dampening);
        params.insert("nesterov".to_string(), if self.nesterov { 1.0 } else { 0.0 });
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
pub struct RMSprop {
    /// Learning rate
    learning_rate: f64,
    /// Alpha parameter
    alpha: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
    /// Weight decay
    weight_decay: f64,
    /// Momentum
    momentum: f64,
    /// Centered
    centered: bool,
    /// Square gradient estimates
    square_grads: HashMap<usize, Tensor>,
    /// Velocity
    velocity: HashMap<usize, Tensor>,
}

impl RMSprop {
    /// Create new RMSprop optimizer
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            alpha: 0.99,
            epsilon: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            square_grads: HashMap::new(),
            velocity: HashMap::new(),
        }
    }
    
    /// Set alpha parameter
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
    
    /// Set momentum
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }
    
    /// Set centered
    pub fn with_centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }
}

impl Optimizer for RMSprop {
    fn step(&self, gradients: &[Tensor]) -> AnyResult<()> {
        for (i, grad) in gradients.iter().enumerate() {
            // Apply weight decay
            let grad = if self.weight_decay > 0.0 {
                grad + self.weight_decay * grad
            } else {
                grad.clone()
            };
            
            // Update square gradient estimate
            let square_grad = if let Some(sq_grad) = self.square_grads.get(&i) {
                self.alpha * sq_grad + (1.0 - self.alpha) * (grad * grad)
            } else {
                grad * grad
            };
            
            // Update velocity
            let velocity = if let Some(vel) = self.velocity.get(&i) {
                self.momentum * vel + self.learning_rate * grad / (square_grad.sqrt() + self.epsilon)
            } else {
                self.learning_rate * grad / (square_grad.sqrt() + self.epsilon)
            };
            
            // Update parameters
            // TODO: Apply update to parameters
        }
        
        Ok(())
    }
    
    fn zero_grad(&self) -> AnyResult<()> {
        // TODO: Clear gradients
        Ok(())
    }
    
    fn name(&self) -> &str {
        "RMSprop"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), self.learning_rate);
        params.insert("alpha".to_string(), self.alpha);
        params.insert("epsilon".to_string(), self.epsilon);
        params.insert("weight_decay".to_string(), self.weight_decay);
        params.insert("momentum".to_string(), self.momentum);
        params.insert("centered".to_string(), if self.centered { 1.0 } else { 0.0 });
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
    
    #[test]
    fn test_adam_creation() {
        let adam = Adam::new(0.001);
        assert_eq!(adam.name(), "Adam");
        assert_eq!(adam.get_learning_rate(), 0.001);
    }
    
    #[test]
    fn test_sgd_creation() {
        let sgd = SGD::new(0.01).with_momentum(0.9);
        assert_eq!(sgd.name(), "SGD");
        assert_eq!(sgd.get_learning_rate(), 0.01);
    }
    
    #[test]
    fn test_rmsprop_creation() {
        let rmsprop = RMSprop::new(0.001).with_alpha(0.99);
        assert_eq!(rmsprop.name(), "RMSprop");
        assert_eq!(rmsprop.get_learning_rate(), 0.001);
    }
}
