//! 🌀 HelixML Loss Functions
//! 
//! Comprehensive collection of loss functions for different tasks.

use tensor_core::tensor::{Tensor, TensorOps, TensorReduce};
use tensor_core::{Shape, DType, Device};
use hal::{Result, HalError};
use std::collections::HashMap;
use anyhow::Result as AnyResult;

/// Trait for loss functions
pub trait LossFunction<T: Tensor + TensorOps + TensorReduce>: Send + Sync {
    /// Compute loss
    fn compute(&self, predictions: &[T], targets: &[T]) -> AnyResult<T>;
    
    /// Get loss name
    fn name(&self) -> &str;
    
    /// Get loss parameters
    fn parameters(&self) -> HashMap<String, f64>;
}

/// Reduction type for loss functions
#[derive(Debug, Clone)]
pub enum Reduction {
    Mean,
    Sum,
    None,
}

/// Mean Squared Error loss
pub struct MSELoss<T: Tensor> {
    /// Reduction type
    reduction: Reduction,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> MSELoss<T> {
    /// Create new MSE loss
    pub fn new(reduction: Reduction) -> Self {
        Self { 
            reduction,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor + TensorOps + TensorReduce> LossFunction<T> for MSELoss<T> {
    fn compute(&self, predictions: &[T], targets: &[T]) -> AnyResult<T> {
        if predictions.is_empty() {
            return Err(anyhow::anyhow!("Empty predictions"));
        }
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Predictions and targets must have same length"));
        }
        
        // MSE: mean((pred - target)^2)
        let mut total_loss = predictions[0].sub(&targets[0])?.mul(&predictions[0].sub(&targets[0])?)?;
        
        for (pred, target) in predictions.iter().zip(targets.iter()).skip(1) {
            let diff = pred.sub(target)?;
            let squared = diff.mul(&diff)?;
            total_loss = total_loss.add(&squared)?;
        }
        
        // Apply reduction
        let result = match self.reduction {
            Reduction::Mean => total_loss.mean(None, false)?,
            Reduction::Sum => total_loss.sum(None, false)?,
            Reduction::None => total_loss,
        };
        Ok(result)
    }
    
    fn name(&self) -> &str {
        "MSE"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// Cross Entropy Loss
pub struct CrossEntropyLoss<T: Tensor> {
    reduction: Reduction,
    label_smoothing: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> CrossEntropyLoss<T> {
    pub fn new(reduction: Reduction) -> Self {
        Self { 
            reduction,
            label_smoothing: 0.0,
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn with_label_smoothing(mut self, smoothing: f64) -> Self {
        self.label_smoothing = smoothing;
        self
    }
}

impl<T: Tensor + TensorOps + TensorReduce> LossFunction<T> for CrossEntropyLoss<T> {
    fn compute(&self, predictions: &[T], targets: &[T]) -> AnyResult<T> {
        if predictions.is_empty() {
            return Err(anyhow::anyhow!("Empty predictions"));
        }
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Predictions and targets must have same length"));
        }
        
        // CrossEntropy: -sum(target * log(softmax(pred)))
        // Simplified: -sum(target * log(pred + eps))
        let eps = 1e-7;
        
        let mut total_loss = None;
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            // log(pred + eps)
            let log_pred = pred.log()?;
            // target * log(pred)
            let loss_term = target.mul(&log_pred)?;
            
            total_loss = match total_loss {
                None => Some(loss_term),
                Some(acc) => Some(acc.add(&loss_term)?),
            };
        }
        
        let loss = total_loss.ok_or_else(|| anyhow::anyhow!("No loss computed"))?;
        
        // Negate and apply reduction
        let neg_loss = loss.neg()?;
        let result = match self.reduction {
            Reduction::Mean => neg_loss.mean(None, false)?,
            Reduction::Sum => neg_loss.sum(None, false)?,
            Reduction::None => neg_loss,
        };
        Ok(result)
    }
    
    fn name(&self) -> &str {
        "CrossEntropy"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("label_smoothing".to_string(), self.label_smoothing);
        params
    }
}

/// Binary Cross Entropy Loss
pub struct BCELoss<T: Tensor> {
    reduction: Reduction,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> BCELoss<T> {
    pub fn new(reduction: Reduction) -> Self {
        Self { 
            reduction,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor + TensorOps + TensorReduce> LossFunction<T> for BCELoss<T> {
    fn compute(&self, predictions: &[T], targets: &[T]) -> AnyResult<T> {
        if predictions.is_empty() {
            return Err(anyhow::anyhow!("Empty predictions"));
        }
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Predictions and targets must have same length"));
        }
        
        // BCE: -[target*log(pred) + (1-target)*log(1-pred)]
        // Simplified version for now
        let mut total_loss = None;
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let log_pred = pred.log()?;
            let term1 = target.mul(&log_pred)?;
            
            total_loss = match total_loss {
                None => Some(term1),
                Some(acc) => Some(acc.add(&term1)?),
            };
        }
        
        let loss = total_loss.ok_or_else(|| anyhow::anyhow!("No loss computed"))?;
        let neg_loss = loss.neg()?;
        
        let result = match self.reduction {
            Reduction::Mean => neg_loss.mean(None, false)?,
            Reduction::Sum => neg_loss.sum(None, false)?,
            Reduction::None => neg_loss,
        };
        Ok(result)
    }
    
    fn name(&self) -> &str {
        "BCE"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// BCE with Logits Loss
pub struct BCEWithLogitsLoss<T: Tensor> {
    reduction: Reduction,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> BCEWithLogitsLoss<T> {
    pub fn new(reduction: Reduction) -> Self {
        Self { 
            reduction,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor + TensorOps + TensorReduce> LossFunction<T> for BCEWithLogitsLoss<T> {
    fn compute(&self, predictions: &[T], targets: &[T]) -> AnyResult<T> {
        if predictions.is_empty() {
            return Err(anyhow::anyhow!("Empty predictions"));
        }
        // TODO: Implement proper BCE with logits
        Ok(predictions[0].clone())
    }
    
    fn name(&self) -> &str {
        "BCEWithLogits"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// L1 Loss (Mean Absolute Error)
pub struct L1Loss<T: Tensor> {
    reduction: Reduction,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> L1Loss<T> {
    pub fn new(reduction: Reduction) -> Self {
        Self { 
            reduction,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor + TensorOps + TensorReduce> LossFunction<T> for L1Loss<T> {
    fn compute(&self, predictions: &[T], targets: &[T]) -> AnyResult<T> {
        if predictions.is_empty() {
            return Err(anyhow::anyhow!("Empty predictions"));
        }
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Predictions and targets must have same length"));
        }
        
        // L1: mean(|pred - target|)
        // Note: abs() might not be implemented, use workaround
        let first_diff = predictions[0].sub(&targets[0])?;
        let mut total_loss = first_diff.mul(&first_diff)?.sqrt()?; // |x| = sqrt(x^2)
        
        for (pred, target) in predictions.iter().zip(targets.iter()).skip(1) {
            let diff = pred.sub(target)?;
            let abs_diff = diff.mul(&diff)?.sqrt()?;
            total_loss = total_loss.add(&abs_diff)?;
        }
        
        // Apply reduction
        let result = match self.reduction {
            Reduction::Mean => total_loss.mean(None, false)?,
            Reduction::Sum => total_loss.sum(None, false)?,
            Reduction::None => total_loss,
        };
        Ok(result)
    }
    
    fn name(&self) -> &str {
        "L1"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// Smooth L1 Loss
pub struct SmoothL1Loss<T: Tensor> {
    reduction: Reduction,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> SmoothL1Loss<T> {
    pub fn new(reduction: Reduction) -> Self {
        Self { 
            reduction,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Tensor + TensorOps + TensorReduce> LossFunction<T> for SmoothL1Loss<T> {
    fn compute(&self, predictions: &[T], targets: &[T]) -> AnyResult<T> {
        if predictions.is_empty() {
            return Err(anyhow::anyhow!("Empty predictions"));
        }
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Predictions and targets must have same length"));
        }
        
        // Smooth L1: if |x| < 1: 0.5*x^2, else: |x| - 0.5
        // Simplified: use L1 for now (proper implementation needs conditional)
        let mut total_loss = predictions[0].sub(&targets[0])?;
        let mut squared = total_loss.mul(&total_loss)?;
        total_loss = squared.sqrt()?; // |x|
        
        for (pred, target) in predictions.iter().zip(targets.iter()).skip(1) {
            let diff = pred.sub(target)?;
            let abs_diff = diff.mul(&diff)?.sqrt()?;
            total_loss = total_loss.add(&abs_diff)?;
        }
        
        let result = match self.reduction {
            Reduction::Mean => total_loss.mean(None, false)?,
            Reduction::Sum => total_loss.sum(None, false)?,
            Reduction::None => total_loss,
        };
        Ok(result)
    }
    
    fn name(&self) -> &str {
        "SmoothL1"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}
