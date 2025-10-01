//! 🌀 HelixML Loss Functions
//! 
//! Comprehensive collection of loss functions for different tasks.

use tensor_core::tensor::Tensor;
use hal::{Result, HalError};
use std::collections::HashMap;
use anyhow::Result as AnyResult;

/// Trait for loss functions
pub trait LossFunction: Send + Sync {
    /// Compute loss
    fn compute(&self, predictions: &[Tensor], targets: &[Tensor]) -> AnyResult<Tensor>;
    
    /// Get loss name
    fn name(&self) -> &str;
    
    /// Get loss parameters
    fn parameters(&self) -> HashMap<String, f64>;
}

/// Mean Squared Error loss
pub struct MSELoss {
    /// Reduction type
    reduction: Reduction,
}

/// Reduction type for loss functions
#[derive(Debug, Clone)]
pub enum Reduction {
    Mean,
    Sum,
    None,
}

impl MSELoss {
    /// Create new MSE loss
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl LossFunction for MSELoss {
    fn compute(&self, predictions: &[Tensor], targets: &[Tensor]) -> AnyResult<Tensor> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Predictions and targets must have same length"));
        }
        
        let mut total_loss = 0.0;
        let mut count = 0;
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let diff = pred - target;
            let squared_diff = diff * diff;
            total_loss += squared_diff.sum();
            count += 1;
        }
        
        let loss = match self.reduction {
            Reduction::Mean => total_loss / count as f64,
            Reduction::Sum => total_loss,
            Reduction::None => total_loss,
        };
        
        Ok(Tensor::from(loss))
    }
    
    fn name(&self) -> &str {
        "MSE"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("reduction".to_string(), match self.reduction {
            Reduction::Mean => 0.0,
            Reduction::Sum => 1.0,
            Reduction::None => 2.0,
        });
        params
    }
}

/// Cross Entropy loss
pub struct CrossEntropyLoss {
    /// Reduction type
    reduction: Reduction,
    /// Label smoothing
    label_smoothing: f64,
    /// Weight for classes
    class_weights: Option<Vec<f64>>,
}

impl CrossEntropyLoss {
    /// Create new Cross Entropy loss
    pub fn new(reduction: Reduction) -> Self {
        Self {
            reduction,
            label_smoothing: 0.0,
            class_weights: None,
        }
    }
    
    /// Set label smoothing
    pub fn with_label_smoothing(mut self, smoothing: f64) -> Self {
        self.label_smoothing = smoothing;
        self
    }
    
    /// Set class weights
    pub fn with_class_weights(mut self, weights: Vec<f64>) -> Self {
        self.class_weights = Some(weights);
        self
    }
}

impl LossFunction for CrossEntropyLoss {
    fn compute(&self, predictions: &[Tensor], targets: &[Tensor]) -> AnyResult<Tensor> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Predictions and targets must have same length"));
        }
        
        let mut total_loss = 0.0;
        let mut count = 0;
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            // Apply softmax to predictions
            let softmax_pred = pred.softmax();
            
            // Compute cross entropy
            let log_pred = softmax_pred.log();
            let loss = -(target * log_pred).sum();
            
            // Apply class weights if provided
            let weighted_loss = if let Some(weights) = &self.class_weights {
                // TODO: Apply class weights
                loss
            } else {
                loss
            };
            
            total_loss += weighted_loss;
            count += 1;
        }
        
        let loss = match self.reduction {
            Reduction::Mean => total_loss / count as f64,
            Reduction::Sum => total_loss,
            Reduction::None => total_loss,
        };
        
        Ok(Tensor::from(loss))
    }
    
    fn name(&self) -> &str {
        "CrossEntropy"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("reduction".to_string(), match self.reduction {
            Reduction::Mean => 0.0,
            Reduction::Sum => 1.0,
            Reduction::None => 2.0,
        });
        params.insert("label_smoothing".to_string(), self.label_smoothing);
        params
    }
}

/// Binary Cross Entropy loss
pub struct BCELoss {
    /// Reduction type
    reduction: Reduction,
    /// Label smoothing
    label_smoothing: f64,
}

impl BCELoss {
    /// Create new BCE loss
    pub fn new(reduction: Reduction) -> Self {
        Self {
            reduction,
            label_smoothing: 0.0,
        }
    }
    
    /// Set label smoothing
    pub fn with_label_smoothing(mut self, smoothing: f64) -> Self {
        self.label_smoothing = smoothing;
        self
    }
}

impl LossFunction for BCELoss {
    fn compute(&self, predictions: &[Tensor], targets: &[Tensor]) -> AnyResult<Tensor> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Predictions and targets must have same length"));
        }
        
        let mut total_loss = 0.0;
        let mut count = 0;
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            // Apply sigmoid to predictions
            let sigmoid_pred = pred.sigmoid();
            
            // Compute binary cross entropy
            let loss = -(target * sigmoid_pred.log() + (1.0 - target) * (1.0 - sigmoid_pred).log()).sum();
            
            total_loss += loss;
            count += 1;
        }
        
        let loss = match self.reduction {
            Reduction::Mean => total_loss / count as f64,
            Reduction::Sum => total_loss,
            Reduction::None => total_loss,
        };
        
        Ok(Tensor::from(loss))
    }
    
    fn name(&self) -> &str {
        "BCE"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("reduction".to_string(), match self.reduction {
            Reduction::Mean => 0.0,
            Reduction::Sum => 1.0,
            Reduction::None => 2.0,
        });
        params.insert("label_smoothing".to_string(), self.label_smoothing);
        params
    }
}

/// Focal loss for imbalanced datasets
pub struct FocalLoss {
    /// Alpha parameter
    alpha: f64,
    /// Gamma parameter
    gamma: f64,
    /// Reduction type
    reduction: Reduction,
}

impl FocalLoss {
    /// Create new Focal loss
    pub fn new(alpha: f64, gamma: f64, reduction: Reduction) -> Self {
        Self {
            alpha,
            gamma,
            reduction,
        }
    }
}

impl LossFunction for FocalLoss {
    fn compute(&self, predictions: &[Tensor], targets: &[Tensor]) -> AnyResult<Tensor> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Predictions and targets must have same length"));
        }
        
        let mut total_loss = 0.0;
        let mut count = 0;
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            // Apply softmax to predictions
            let softmax_pred = pred.softmax();
            
            // Compute focal loss
            let ce_loss = -(target * softmax_pred.log()).sum();
            let pt = softmax_pred * target;
            let focal_loss = self.alpha * (1.0 - pt).powf(self.gamma) * ce_loss;
            
            total_loss += focal_loss;
            count += 1;
        }
        
        let loss = match self.reduction {
            Reduction::Mean => total_loss / count as f64,
            Reduction::Sum => total_loss,
            Reduction::None => total_loss,
        };
        
        Ok(Tensor::from(loss))
    }
    
    fn name(&self) -> &str {
        "Focal"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("alpha".to_string(), self.alpha);
        params.insert("gamma".to_string(), self.gamma);
        params.insert("reduction".to_string(), match self.reduction {
            Reduction::Mean => 0.0,
            Reduction::Sum => 1.0,
            Reduction::None => 2.0,
        });
        params
    }
}

/// Huber loss (smooth L1)
pub struct HuberLoss {
    /// Delta parameter
    delta: f64,
    /// Reduction type
    reduction: Reduction,
}

impl HuberLoss {
    /// Create new Huber loss
    pub fn new(delta: f64, reduction: Reduction) -> Self {
        Self { delta, reduction }
    }
}

impl LossFunction for HuberLoss {
    fn compute(&self, predictions: &[Tensor], targets: &[Tensor]) -> AnyResult<Tensor> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Predictions and targets must have same length"));
        }
        
        let mut total_loss = 0.0;
        let mut count = 0;
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let diff = pred - target;
            let abs_diff = diff.abs();
            
            let loss = if abs_diff <= self.delta {
                0.5 * diff * diff
            } else {
                self.delta * (abs_diff - 0.5 * self.delta)
            };
            
            total_loss += loss.sum();
            count += 1;
        }
        
        let loss = match self.reduction {
            Reduction::Mean => total_loss / count as f64,
            Reduction::Sum => total_loss,
            Reduction::None => total_loss,
        };
        
        Ok(Tensor::from(loss))
    }
    
    fn name(&self) -> &str {
        "Huber"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("delta".to_string(), self.delta);
        params.insert("reduction".to_string(), match self.reduction {
            Reduction::Mean => 0.0,
            Reduction::Sum => 1.0,
            Reduction::None => 2.0,
        });
        params
    }
}

/// KL Divergence loss
pub struct KLDivLoss {
    /// Reduction type
    reduction: Reduction,
}

impl KLDivLoss {
    /// Create new KL Divergence loss
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl LossFunction for KLDivLoss {
    fn compute(&self, predictions: &[Tensor], targets: &[Tensor]) -> AnyResult<Tensor> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Predictions and targets must have same length"));
        }
        
        let mut total_loss = 0.0;
        let mut count = 0;
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            // Apply softmax to predictions
            let softmax_pred = pred.softmax();
            let softmax_target = target.softmax();
            
            // Compute KL divergence
            let loss = (softmax_target * (softmax_target / softmax_pred).log()).sum();
            
            total_loss += loss;
            count += 1;
        }
        
        let loss = match self.reduction {
            Reduction::Mean => total_loss / count as f64,
            Reduction::Sum => total_loss,
            Reduction::None => total_loss,
        };
        
        Ok(Tensor::from(loss))
    }
    
    fn name(&self) -> &str {
        "KLDiv"
    }
    
    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("reduction".to_string(), match self.reduction {
            Reduction::Mean => 0.0,
            Reduction::Sum => 1.0,
            Reduction::None => 2.0,
        });
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mse_loss() {
        let mse = MSELoss::new(Reduction::Mean);
        assert_eq!(mse.name(), "MSE");
    }
    
    #[test]
    fn test_cross_entropy_loss() {
        let ce = CrossEntropyLoss::new(Reduction::Mean);
        assert_eq!(ce.name(), "CrossEntropy");
    }
    
    #[test]
    fn test_focal_loss() {
        let focal = FocalLoss::new(0.25, 2.0, Reduction::Mean);
        assert_eq!(focal.name(), "Focal");
    }
}
