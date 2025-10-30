//! 🌀 VortexGrad - Gradient Memory and Resonance Amplification
//!
//! VortexGrad extends traditional backprop with:
//! - **Gradient History**: Remembers past gradient flows
//! - **Resonance Detection**: Identifies stable gradient patterns
//! - **Adaptive Amplification**: Boosts resonant weights, dampens noise

use std::collections::{HashMap, VecDeque};
use tensor_core::{Tensor, Result};
use serde::{Serialize, Deserialize};

/// Configuration for VortexGrad
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VortexConfig {
    /// Number of historical gradients to store
    pub history_size: usize,
    /// Resonance detection threshold (0.0 - 1.0)
    pub resonance_threshold: f32,
    /// Amplification factor for resonant gradients
    pub amplification_factor: f32,
    /// Damping factor for noisy gradients
    pub damping_factor: f32,
    /// Cycle detection window
    pub cycle_window: usize,
}

impl Default for VortexConfig {
    fn default() -> Self {
        Self {
            history_size: 10,
            resonance_threshold: 0.7,
            amplification_factor: 1.5,
            damping_factor: 0.5,
            cycle_window: 5,
        }
    }
}

/// Gradient history for a single parameter
#[derive(Debug, Clone)]
pub struct GradientHistory<T: Tensor> {
    /// Historical gradients (most recent last)
    history: VecDeque<T>,
    /// Maximum history size
    max_size: usize,
    /// Vortex cycle counter
    cycle: usize,
}

impl<T: Tensor> GradientHistory<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_size),
            max_size,
            cycle: 0,
        }
    }
    
    /// Add new gradient to history
    pub fn push(&mut self, gradient: T) {
        if self.history.len() >= self.max_size {
            self.history.pop_front();
        }
        self.history.push_back(gradient);
        self.cycle += 1;
    }
    
    /// Get the current gradient (most recent)
    pub fn current(&self) -> Option<&T> {
        self.history.back()
    }
    
    /// Get all historical gradients
    pub fn all(&self) -> &VecDeque<T> {
        &self.history
    }
    
    /// Get gradient at specific cycle
    pub fn at_cycle(&self, cycle: usize) -> Option<&T> {
        if cycle >= self.cycle || cycle + self.max_size < self.cycle {
            return None;
        }
        let index = self.max_size.saturating_sub(self.cycle - cycle);
        self.history.get(index)
    }
}

/// Resonance weight for gradient amplification/damping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceWeight {
    /// Current resonance score (0.0 - 1.0)
    pub score: f32,
    /// Amplification multiplier
    pub multiplier: f32,
    /// Is this weight resonating?
    pub is_resonant: bool,
    /// Pattern type detected
    pub pattern: ResonancePattern,
}

/// Types of resonance patterns
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ResonancePattern {
    /// Stable gradient direction
    Stable,
    /// Oscillating gradients
    Oscillating,
    /// Exploding gradients
    Exploding,
    /// Vanishing gradients
    Vanishing,
    /// Chaotic/noisy
    Chaotic,
    /// Unknown pattern
    Unknown,
}

/// VortexGrad engine
pub struct VortexGrad<T: Tensor> {
    config: VortexConfig,
    /// Gradient histories per parameter
    histories: HashMap<usize, GradientHistory<T>>,
    /// Resonance weights per parameter
    resonance_weights: HashMap<usize, ResonanceWeight>,
}

impl<T: Tensor + tensor_core::tensor::TensorOps> VortexGrad<T> {
    pub fn new(config: VortexConfig) -> Self {
        Self {
            config,
            histories: HashMap::new(),
            resonance_weights: HashMap::new(),
        }
    }
    
    /// Process gradient with vortex memory
    pub fn process_gradient(&mut self, param_id: usize, gradient: T) -> Result<T> {
        // Compute resonance with history first (before mutably borrowing)
        let prev_grads = self.get_recent_gradients(param_id);
        let resonance = if let Some(prev_grads) = prev_grads {
            self.compute_resonance(&gradient, &prev_grads)?
        } else {
            ResonanceWeight {
                score: 0.5,
                multiplier: 1.0,
                is_resonant: false,
                pattern: ResonancePattern::Unknown,
            }
        };
        
        // Now safely mutate
        let history = self.histories
            .entry(param_id)
            .or_insert_with(|| GradientHistory::new(self.config.history_size));
        
        history.push(gradient.clone());
        
        // Update resonance weights
        self.resonance_weights.insert(param_id, resonance.clone());
        
        // Apply resonance amplification/damping
        self.apply_resonance(&gradient, &resonance)
    }
    
    /// Compute resonance between current and historical gradients
    fn compute_resonance(&self, current: &T, history: &[T]) -> Result<ResonanceWeight> {
        if history.is_empty() {
            return Ok(ResonanceWeight {
                score: 0.5,
                multiplier: 1.0,
                is_resonant: false,
                pattern: ResonancePattern::Unknown,
            });
        }
        
        // Compute cosine similarity with recent history
        let mut similarities = Vec::new();
        for hist_grad in history.iter().rev().take(self.config.cycle_window) {
            let sim = self.cosine_similarity(current, hist_grad)?;
            similarities.push(sim);
        }
        
        let avg_similarity = similarities.iter().sum::<f32>() / similarities.len() as f32;
        
        // Detect pattern
        let pattern = self.detect_pattern(current, history)?;
        
        // Determine if resonant
        let is_resonant = avg_similarity > self.config.resonance_threshold;
        
        // Calculate multiplier
        let multiplier = if is_resonant {
            self.config.amplification_factor
        } else {
            self.config.damping_factor
        };
        
        Ok(ResonanceWeight {
            score: avg_similarity,
            multiplier,
            is_resonant,
            pattern,
        })
    }
    
    /// Compute cosine similarity between two gradients
    fn cosine_similarity(&self, _a: &T, _b: &T) -> Result<f32> {
        // TODO: Implement proper cosine similarity when we have TensorReduce trait
        // dot(a, b) / (norm(a) * norm(b))
        // let dot = a.mul(b)?.sum(None, false)?;
        // let norm_a = a.mul(a)?.sum(None, false)?.sqrt()?;
        // let norm_b = b.mul(b)?.sum(None, false)?.sqrt()?;
        
        // For now, return placeholder
        Ok(0.5)
    }
    
    /// Detect resonance pattern
    fn detect_pattern(&self, _current: &T, history: &[T]) -> Result<ResonancePattern> {
        if history.len() < 3 {
            return Ok(ResonancePattern::Unknown);
        }
        
        // TODO: Implement pattern detection
        // - Check for oscillations
        // - Check for explosion/vanishing
        // - Check for stability
        
        Ok(ResonancePattern::Stable)
    }
    
    /// Apply resonance amplification/damping to gradient
    fn apply_resonance(&self, gradient: &T, resonance: &ResonanceWeight) -> Result<T> {
        gradient.mul_scalar(resonance.multiplier)
    }
    
    /// Get recent gradients for a parameter
    fn get_recent_gradients(&self, param_id: usize) -> Option<Vec<T>> {
        self.histories.get(&param_id).map(|h| {
            h.all().iter().cloned().collect()
        })
    }
    
    /// Get resonance statistics
    pub fn stats(&self) -> VortexStats {
        let total_params = self.resonance_weights.len();
        let resonant_params = self.resonance_weights
            .values()
            .filter(|w| w.is_resonant)
            .count();
        
        let avg_resonance = if total_params > 0 {
            self.resonance_weights
                .values()
                .map(|w| w.score)
                .sum::<f32>() / total_params as f32
        } else {
            0.0
        };
        
        VortexStats {
            total_params,
            resonant_params,
            avg_resonance,
            patterns: self.collect_pattern_stats(),
        }
    }
    
    fn collect_pattern_stats(&self) -> HashMap<ResonancePattern, usize> {
        let mut stats = HashMap::new();
        for weight in self.resonance_weights.values() {
            *stats.entry(weight.pattern).or_insert(0) += 1;
        }
        stats
    }
}

/// VortexGrad statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VortexStats {
    pub total_params: usize,
    pub resonant_params: usize,
    pub avg_resonance: f32,
    pub patterns: HashMap<ResonancePattern, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend_cpu::CpuTensor;
    use tensor_core::{Shape, DType, Device, Tensor};
    use tensor_core::tensor::TensorRandom;
    
    #[test]
    fn test_vortex_basic() {
        let config = VortexConfig::default();
        let mut vortex = VortexGrad::<CpuTensor>::new(config);
        
        let grad = CpuTensor::ones(Shape::new(vec![10]), DType::F32, &Device::cpu()).unwrap();
        let result = vortex.process_gradient(0, grad).unwrap();
        
        assert!(result.shape().numel() == 10);
    }
    
    #[test]
    fn test_gradient_history() {
        let mut history = GradientHistory::<CpuTensor>::new(5);
        
        for _i in 0..10 {
            let grad = CpuTensor::ones(Shape::new(vec![10]), DType::F32, &Device::cpu()).unwrap();
            history.push(grad);
        }
        
        assert_eq!(history.all().len(), 5); // Should keep only last 5
    }
}

