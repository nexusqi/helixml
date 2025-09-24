//! M0 - Motif Detection
//! 
//! Detection of short patterns in sequences using various algorithms.

use tensor_core::{Tensor, Shape, DType, Device, Result};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision};
use std::collections::HashMap;
use super::{Motif, MemoryStats};

/// Motif detector for M0 level memory
#[derive(Debug, Clone)]
pub struct MotifDetector<T: Tensor> {
    // Pattern matching weights
    pattern_weights: T,
    similarity_threshold: f32,
    
    // Motif storage
    motifs: Vec<Motif<T>>,
    motif_frequency: HashMap<Vec<f32>, usize>,
    
    // Configuration
    max_motif_length: usize,
    min_frequency: usize,
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision> MotifDetector<T> {
    pub fn new(d_model: usize, max_motif_length: usize, device: &Device) -> Result<Self> {
        let pattern_weights = T::random_normal(
            Shape::new(vec![max_motif_length, d_model]),
            0.0,
            0.1,
            device,
        )?;
        
        Ok(Self {
            pattern_weights,
            similarity_threshold: 0.8,
            motifs: Vec::new(),
            motif_frequency: HashMap::new(),
            max_motif_length,
            min_frequency: 2,
            device: device.clone(),
        })
    }
    
    /// Detect motifs in a sequence
    pub fn detect_motifs(&mut self, sequence: &T) -> Result<Vec<Motif<T>>> {
        let mut detected_motifs = Vec::new();
        let seq_len = sequence.shape().dim(0).unwrap();
        
        // Sliding window approach for motif detection
        for window_size in 1..=self.max_motif_length {
            for start in 0..=(seq_len - window_size) {
                // Extract pattern
                let pattern = self.extract_pattern(sequence, start, window_size)?;
                
                // Check similarity with existing motifs
                let similarity = self.calculate_similarity(&pattern, &self.motifs)?;
                
                if similarity > self.similarity_threshold {
                    // Find most similar existing motif
                    let mut best_match_idx = None;
                    let mut best_similarity = 0.0;
                    
                    for (idx, existing_motif) in self.motifs.iter().enumerate() {
                        let sim = self.calculate_pattern_similarity(&pattern, &existing_motif.pattern)?;
                        if sim > best_similarity {
                            best_similarity = sim;
                            best_match_idx = Some(idx);
                        }
                    }
                    
                    if let Some(idx) = best_match_idx {
                        // Update existing motif
                        self.motifs[idx].frequency += 1.0;
                        self.motifs[idx].stability = self.calculate_stability(&self.motifs[idx])?;
                    } else {
                        // Create new motif
                        let motif = Motif {
                            pattern: pattern.clone(),
                            frequency: 1.0,
                            stability: self.calculate_initial_stability(&pattern)?,
                            position: start,
                        };
                        detected_motifs.push(motif);
                    }
                }
            }
        }
        
        // Filter by minimum frequency
        detected_motifs.retain(|m| m.frequency >= self.min_frequency as f32);
        
        Ok(detected_motifs)
    }
    
    /// Extract pattern from sequence
    fn extract_pattern(&self, sequence: &T, start: usize, length: usize) -> Result<T> {
        // For now, simplified extraction
        // In practice, would use proper slicing
        let pattern_shape = Shape::new(vec![length, sequence.shape().dim(1).unwrap()]);
        let pattern = T::random_normal(pattern_shape, 0.0, 0.1, &self.device)?;
        Ok(pattern)
    }
    
    /// Calculate similarity between patterns
    fn calculate_pattern_similarity(&self, pattern1: &T, pattern2: &T) -> Result<f32> {
        // Simplified cosine similarity
        // In practice, would use proper tensor operations
        Ok(0.85) // Placeholder
    }
    
    /// Calculate similarity with existing motifs
    fn calculate_similarity(&self, pattern: &T, motifs: &[Motif<T>]) -> Result<f32> {
        if motifs.is_empty() {
            return Ok(0.0);
        }
        
        let mut max_similarity: f32 = 0.0;
        for motif in motifs {
            let similarity = self.calculate_pattern_similarity(pattern, &motif.pattern)?;
            max_similarity = max_similarity.max(similarity);
        }
        
        Ok(max_similarity)
    }
    
    /// Calculate motif stability
    fn calculate_stability(&self, motif: &Motif<T>) -> Result<f32> {
        // Stability based on frequency and pattern consistency
        let frequency_factor = motif.frequency / 10.0; // Normalize frequency
        let consistency_factor = 0.8; // Placeholder for pattern consistency
        
        Ok(frequency_factor * consistency_factor)
    }
    
    /// Calculate initial stability for new motif
    fn calculate_initial_stability(&self, pattern: &T) -> Result<f32> {
        // Initial stability for new patterns
        Ok(0.5)
    }
    
    /// Retrieve information by motifs
    pub fn retrieve_by_motifs(&self, query: &T) -> Result<T> {
        // Find most similar motifs
        let similarities = self.calculate_similarity(query, &self.motifs)?;
        
        if similarities > 0.7 {
            // Return most relevant motif pattern
            if let Some(best_motif) = self.motifs.iter().max_by(|a, b| {
                a.stability.partial_cmp(&b.stability).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                return Ok(best_motif.pattern.clone());
            }
        }
        
        // Return original query if no similar motifs found
        Ok(query.clone())
    }
    
    /// Update motifs with new data
    pub fn update_motifs(&mut self, new_motifs: &[Motif<T>]) -> Result<()> {
        for new_motif in new_motifs {
            // Find most similar existing motif
            let mut best_match_idx = None;
            let mut best_similarity = 0.0;
            
            for (idx, existing_motif) in self.motifs.iter().enumerate() {
                let sim = self.calculate_pattern_similarity(&new_motif.pattern, &existing_motif.pattern)?;
                if sim > best_similarity {
                    best_similarity = sim;
                    best_match_idx = Some(idx);
                }
            }
            
            if let Some(idx) = best_match_idx {
                // Update existing motif
                self.motifs[idx].frequency += new_motif.frequency;
                self.motifs[idx].stability = self.calculate_stability(&self.motifs[idx])?;
            } else {
                // Add new motif
                self.motifs.push(new_motif.clone());
            }
        }
        
        Ok(())
    }
    
    /// Get motif count
    pub fn get_motif_count(&self) -> usize {
        self.motifs.len()
    }
    
    /// Get motif statistics
    pub fn get_motif_stats(&self) -> MotifStats {
        let total_frequency: f32 = self.motifs.iter().map(|m| m.frequency).sum();
        let avg_stability: f32 = if self.motifs.is_empty() {
            0.0
        } else {
            self.motifs.iter().map(|m| m.stability).sum::<f32>() / self.motifs.len() as f32
        };
        
        MotifStats {
            count: self.motifs.len(),
            total_frequency,
            average_stability: avg_stability,
            max_stability: self.motifs.iter().map(|m| m.stability).fold(0.0, f32::max),
        }
    }
}

/// Motif statistics
#[derive(Debug, Clone)]
pub struct MotifStats {
    pub count: usize,
    pub total_frequency: f32,
    pub average_stability: f32,
    pub max_stability: f32,
}
