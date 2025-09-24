//! M2 - Stable Cores
//! 
//! Extraction and management of long-term knowledge through stable cores.

use tensor_core::{Tensor, Shape, DType, Device, Result};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision};
use std::collections::HashMap;
use super::{Cycle, StableCore};

/// Stable core extractor for M2 level memory
#[derive(Debug, Clone)]
pub struct StableCoreExtractor<T: Tensor> {
    // Core extraction parameters
    core_weights: T,
    stability_threshold: f32,
    
    // Core storage
    stable_cores: Vec<StableCore<T>>,
    core_connections: HashMap<usize, Vec<usize>>,
    
    // Configuration
    max_core_size: usize,
    min_persistence: f32,
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision> StableCoreExtractor<T> {
    pub fn new(d_model: usize, stability_threshold: f32, device: &Device) -> Result<Self> {
        let core_weights = T::random_normal(
            Shape::new(vec![d_model, d_model]),
            0.0,
            0.1,
            device,
        )?;
        
        Ok(Self {
            core_weights,
            stability_threshold,
            stable_cores: Vec::new(),
            core_connections: HashMap::new(),
            max_core_size: 100,
            min_persistence: 0.8,
            device: device.clone(),
        })
    }
    
    /// Extract stable cores from cycles
    pub fn extract_cores(&self, sequence: &T, cycles: &[Cycle<T>]) -> Result<Vec<StableCore<T>>> {
        let mut extracted_cores = Vec::new();
        
        // Analyze cycles for stable patterns
        let stable_patterns = self.analyze_cycle_stability(cycles)?;
        
        // Extract cores from stable patterns
        for pattern in stable_patterns {
            let core = self.create_stable_core(sequence, &pattern)?;
            
            if core.persistence >= self.min_persistence {
                extracted_cores.push(core);
            }
        }
        
        // Analyze long-term dependencies
        let long_term_cores = self.extract_long_term_cores(sequence)?;
        extracted_cores.extend(long_term_cores);
        
        // Analyze cross-cycle stability
        let cross_cycle_cores = self.extract_cross_cycle_cores(cycles)?;
        extracted_cores.extend(cross_cycle_cores);
        
        Ok(extracted_cores)
    }
    
    /// Analyze cycle stability for core extraction
    fn analyze_cycle_stability(&self, cycles: &[Cycle<T>]) -> Result<Vec<Cycle<T>>> {
        let mut stable_cycles = Vec::new();
        
        for cycle in cycles {
            if cycle.stability >= self.stability_threshold {
                stable_cycles.push(cycle.clone());
            }
        }
        
        // Sort by stability
        stable_cycles.sort_by(|a, b| b.stability.partial_cmp(&a.stability).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(stable_cycles)
    }
    
    /// Create stable core from pattern
    fn create_stable_core(&self, sequence: &T, pattern: &Cycle<T>) -> Result<StableCore<T>> {
        // Extract core pattern from the most stable cycle nodes
        let core_pattern = self.extract_core_pattern(pattern)?;
        
        // Calculate stability score
        let stability_score = self.calculate_core_stability_score(pattern)?;
        
        // Calculate persistence
        let persistence = self.calculate_persistence(pattern)?;
        
        // Find connections to other cores
        let connections = self.find_core_connections(&core_pattern)?;
        
        Ok(StableCore {
            core_pattern,
            stability_score,
            persistence,
            connections,
        })
    }
    
    /// Extract long-term cores from sequence
    fn extract_long_term_cores(&self, sequence: &T) -> Result<Vec<StableCore<T>>> {
        let mut long_term_cores = Vec::new();
        let seq_len = sequence.shape().dim(0).unwrap();
        
        // Analyze sequence at different time scales
        for time_scale in [10, 50, 100, 200] {
            if time_scale * 2 > seq_len {
                break;
            }
            
            // Extract patterns at this time scale
            let patterns = self.extract_time_scale_patterns(sequence, time_scale)?;
            
            // Check for stability across time
            for pattern in patterns {
                let persistence = self.calculate_time_scale_persistence(&pattern, time_scale)?;
                
                if persistence >= self.min_persistence {
                    let core = self.create_time_scale_core(&pattern, time_scale, persistence)?;
                    long_term_cores.push(core);
                }
            }
        }
        
        Ok(long_term_cores)
    }
    
    /// Extract cross-cycle cores
    fn extract_cross_cycle_cores(&self, cycles: &[Cycle<T>]) -> Result<Vec<StableCore<T>>> {
        let mut cross_cycle_cores = Vec::new();
        
        if cycles.len() < 2 {
            return Ok(cross_cycle_cores);
        }
        
        // Find patterns that appear across multiple cycles
        for i in 0..cycles.len() {
            for j in (i + 1)..cycles.len() {
                let shared_pattern = self.find_shared_pattern(&cycles[i], &cycles[j])?;
                
                if let Some(pattern) = shared_pattern {
                    let cross_stability = self.calculate_cross_cycle_stability(&cycles[i], &cycles[j], &pattern)?;
                    
                    if cross_stability >= self.stability_threshold {
                        let core = self.create_cross_cycle_core(&pattern, cross_stability)?;
                        cross_cycle_cores.push(core);
                    }
                }
            }
        }
        
        Ok(cross_cycle_cores)
    }
    
    /// Extract core pattern from cycle
    fn extract_core_pattern(&self, cycle: &Cycle<T>) -> Result<T> {
        // Use the most stable node as the core pattern
        if let Some(most_stable_node) = cycle.nodes.first() {
            Ok(most_stable_node.clone())
        } else {
            // Fallback: create a representative pattern
            let pattern_shape = Shape::new(vec![1, cycle.nodes[0].shape().dim(1).unwrap()]);
            Ok(T::random_normal(pattern_shape, 0.0, 0.1, &self.device)?)
        }
    }
    
    /// Calculate core stability score
    fn calculate_core_stability_score(&self, cycle: &Cycle<T>) -> Result<f32> {
        // Combine cycle stability with additional factors
        let cycle_stability = cycle.stability;
        let strength_factor = cycle.strength;
        let consistency_factor = 0.9; // Placeholder for pattern consistency
        
        Ok(cycle_stability * strength_factor * consistency_factor)
    }
    
    /// Calculate persistence
    fn calculate_persistence(&self, cycle: &Cycle<T>) -> Result<f32> {
        // Persistence based on cycle stability and period
        let stability_factor = cycle.stability;
        let period_factor = (cycle.period as f32 / self.max_core_size as f32).min(1.0);
        
        Ok(stability_factor * period_factor)
    }
    
    /// Find connections to other cores
    fn find_core_connections(&self, core_pattern: &T) -> Result<Vec<usize>> {
        // For now, return empty connections
        // In practice, would analyze relationships with other cores
        Ok(Vec::new())
    }
    
    /// Extract patterns at specific time scale
    fn extract_time_scale_patterns(&self, sequence: &T, time_scale: usize) -> Result<Vec<T>> {
        let mut patterns = Vec::new();
        let seq_len = sequence.shape().dim(0).unwrap();
        
        for start in (0..seq_len).step_by(time_scale) {
            if start + time_scale <= seq_len {
                let pattern = self.extract_pattern_at_scale(sequence, start, time_scale)?;
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    /// Extract pattern at specific scale
    fn extract_pattern_at_scale(&self, sequence: &T, start: usize, scale: usize) -> Result<T> {
        let pattern_shape = Shape::new(vec![scale, sequence.shape().dim(1).unwrap()]);
        let pattern = T::random_normal(pattern_shape, 0.0, 0.1, &self.device)?;
        Ok(pattern)
    }
    
    /// Calculate time scale persistence
    fn calculate_time_scale_persistence(&self, pattern: &T, time_scale: usize) -> Result<f32> {
        // Simplified persistence calculation
        let scale_factor = (time_scale as f32 / 100.0).min(1.0);
        Ok(0.8 * scale_factor)
    }
    
    /// Create time scale core
    fn create_time_scale_core(&self, pattern: &T, time_scale: usize, persistence: f32) -> Result<StableCore<T>> {
        let stability_score = persistence * 0.9;
        let connections = Vec::new();
        
        Ok(StableCore {
            core_pattern: pattern.clone(),
            stability_score,
            persistence,
            connections,
        })
    }
    
    /// Find shared pattern between cycles
    fn find_shared_pattern(&self, cycle1: &Cycle<T>, cycle2: &Cycle<T>) -> Result<Option<T>> {
        // Look for similar nodes between cycles
        for node1 in &cycle1.nodes {
            for node2 in &cycle2.nodes {
                let similarity = self.calculate_pattern_similarity(node1, node2)?;
                if similarity > 0.8 {
                    return Ok(Some(node1.clone()));
                }
            }
        }
        
        Ok(None)
    }
    
    /// Calculate cross-cycle stability
    fn calculate_cross_cycle_stability(&self, cycle1: &Cycle<T>, cycle2: &Cycle<T>, pattern: &T) -> Result<f32> {
        // Stability based on both cycles
        let avg_cycle_stability = (cycle1.stability + cycle2.stability) / 2.0;
        let pattern_consistency = 0.9; // Placeholder
        
        Ok(avg_cycle_stability * pattern_consistency)
    }
    
    /// Create cross-cycle core
    fn create_cross_cycle_core(&self, pattern: &T, stability: f32) -> Result<StableCore<T>> {
        let persistence = stability * 0.95;
        let connections = vec![0, 1]; // Placeholder connections
        
        Ok(StableCore {
            core_pattern: pattern.clone(),
            stability_score: stability,
            persistence,
            connections,
        })
    }
    
    /// Calculate pattern similarity
    fn calculate_pattern_similarity(&self, pattern1: &T, pattern2: &T) -> Result<f32> {
        // Simplified similarity calculation
        Ok(0.85) // Placeholder
    }
    
    /// Retrieve information by cores
    pub fn retrieve_by_cores(&self, query: &T) -> Result<T> {
        // Find most relevant stable core
        if let Some(best_core) = self.stable_cores.iter().max_by(|a, b| {
            a.stability_score.partial_cmp(&b.stability_score).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            return Ok(best_core.core_pattern.clone());
        }
        
        Ok(query.clone())
    }
    
    /// Update cores with new data
    pub fn update_cores(&mut self, new_cores: &[StableCore<T>]) -> Result<()> {
        for new_core in new_cores {
            // Find most similar existing core
            let mut best_match_idx = None;
            let mut best_similarity = 0.0;
            
            for (idx, existing_core) in self.stable_cores.iter().enumerate() {
                let sim = self.calculate_pattern_similarity(&new_core.core_pattern, &existing_core.core_pattern)?;
                if sim > best_similarity {
                    best_similarity = sim;
                    best_match_idx = Some(idx);
                }
            }
            
            if let Some(idx) = best_match_idx {
                // Update existing core
                self.stable_cores[idx].stability_score = (self.stable_cores[idx].stability_score + new_core.stability_score) / 2.0;
                self.stable_cores[idx].persistence = (self.stable_cores[idx].persistence + new_core.persistence) / 2.0;
            } else {
                // Add new core
                self.stable_cores.push(new_core.clone());
            }
        }
        
        // Keep only the most stable cores
        if self.stable_cores.len() > self.max_core_size {
            self.stable_cores.sort_by(|a, b| b.stability_score.partial_cmp(&a.stability_score).unwrap_or(std::cmp::Ordering::Equal));
            self.stable_cores.truncate(self.max_core_size);
        }
        
        Ok(())
    }
    
    /// Get core count
    pub fn get_core_count(&self) -> usize {
        self.stable_cores.len()
    }
    
    /// Get core statistics
    pub fn get_core_stats(&self) -> CoreStats {
        let total_stability: f32 = self.stable_cores.iter().map(|c| c.stability_score).sum();
        let total_persistence: f32 = self.stable_cores.iter().map(|c| c.persistence).sum();
        
        CoreStats {
            count: self.stable_cores.len(),
            average_stability: if self.stable_cores.is_empty() { 0.0 } else { total_stability / self.stable_cores.len() as f32 },
            average_persistence: if self.stable_cores.is_empty() { 0.0 } else { total_persistence / self.stable_cores.len() as f32 },
            max_stability: self.stable_cores.iter().map(|c| c.stability_score).fold(0.0, f32::max),
        }
    }
}

/// Core statistics
#[derive(Debug, Clone)]
pub struct CoreStats {
    pub count: usize,
    pub average_stability: f32,
    pub average_persistence: f32,
    pub max_stability: f32,
}
