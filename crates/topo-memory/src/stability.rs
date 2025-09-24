//! Stability Formula: S = f(R, E, C, Φ, S)
//! 
//! Mathematical foundation for calculating stability across all memory levels.

use tensor_core::{Tensor, Shape, DType, Device, Result};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision};
use super::{Motif, Cycle, StableCore};

/// Stability calculator implementing the formula S = f(R, E, C, Φ, S)
#[derive(Debug, Clone)]
pub struct StabilityCalculator<T: Tensor> {
    // Stability calculation weights
    stability_weights: T,
    
    // Formula parameters
    r_weight: f32,      // Relevance weight
    e_weight: f32,      // Energy weight  
    c_weight: f32,      // Coherence weight
    phi_weight: f32,    // Phase weight
    s_weight: f32,      // Self-stability weight
    
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision> StabilityCalculator<T> {
    pub fn new(device: &Device) -> Result<Self> {
        let stability_weights = T::random_normal(
            Shape::new(vec![5, 1]), // 5 parameters for the formula
            0.0,
            0.1,
            device,
        )?;
        
        Ok(Self {
            stability_weights,
            r_weight: 0.3,    // Relevance
            e_weight: 0.2,    // Energy
            c_weight: 0.2,    // Coherence
            phi_weight: 0.15, // Phase
            s_weight: 0.15,   // Self-stability
            device: device.clone(),
        })
    }
    
    /// Calculate stability using the formula S = f(R, E, C, Φ, S)
    pub fn calculate_stability(
        &self,
        motifs: &[Motif<T>],
        cycles: &[Cycle<T>],
        stable_cores: &[StableCore<T>],
    ) -> Result<T> {
        // Calculate each component of the stability formula
        let r = self.calculate_relevance(motifs, cycles, stable_cores)?;
        let e = self.calculate_energy(motifs, cycles, stable_cores)?;
        let c = self.calculate_coherence(motifs, cycles, stable_cores)?;
        let phi = self.calculate_phase(motifs, cycles, stable_cores)?;
        let s = self.calculate_self_stability(motifs, cycles, stable_cores)?;
        
        // Apply the stability formula: S = f(R, E, C, Φ, S)
        let stability = self.apply_stability_formula(r, e, c, phi, s)?;
        
        Ok(stability)
    }
    
    /// Calculate relevance (R) - how relevant the patterns are to current context
    fn calculate_relevance(
        &self,
        motifs: &[Motif<T>],
        cycles: &[Cycle<T>],
        stable_cores: &[StableCore<T>],
    ) -> Result<f32> {
        // Relevance based on pattern frequency and recency
        let motif_relevance = self.calculate_motif_relevance(motifs)?;
        let cycle_relevance = self.calculate_cycle_relevance(cycles)?;
        let core_relevance = self.calculate_core_relevance(stable_cores)?;
        
        // Weighted average of relevance across levels
        let total_relevance = (motif_relevance * 0.4 + cycle_relevance * 0.3 + core_relevance * 0.3);
        
        Ok(total_relevance)
    }
    
    /// Calculate energy (E) - the information content and activity level
    fn calculate_energy(
        &self,
        motifs: &[Motif<T>],
        cycles: &[Cycle<T>],
        stable_cores: &[StableCore<T>],
    ) -> Result<f32> {
        // Energy based on pattern diversity and complexity
        let motif_energy = self.calculate_motif_energy(motifs)?;
        let cycle_energy = self.calculate_cycle_energy(cycles)?;
        let core_energy = self.calculate_core_energy(stable_cores)?;
        
        // Total energy as sum of all levels
        let total_energy = motif_energy + cycle_energy + core_energy;
        
        Ok(total_energy.min(1.0)) // Normalize to [0, 1]
    }
    
    /// Calculate coherence (C) - how well patterns fit together
    fn calculate_coherence(
        &self,
        motifs: &[Motif<T>],
        cycles: &[Cycle<T>],
        stable_cores: &[StableCore<T>],
    ) -> Result<f32> {
        // Coherence based on pattern consistency and alignment
        let motif_coherence = self.calculate_motif_coherence(motifs)?;
        let cycle_coherence = self.calculate_cycle_coherence(cycles)?;
        let core_coherence = self.calculate_core_coherence(stable_cores)?;
        
        // Cross-level coherence
        let cross_level_coherence = self.calculate_cross_level_coherence(motifs, cycles, stable_cores)?;
        
        // Combined coherence
        let total_coherence = (motif_coherence + cycle_coherence + core_coherence + cross_level_coherence) / 4.0;
        
        Ok(total_coherence)
    }
    
    /// Calculate phase (Φ) - the temporal/spatial phase relationships
    fn calculate_phase(
        &self,
        motifs: &[Motif<T>],
        cycles: &[Cycle<T>],
        stable_cores: &[StableCore<T>],
    ) -> Result<f32> {
        // Phase based on temporal alignment and periodicity
        let motif_phase = self.calculate_motif_phase(motifs)?;
        let cycle_phase = self.calculate_cycle_phase(cycles)?;
        let core_phase = self.calculate_core_phase(stable_cores)?;
        
        // Phase coherence across levels
        let phase_coherence = self.calculate_phase_coherence(motif_phase, cycle_phase, core_phase)?;
        
        Ok(phase_coherence)
    }
    
    /// Calculate self-stability (S) - the inherent stability of the system
    fn calculate_self_stability(
        &self,
        motifs: &[Motif<T>],
        cycles: &[Cycle<T>],
        stable_cores: &[StableCore<T>],
    ) -> Result<f32> {
        // Self-stability based on system equilibrium and resilience
        let motif_stability = self.calculate_motif_self_stability(motifs)?;
        let cycle_stability = self.calculate_cycle_self_stability(cycles)?;
        let core_stability = self.calculate_core_self_stability(stable_cores)?;
        
        // System-level stability
        let system_stability = (motif_stability + cycle_stability + core_stability) / 3.0;
        
        Ok(system_stability)
    }
    
    /// Apply the stability formula: S = f(R, E, C, Φ, S)
    fn apply_stability_formula(&self, r: f32, e: f32, c: f32, phi: f32, s: f32) -> Result<T> {
        // Linear combination of components
        let stability_value = self.r_weight * r + 
                             self.e_weight * e + 
                             self.c_weight * c + 
                             self.phi_weight * phi + 
                             self.s_weight * s;
        
        // Create tensor with stability value
        let stability_shape = Shape::new(vec![1]);
        let stability = T::random_normal(stability_shape, stability_value, 0.01, &self.device)?;
        
        Ok(stability)
    }
    
    // Helper methods for calculating individual components
    
    fn calculate_motif_relevance(&self, motifs: &[Motif<T>]) -> Result<f32> {
        if motifs.is_empty() {
            return Ok(0.0);
        }
        
        let avg_frequency: f32 = motifs.iter().map(|m| m.frequency).sum::<f32>() / motifs.len() as f32;
        let avg_stability: f32 = motifs.iter().map(|m| m.stability).sum::<f32>() / motifs.len() as f32;
        
        Ok((avg_frequency + avg_stability) / 2.0)
    }
    
    fn calculate_cycle_relevance(&self, cycles: &[Cycle<T>]) -> Result<f32> {
        if cycles.is_empty() {
            return Ok(0.0);
        }
        
        let avg_strength: f32 = cycles.iter().map(|c| c.strength).sum::<f32>() / cycles.len() as f32;
        let avg_stability: f32 = cycles.iter().map(|c| c.stability).sum::<f32>() / cycles.len() as f32;
        
        Ok((avg_strength + avg_stability) / 2.0)
    }
    
    fn calculate_core_relevance(&self, cores: &[StableCore<T>]) -> Result<f32> {
        if cores.is_empty() {
            return Ok(0.0);
        }
        
        let avg_stability: f32 = cores.iter().map(|c| c.stability_score).sum::<f32>() / cores.len() as f32;
        let avg_persistence: f32 = cores.iter().map(|c| c.persistence).sum::<f32>() / cores.len() as f32;
        
        Ok((avg_stability + avg_persistence) / 2.0)
    }
    
    fn calculate_motif_energy(&self, motifs: &[Motif<T>]) -> Result<f32> {
        Ok(motifs.len() as f32 * 0.1) // Simplified energy calculation
    }
    
    fn calculate_cycle_energy(&self, cycles: &[Cycle<T>]) -> Result<f32> {
        Ok(cycles.len() as f32 * 0.2) // Simplified energy calculation
    }
    
    fn calculate_core_energy(&self, cores: &[StableCore<T>]) -> Result<f32> {
        Ok(cores.len() as f32 * 0.3) // Simplified energy calculation
    }
    
    fn calculate_motif_coherence(&self, motifs: &[Motif<T>]) -> Result<f32> {
        // Simplified coherence calculation
        Ok(0.8)
    }
    
    fn calculate_cycle_coherence(&self, cycles: &[Cycle<T>]) -> Result<f32> {
        // Simplified coherence calculation
        Ok(0.85)
    }
    
    fn calculate_core_coherence(&self, cores: &[StableCore<T>]) -> Result<f32> {
        // Simplified coherence calculation
        Ok(0.9)
    }
    
    fn calculate_cross_level_coherence(&self, motifs: &[Motif<T>], cycles: &[Cycle<T>], cores: &[StableCore<T>]) -> Result<f32> {
        // Simplified cross-level coherence
        let motif_count = motifs.len() as f32;
        let cycle_count = cycles.len() as f32;
        let core_count = cores.len() as f32;
        
        // Balance across levels indicates good coherence
        let total = motif_count + cycle_count + core_count;
        if total == 0.0 {
            return Ok(0.0);
        }
        
        let balance = 1.0 - ((motif_count - cycle_count).abs() + (cycle_count - core_count).abs() + (motif_count - core_count).abs()) / (total * 2.0);
        
        Ok(balance)
    }
    
    fn calculate_motif_phase(&self, motifs: &[Motif<T>]) -> Result<f32> {
        // Simplified phase calculation
        Ok(0.7)
    }
    
    fn calculate_cycle_phase(&self, cycles: &[Cycle<T>]) -> Result<f32> {
        // Simplified phase calculation based on cycle periods
        if cycles.is_empty() {
            return Ok(0.0);
        }
        
        let avg_period = cycles.iter().map(|c| c.period).sum::<usize>() as f32 / cycles.len() as f32;
        let normalized_period = (avg_period / 20.0).min(1.0); // Normalize period
        
        Ok(normalized_period)
    }
    
    fn calculate_core_phase(&self, cores: &[StableCore<T>]) -> Result<f32> {
        // Simplified phase calculation
        Ok(0.9)
    }
    
    fn calculate_phase_coherence(&self, motif_phase: f32, cycle_phase: f32, core_phase: f32) -> Result<f32> {
        // Phase coherence as consistency across levels
        let avg_phase = (motif_phase + cycle_phase + core_phase) / 3.0;
        let variance = ((motif_phase - avg_phase).powi(2) + (cycle_phase - avg_phase).powi(2) + (core_phase - avg_phase).powi(2)) / 3.0;
        
        Ok(1.0 - variance) // Higher consistency = higher coherence
    }
    
    fn calculate_motif_self_stability(&self, motifs: &[Motif<T>]) -> Result<f32> {
        if motifs.is_empty() {
            return Ok(0.0);
        }
        
        let avg_stability: f32 = motifs.iter().map(|m| m.stability).sum::<f32>() / motifs.len() as f32;
        Ok(avg_stability)
    }
    
    fn calculate_cycle_self_stability(&self, cycles: &[Cycle<T>]) -> Result<f32> {
        if cycles.is_empty() {
            return Ok(0.0);
        }
        
        let avg_stability: f32 = cycles.iter().map(|c| c.stability).sum::<f32>() / cycles.len() as f32;
        Ok(avg_stability)
    }
    
    fn calculate_core_self_stability(&self, cores: &[StableCore<T>]) -> Result<f32> {
        if cores.is_empty() {
            return Ok(0.0);
        }
        
        let avg_stability: f32 = cores.iter().map(|c| c.stability_score).sum::<f32>() / cores.len() as f32;
        Ok(avg_stability)
    }
}

/// Stability metrics for analysis
#[derive(Debug, Clone)]
pub struct StabilityMetrics {
    pub relevance: f32,
    pub energy: f32,
    pub coherence: f32,
    pub phase: f32,
    pub self_stability: f32,
    pub total_stability: f32,
}
