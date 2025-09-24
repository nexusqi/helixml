//! 🌀 HelixML Topological Memory
//! 
//! Topological memory system with motifs, cycles, and stable cores.
//! Based on the stability formula: S = f(R, E, C, Φ, S)

use tensor_core::{Tensor, Shape, DType, Device, Result};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision};
use std::collections::HashMap;

pub mod motifs;
pub mod cycles;
pub mod stable_cores;
pub mod links;
pub mod stability;

pub use motifs::*;
pub use cycles::*;
pub use stable_cores::*;
pub use links::*;
pub use stability::*;

/// Core topological memory system
#[derive(Debug, Clone)]
pub struct TopologicalMemory<T: Tensor> {
    // Memory levels
    m0_motifs: MotifDetector<T>,
    m1_cycles: CycleAnalyzer<T>,
    m2_stable_cores: StableCoreExtractor<T>,
    
    // Link system
    u_links: TemporalLinks<T>,      // Temporal links
    i_links: IntermediateLinks<T>,  // Intermediate links
    s_links: StableLinks<T>,        // Stable links
    
    // Stability calculator
    stability_calc: StabilityCalculator<T>,
    
    // Configuration
    max_motif_length: usize,
    cycle_detection_threshold: f32,
    stability_threshold: f32,
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision> TopologicalMemory<T> {
    pub fn new(
        d_model: usize,
        max_motif_length: usize,
        cycle_threshold: f32,
        stability_threshold: f32,
        device: &Device,
    ) -> Result<Self> {
        let m0_motifs = MotifDetector::new(d_model, max_motif_length, device)?;
        let m1_cycles = CycleAnalyzer::new(d_model, cycle_threshold, device)?;
        let m2_stable_cores = StableCoreExtractor::new(d_model, stability_threshold, device)?;
        
        let u_links = TemporalLinks::new(d_model, device)?;
        let i_links = IntermediateLinks::new(d_model, device)?;
        let s_links = StableLinks::new(d_model, device)?;
        
        let stability_calc = StabilityCalculator::new(device)?;
        
        Ok(Self {
            m0_motifs,
            m1_cycles,
            m2_stable_cores,
            u_links,
            i_links,
            s_links,
            stability_calc,
            max_motif_length,
            cycle_detection_threshold: cycle_threshold,
            stability_threshold,
            device: device.clone(),
        })
    }
    
    /// Process sequence through all memory levels
    pub fn process_sequence(&mut self, sequence: &T) -> Result<TopologicalMemoryOutput<T>> {
        // M0: Detect motifs (short patterns)
        let motifs = self.m0_motifs.detect_motifs(sequence)?;
        
        // M1: Analyze cycles (medium-term dependencies)
        let cycles = self.m1_cycles.analyze_cycles(sequence, &motifs)?;
        
        // M2: Extract stable cores (long-term knowledge)
        let stable_cores = self.m2_stable_cores.extract_cores(sequence, &cycles)?;
        
        // Calculate stability using formula: S = f(R, E, C, Φ, S)
        let stability = self.stability_calc.calculate_stability(
            &motifs, &cycles, &stable_cores
        )?;
        
        // Update links
        let u_links = self.u_links.update_temporal_links(&motifs)?;
        let i_links = self.i_links.update_intermediate_links(&cycles)?;
        let s_links = self.s_links.update_stable_links(&stable_cores)?;
        
        Ok(TopologicalMemoryOutput {
            motifs,
            cycles,
            stable_cores,
            stability,
            temporal_links: u_links,
            intermediate_links: i_links,
            stable_links: s_links,
        })
    }
    
    /// Retrieve information from memory
    pub fn retrieve(&self, query: &T, memory_level: MemoryLevel) -> Result<T> {
        match memory_level {
            MemoryLevel::M0 => self.m0_motifs.retrieve_by_motifs(query),
            MemoryLevel::M1 => self.m1_cycles.retrieve_by_cycles(query),
            MemoryLevel::M2 => self.m2_stable_cores.retrieve_by_cores(query),
        }
    }
    
    /// Update memory with new information
    pub fn update(&mut self, new_data: &T) -> Result<()> {
        let output = self.process_sequence(new_data)?;
        
        // Update all memory levels
        self.m0_motifs.update_motifs(&output.motifs)?;
        self.m1_cycles.update_cycles(&output.cycles)?;
        self.m2_stable_cores.update_cores(&output.stable_cores)?;
        
        // Update links
        self.u_links = output.temporal_links;
        self.i_links = output.intermediate_links;
        self.s_links = output.stable_links;
        
        Ok(())
    }
    
    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            motif_count: self.m0_motifs.get_motif_count(),
            cycle_count: self.m1_cycles.get_cycle_count(),
            stable_core_count: self.m2_stable_cores.get_core_count(),
            temporal_link_count: self.u_links.get_link_count(),
            intermediate_link_count: self.i_links.get_link_count(),
            stable_link_count: self.s_links.get_link_count(),
        }
    }
}

/// Memory levels for retrieval
#[derive(Debug, Clone, Copy)]
pub enum MemoryLevel {
    M0,  // Motifs (short patterns)
    M1,  // Cycles (medium-term dependencies)
    M2,  // Stable cores (long-term knowledge)
}

/// Output from topological memory processing
#[derive(Debug, Clone)]
pub struct TopologicalMemoryOutput<T: Tensor> {
    pub motifs: Vec<Motif<T>>,
    pub cycles: Vec<Cycle<T>>,
    pub stable_cores: Vec<StableCore<T>>,
    pub stability: T,
    pub temporal_links: TemporalLinks<T>,
    pub intermediate_links: IntermediateLinks<T>,
    pub stable_links: StableLinks<T>,
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub motif_count: usize,
    pub cycle_count: usize,
    pub stable_core_count: usize,
    pub temporal_link_count: usize,
    pub intermediate_link_count: usize,
    pub stable_link_count: usize,
}

/// Motif representation
#[derive(Debug, Clone)]
pub struct Motif<T: Tensor> {
    pub pattern: T,
    pub frequency: f32,
    pub stability: f32,
    pub position: usize,
}

/// Cycle representation
#[derive(Debug, Clone)]
pub struct Cycle<T: Tensor> {
    pub nodes: Vec<T>,
    pub strength: f32,
    pub period: usize,
    pub stability: f32,
}

/// Stable core representation
#[derive(Debug, Clone)]
pub struct StableCore<T: Tensor> {
    pub core_pattern: T,
    pub stability_score: f32,
    pub persistence: f32,
    pub connections: Vec<usize>,
}