//! U/I/S Links - Temporal/Intermediate/Stable connections
//! 
//! Link system for connecting different memory levels.

use tensor_core::{Tensor, Shape, DType, Device, Result};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision};
use std::collections::HashMap;
use super::{Motif, Cycle, StableCore};

/// Temporal links (U-links) for short-term connections
#[derive(Debug, Clone)]
pub struct TemporalLinks<T: Tensor> {
    // Link weights
    temporal_weights: T,
    
    // Link storage
    links: HashMap<usize, Vec<usize>>,
    link_strengths: HashMap<(usize, usize), f32>,
    
    // Configuration
    max_links_per_node: usize,
    link_decay_rate: f32,
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision> TemporalLinks<T> {
    pub fn new(d_model: usize, device: &Device) -> Result<Self> {
        let temporal_weights = T::random_normal(
            Shape::new(vec![d_model, d_model]),
            0.0,
            0.1,
            device,
        )?;
        
        Ok(Self {
            temporal_weights,
            links: HashMap::new(),
            link_strengths: HashMap::new(),
            max_links_per_node: 5,
            link_decay_rate: 0.95,
            device: device.clone(),
        })
    }
    
    /// Update temporal links based on motifs
    pub fn update_temporal_links(&self, motifs: &[Motif<T>]) -> Result<Self> {
        let mut new_links = self.clone();
        
        // Create links between temporally close motifs
        for (i, motif1) in motifs.iter().enumerate() {
            for (j, motif2) in motifs.iter().enumerate() {
                if i != j {
                    let temporal_distance = (motif1.position as i32 - motif2.position as i32).abs() as usize;
                    
                    if temporal_distance <= 10 { // Short-term temporal window
                        let link_strength = self.calculate_temporal_link_strength(motif1, motif2, temporal_distance)?;
                        
                        if link_strength > 0.5 {
                            new_links.add_link(i, j, link_strength)?;
                        }
                    }
                }
            }
        }
        
        // Apply decay to existing links
        new_links.apply_link_decay()?;
        
        Ok(new_links)
    }
    
    /// Calculate temporal link strength
    fn calculate_temporal_link_strength(&self, motif1: &Motif<T>, motif2: &Motif<T>, distance: usize) -> Result<f32> {
        // Strength based on motif similarity and temporal proximity
        let similarity = 0.8; // Placeholder for motif similarity
        let proximity_factor = 1.0 / (1.0 + distance as f32 * 0.1);
        
        Ok(similarity * proximity_factor)
    }
    
    /// Add link between nodes
    fn add_link(&mut self, from: usize, to: usize, strength: f32) -> Result<()> {
        // Add to links
        self.links.entry(from).or_insert_with(Vec::new).push(to);
        
        // Limit links per node
        if let Some(node_links) = self.links.get_mut(&from) {
            if node_links.len() > self.max_links_per_node {
                // Remove weakest link
                node_links.pop();
            }
        }
        
        // Store link strength
        self.link_strengths.insert((from, to), strength);
        
        Ok(())
    }
    
    /// Apply decay to all links
    fn apply_link_decay(&mut self) -> Result<()> {
        // Decay all link strengths
        for strength in self.link_strengths.values_mut() {
            *strength *= self.link_decay_rate;
        }
        
        // Remove weak links
        self.link_strengths.retain(|_, strength| *strength > 0.1);
        
        Ok(())
    }
    
    /// Get link count
    pub fn get_link_count(&self) -> usize {
        self.link_strengths.len()
    }
}

/// Intermediate links (I-links) for medium-term connections
#[derive(Debug, Clone)]
pub struct IntermediateLinks<T: Tensor> {
    // Link weights
    intermediate_weights: T,
    
    // Link storage
    links: HashMap<usize, Vec<usize>>,
    link_strengths: HashMap<(usize, usize), f32>,
    
    // Configuration
    max_links_per_node: usize,
    link_decay_rate: f32,
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision> IntermediateLinks<T> {
    pub fn new(d_model: usize, device: &Device) -> Result<Self> {
        let intermediate_weights = T::random_normal(
            Shape::new(vec![d_model, d_model]),
            0.0,
            0.1,
            device,
        )?;
        
        Ok(Self {
            intermediate_weights,
            links: HashMap::new(),
            link_strengths: HashMap::new(),
            max_links_per_node: 10,
            link_decay_rate: 0.98,
            device: device.clone(),
        })
    }
    
    /// Update intermediate links based on cycles
    pub fn update_intermediate_links(&self, cycles: &[Cycle<T>]) -> Result<Self> {
        let mut new_links = self.clone();
        
        // Create links between cycles
        for (i, cycle1) in cycles.iter().enumerate() {
            for (j, cycle2) in cycles.iter().enumerate() {
                if i != j {
                    let link_strength = self.calculate_cycle_link_strength(cycle1, cycle2)?;
                    
                    if link_strength > 0.6 {
                        new_links.add_link(i, j, link_strength)?;
                    }
                }
            }
        }
        
        // Apply decay to existing links
        new_links.apply_link_decay()?;
        
        Ok(new_links)
    }
    
    /// Calculate cycle link strength
    fn calculate_cycle_link_strength(&self, cycle1: &Cycle<T>, cycle2: &Cycle<T>) -> Result<f32> {
        // Strength based on cycle similarity and stability
        let similarity = 0.85; // Placeholder for cycle similarity
        let stability_factor = (cycle1.stability + cycle2.stability) / 2.0;
        
        Ok(similarity * stability_factor)
    }
    
    /// Add link between nodes
    fn add_link(&mut self, from: usize, to: usize, strength: f32) -> Result<()> {
        self.links.entry(from).or_insert_with(Vec::new).push(to);
        
        if let Some(node_links) = self.links.get_mut(&from) {
            if node_links.len() > self.max_links_per_node {
                node_links.pop();
            }
        }
        
        self.link_strengths.insert((from, to), strength);
        
        Ok(())
    }
    
    /// Apply decay to all links
    fn apply_link_decay(&mut self) -> Result<()> {
        for strength in self.link_strengths.values_mut() {
            *strength *= self.link_decay_rate;
        }
        
        self.link_strengths.retain(|_, strength| *strength > 0.2);
        
        Ok(())
    }
    
    /// Get link count
    pub fn get_link_count(&self) -> usize {
        self.link_strengths.len()
    }
}

/// Stable links (S-links) for long-term connections
#[derive(Debug, Clone)]
pub struct StableLinks<T: Tensor> {
    // Link weights
    stable_weights: T,
    
    // Link storage
    links: HashMap<usize, Vec<usize>>,
    link_strengths: HashMap<(usize, usize), f32>,
    
    // Configuration
    max_links_per_node: usize,
    link_decay_rate: f32,
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision> StableLinks<T> {
    pub fn new(d_model: usize, device: &Device) -> Result<Self> {
        let stable_weights = T::random_normal(
            Shape::new(vec![d_model, d_model]),
            0.0,
            0.1,
            device,
        )?;
        
        Ok(Self {
            stable_weights,
            links: HashMap::new(),
            link_strengths: HashMap::new(),
            max_links_per_node: 20,
            link_decay_rate: 0.99,
            device: device.clone(),
        })
    }
    
    /// Update stable links based on stable cores
    pub fn update_stable_links(&self, stable_cores: &[StableCore<T>]) -> Result<Self> {
        let mut new_links = self.clone();
        
        // Create links between stable cores
        for (i, core1) in stable_cores.iter().enumerate() {
            for (j, core2) in stable_cores.iter().enumerate() {
                if i != j {
                    let link_strength = self.calculate_stable_link_strength(core1, core2)?;
                    
                    if link_strength > 0.7 {
                        new_links.add_link(i, j, link_strength)?;
                    }
                }
            }
        }
        
        // Apply decay to existing links
        new_links.apply_link_decay()?;
        
        Ok(new_links)
    }
    
    /// Calculate stable link strength
    fn calculate_stable_link_strength(&self, core1: &StableCore<T>, core2: &StableCore<T>) -> Result<f32> {
        // Strength based on core similarity and persistence
        let similarity = 0.9; // Placeholder for core similarity
        let persistence_factor = (core1.persistence + core2.persistence) / 2.0;
        
        Ok(similarity * persistence_factor)
    }
    
    /// Add link between nodes
    fn add_link(&mut self, from: usize, to: usize, strength: f32) -> Result<()> {
        self.links.entry(from).or_insert_with(Vec::new).push(to);
        
        if let Some(node_links) = self.links.get_mut(&from) {
            if node_links.len() > self.max_links_per_node {
                node_links.pop();
            }
        }
        
        self.link_strengths.insert((from, to), strength);
        
        Ok(())
    }
    
    /// Apply decay to all links
    fn apply_link_decay(&mut self) -> Result<()> {
        for strength in self.link_strengths.values_mut() {
            *strength *= self.link_decay_rate;
        }
        
        self.link_strengths.retain(|_, strength| *strength > 0.3);
        
        Ok(())
    }
    
    /// Get link count
    pub fn get_link_count(&self) -> usize {
        self.link_strengths.len()
    }
}
