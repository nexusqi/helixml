//! ⚖️ Load Balancer
//! 
//! Load balancing strategies for multi-device task distribution

use tensor_core::{Tensor, Shape, DType, Device, Result};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision, TensorStats, TensorReduce};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use anyhow::Context;
use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;

use super::*;

/// Load balancer for distributing tasks across devices
#[derive(Debug)]
pub struct LoadBalancer<T: Tensor> {
    strategy: LoadBalancingStrategy,
    device_weights: Arc<RwLock<HashMap<Device, f32>>>,
    device_loads: Arc<RwLock<HashMap<Device, f32>>>,
    device_performance: Arc<RwLock<HashMap<Device, DevicePerformance>>>,
    round_robin_index: Arc<Mutex<usize>>,
    load_history: Arc<RwLock<HashMap<Device, VecDeque<f32>>>>,
    adaptive_parameters: Arc<RwLock<AdaptiveParameters>>,
}

/// Device performance metrics
#[derive(Debug, Clone)]
pub struct DevicePerformance {
    pub throughput: f32,
    pub latency: Duration,
    pub efficiency: f32,
    pub error_rate: f32,
    pub success_rate: f32,
    pub last_updated: Instant,
}

/// Adaptive parameters for load balancing
#[derive(Debug, Clone)]
pub struct AdaptiveParameters {
    pub learning_rate: f32,
    pub exploration_rate: f32,
    pub performance_window: Duration,
    pub load_threshold: f32,
    pub rebalance_threshold: f32,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorStats + TensorReduce> LoadBalancer<T> {
    pub fn new(strategy: LoadBalancingStrategy) -> Result<Self> {
        Ok(Self {
            strategy,
            device_weights: Arc::new(RwLock::new(HashMap::new())),
            device_loads: Arc::new(RwLock::new(HashMap::new())),
            device_performance: Arc::new(RwLock::new(HashMap::new())),
            round_robin_index: Arc::new(Mutex::new(0)),
            load_history: Arc::new(RwLock::new(HashMap::new())),
            adaptive_parameters: Arc::new(RwLock::new(AdaptiveParameters {
                learning_rate: 0.1,
                exploration_rate: 0.1,
                performance_window: Duration::from_secs(60),
                load_threshold: 0.8,
                rebalance_threshold: 0.2,
            })),
        })
    }
    
    /// Select the best device for a task
    pub fn select_device(&self, available_devices: &[Device], task: &Task<T>) -> Result<Device> {
        if available_devices.is_empty() {
            return Err(anyhow::anyhow!("No available devices"));
        }
        
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                self.select_round_robin(available_devices)
            }
            LoadBalancingStrategy::LeastLoaded => {
                self.select_least_loaded(available_devices)
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.select_weighted_round_robin(available_devices)
            }
            LoadBalancingStrategy::LeastConnections => {
                self.select_least_connections(available_devices)
            }
            LoadBalancingStrategy::ResourceBased => {
                self.select_resource_based(available_devices, task)
            }
            LoadBalancingStrategy::PerformanceBased => {
                self.select_performance_based(available_devices, task)
            }
            LoadBalancingStrategy::Adaptive => {
                self.select_adaptive(available_devices, task)
            }
        }
    }
    
    /// Update device load
    pub fn update_device_load(&self, device: &Device, load: f32) -> Result<()> {
        // Update current load
        {
            let mut loads = self.device_loads.write().unwrap();
            loads.insert(device.clone(), load);
        }
        
        // Update load history
        {
            let mut history = self.load_history.write().unwrap();
            let device_history = history.entry(device.clone()).or_insert_with(VecDeque::new);
            device_history.push_back(load);
            
            // Keep only recent history
            if device_history.len() > 100 {
                device_history.pop_front();
            }
        }
        
        Ok(())
    }
    
    /// Update device performance
    pub fn update_device_performance(&self, device: &Device, performance: DevicePerformance) -> Result<()> {
        let mut perf_map = self.device_performance.write().unwrap();
        perf_map.insert(device.clone(), performance);
        Ok(())
    }
    
    /// Update load balancing strategy
    pub fn update_strategy(&self, strategy: LoadBalancingStrategy) -> Result<()> {
        // This would require changing the strategy field, which is not mutable
        // In practice, you'd need to make this field mutable or use interior mutability
        Ok(())
    }
    
    /// Get device load
    pub fn get_device_load(&self, device: &Device) -> Result<f32> {
        let loads = self.device_loads.read().unwrap();
        Ok(loads.get(device).copied().unwrap_or(0.0))
    }
    
    /// Get device performance
    pub fn get_device_performance(&self, device: &Device) -> Result<Option<DevicePerformance>> {
        let perf_map = self.device_performance.read().unwrap();
        Ok(perf_map.get(device).cloned())
    }
    
    /// Get load balancing statistics
    pub fn get_statistics(&self) -> Result<LoadBalancingStatistics> {
        let loads = self.device_loads.read().unwrap();
        let performance = self.device_performance.read().unwrap();
        
        let mut total_load = 0.0;
        let mut device_count = 0;
        let mut load_variance = 0.0;
        
        for load in loads.values() {
            total_load += load;
            device_count += 1;
        }
        
        let average_load = if device_count > 0 { total_load / device_count as f32 } else { 0.0 };
        
        // Calculate variance
        for load in loads.values() {
            let diff = load - average_load;
            load_variance += diff * diff;
        }
        load_variance /= device_count as f32;
        
        Ok(LoadBalancingStatistics {
            average_load,
            load_variance,
            device_count,
            strategy: self.strategy.clone(),
        })
    }
    
    /// Round-robin device selection
    fn select_round_robin(&self, available_devices: &[Device]) -> Result<Device> {
        let mut index = self.round_robin_index.lock().unwrap();
        let device = available_devices[*index % available_devices.len()].clone();
        *index += 1;
        Ok(device)
    }
    
    /// Least loaded device selection
    fn select_least_loaded(&self, available_devices: &[Device]) -> Result<Device> {
        let loads = self.device_loads.read().unwrap();
        let mut best_device = available_devices[0].clone();
        let mut best_load = f32::MAX;
        
        for device in available_devices {
            let load = loads.get(device).copied().unwrap_or(0.0);
            if load < best_load {
                best_load = load;
                best_device = device.clone();
            }
        }
        
        Ok(best_device)
    }
    
    /// Weighted round-robin device selection
    fn select_weighted_round_robin(&self, available_devices: &[Device]) -> Result<Device> {
        let weights = self.device_weights.read().unwrap();
        let loads = self.device_loads.read().unwrap();
        
        // Calculate weighted scores
        let mut scores = Vec::new();
        for device in available_devices {
            let weight = weights.get(device).copied().unwrap_or(1.0);
            let load = loads.get(device).copied().unwrap_or(0.0);
            let score = weight / (1.0 + load); // Higher weight, lower load = better score
            scores.push((device.clone(), score));
        }
        
        // Select device with highest score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(scores[0].0.clone())
    }
    
    /// Least connections device selection
    fn select_least_connections(&self, available_devices: &[Device]) -> Result<Device> {
        // This would require tracking active connections per device
        // For now, fall back to least loaded
        self.select_least_loaded(available_devices)
    }
    
    /// Resource-based device selection
    fn select_resource_based(&self, available_devices: &[Device], task: &Task<T>) -> Result<Device> {
        let loads = self.device_loads.read().unwrap();
        let mut best_device = available_devices[0].clone();
        let mut best_score = f32::MIN;
        
        for device in available_devices {
            let load = loads.get(device).copied().unwrap_or(0.0);
            
            // Calculate resource score based on task requirements
            let memory_score = if task.resource_requirements.memory > 0 {
                1.0 / (1.0 + task.resource_requirements.memory as f32 / 1024.0 / 1024.0) // Normalize to MB
            } else {
                1.0
            };
            
            let compute_score = if task.resource_requirements.compute > 0.0 {
                1.0 / (1.0 + task.resource_requirements.compute)
            } else {
                1.0
            };
            
            let load_score = 1.0 - load; // Lower load is better
            let total_score = memory_score * compute_score * load_score;
            
            if total_score > best_score {
                best_score = total_score;
                best_device = device.clone();
            }
        }
        
        Ok(best_device)
    }
    
    /// Performance-based device selection
    fn select_performance_based(&self, available_devices: &[Device], task: &Task<T>) -> Result<Device> {
        let performance = self.device_performance.read().unwrap();
        let loads = self.device_loads.read().unwrap();
        
        let mut best_device = available_devices[0].clone();
        let mut best_score = f32::MIN;
        
        for device in available_devices {
            let load = loads.get(device).copied().unwrap_or(0.0);
            
            if let Some(perf) = performance.get(device) {
                // Calculate performance score
                let throughput_score = perf.throughput;
                let latency_score = 1.0 / (perf.latency.as_secs_f32() + 0.001); // Avoid division by zero
                let efficiency_score = perf.efficiency;
                let success_score = perf.success_rate;
                let load_score = 1.0 - load;
                
                let total_score = throughput_score * latency_score * efficiency_score * success_score * load_score;
                
                if total_score > best_score {
                    best_score = total_score;
                    best_device = device.clone();
                }
            }
        }
        
        Ok(best_device)
    }
    
    /// Adaptive device selection
    fn select_adaptive(&self, available_devices: &[Device], task: &Task<T>) -> Result<Device> {
        let params = self.adaptive_parameters.read().unwrap();
        let mut rng = thread_rng();
        
        // Exploration vs exploitation
        if rng.gen::<f32>() < params.exploration_rate {
            // Random selection for exploration
            let device = available_devices.choose(&mut rng).unwrap().clone();
            return Ok(device);
        }
        
        // Exploitation: use learned knowledge
        let performance = self.device_performance.read().unwrap();
        let loads = self.device_loads.read().unwrap();
        
        let mut best_device = available_devices[0].clone();
        let mut best_score = f32::MIN;
        
        for device in available_devices {
            let load = loads.get(device).copied().unwrap_or(0.0);
            
            if let Some(perf) = performance.get(device) {
                // Calculate adaptive score
                let recency_factor = if perf.last_updated.elapsed() < params.performance_window {
                    1.0
                } else {
                    0.5 // Penalize stale performance data
                };
                
                let throughput_score = perf.throughput * recency_factor;
                let latency_score = 1.0 / (perf.latency.as_secs_f32() + 0.001);
                let efficiency_score = perf.efficiency;
                let success_score = perf.success_rate;
                let load_score = 1.0 - load;
                
                // Weighted combination
                let total_score = (throughput_score * 0.3 + latency_score * 0.2 + 
                                 efficiency_score * 0.2 + success_score * 0.2 + load_score * 0.1);
                
                if total_score > best_score {
                    best_score = total_score;
                    best_device = device.clone();
                }
            }
        }
        
        Ok(best_device)
    }
    
    /// Update device weights based on performance
    pub fn update_device_weights(&self) -> Result<()> {
        let performance = self.device_performance.read().unwrap();
        let mut weights = self.device_weights.write().unwrap();
        let params = self.adaptive_parameters.read().unwrap();
        
        for (device, perf) in performance.iter() {
            let current_weight = weights.get(device).copied().unwrap_or(1.0);
            
            // Calculate performance-based weight update
            let performance_score = perf.throughput * perf.efficiency * perf.success_rate;
            let weight_update = params.learning_rate * (performance_score - current_weight);
            
            let new_weight = (current_weight + weight_update).max(0.1).min(10.0); // Clamp between 0.1 and 10.0
            weights.insert(device.clone(), new_weight);
        }
        
        Ok(())
    }
    
    /// Check if rebalancing is needed
    pub fn needs_rebalancing(&self) -> Result<bool> {
        let loads = self.device_loads.read().unwrap();
        let params = self.adaptive_parameters.read().unwrap();
        
        if loads.is_empty() {
            return Ok(false);
        }
        
        // Calculate load variance
        let total_load: f32 = loads.values().sum();
        let average_load = total_load / loads.len() as f32;
        
        let mut variance = 0.0;
        for load in loads.values() {
            let diff = load - average_load;
            variance += diff * diff;
        }
        variance /= loads.len() as f32;
        
        let load_std = variance.sqrt();
        
        // Check if any device is overloaded
        let overloaded = loads.values().any(|&load| load > params.load_threshold);
        
        // Check if load distribution is uneven
        let uneven_distribution = load_std > params.rebalance_threshold;
        
        Ok(overloaded || uneven_distribution)
    }
    
    /// Rebalance load across devices
    pub fn rebalance(&self) -> Result<RebalancingResult> {
        let loads = self.device_loads.read().unwrap();
        let performance = self.device_performance.read().unwrap();
        
        let mut rebalancing_actions = Vec::new();
        
        // Find overloaded devices
        let mut overloaded_devices = Vec::new();
        let mut underloaded_devices = Vec::new();
        
        let total_load: f32 = loads.values().sum();
        let average_load = total_load / loads.len() as f32;
        
        for (device, &load) in loads.iter() {
            if load > average_load * 1.2 { // 20% above average
                overloaded_devices.push(device.clone());
            } else if load < average_load * 0.8 { // 20% below average
                underloaded_devices.push(device.clone());
            }
        }
        
        // Suggest load redistribution
        for overloaded in &overloaded_devices {
            for underloaded in &underloaded_devices {
                rebalancing_actions.push(RebalancingAction {
                    action_type: RebalancingActionType::MoveLoad,
                    source_device: overloaded.clone(),
                    target_device: underloaded.clone(),
                    priority: 1.0,
                });
            }
        }
        
        // Update device weights
        self.update_device_weights()?;
        
        Ok(RebalancingResult {
            actions: rebalancing_actions,
            estimated_improvement: 0.3, // Placeholder
            rebalancing_cost: 0.1, // Placeholder
        })
    }
}

/// Load balancing statistics
#[derive(Debug, Clone)]
pub struct LoadBalancingStatistics {
    pub average_load: f32,
    pub load_variance: f32,
    pub device_count: usize,
    pub strategy: LoadBalancingStrategy,
}

/// Rebalancing result
#[derive(Debug, Clone)]
pub struct RebalancingResult {
    pub actions: Vec<RebalancingAction>,
    pub estimated_improvement: f32,
    pub rebalancing_cost: f32,
}

/// Rebalancing action
#[derive(Debug, Clone)]
pub struct RebalancingAction {
    pub action_type: RebalancingActionType,
    pub source_device: Device,
    pub target_device: Device,
    pub priority: f32,
}

/// Rebalancing action types
#[derive(Debug, Clone)]
pub enum RebalancingActionType {
    MoveLoad,
    AdjustWeights,
    ChangeStrategy,
    ScaleResources,
}
