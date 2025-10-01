//! 📋 Scheduling Policies
//! 
//! Policy management for adaptive scheduling decisions

use tensor_core::{Tensor, Shape, DType, Device, Result};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision, TensorStats, TensorReduce};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use anyhow::Context;

use super::*;

/// Policy manager for scheduling decisions
#[derive(Debug)]
pub struct PolicyManager<T: Tensor> {
    policies: Arc<RwLock<HashMap<PolicyType, Box<dyn SchedulingPolicy + Send + Sync>>>>,
    policy_weights: Arc<RwLock<HashMap<PolicyType, f32>>>,
    policy_history: Arc<RwLock<VecDeque<PolicyDecision>>>,
    adaptive_parameters: Arc<RwLock<AdaptivePolicyParameters>>,
    policy_metrics: Arc<RwLock<PolicyMetrics>>,
}

/// Scheduling policy trait
pub trait SchedulingPolicy {
    fn should_schedule(&self, task: &Task<()>, device: &Device) -> Result<bool>;
    fn get_priority(&self, task: &Task<()>, device: &Device) -> Result<f32>;
    fn get_policy_name(&self) -> &str;
    fn get_parameters(&self) -> HashMap<String, f32>;
}

/// Policy types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PolicyType {
    ResourcePolicy,
    LoadBalancingPolicy,
    PriorityPolicy,
    EnergyPolicy,
    LatencyPolicy,
    ThroughputPolicy,
    FairnessPolicy,
    CostPolicy,
}

/// Policy decision
#[derive(Debug, Clone)]
pub struct PolicyDecision {
    pub policy_type: PolicyType,
    pub task_id: TaskId,
    pub device: Device,
    pub decision: bool,
    pub priority: f32,
    pub timestamp: Instant,
    pub reasoning: String,
}

/// Adaptive policy parameters
#[derive(Debug, Clone)]
pub struct AdaptivePolicyParameters {
    pub learning_rate: f32,
    pub exploration_rate: f32,
    pub policy_update_frequency: Duration,
    pub performance_window: Duration,
    pub convergence_threshold: f32,
    pub max_iterations: usize,
}

/// Policy metrics
#[derive(Debug, Clone)]
pub struct PolicyMetrics {
    pub total_decisions: usize,
    pub successful_decisions: usize,
    pub average_priority: f32,
    pub policy_effectiveness: HashMap<PolicyType, f32>,
    pub decision_time: Duration,
    pub convergence_rate: f32,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorStats + TensorReduce> PolicyManager<T> {
    pub fn new() -> Result<Self> {
        let mut manager = Self {
            policies: Arc::new(RwLock::new(HashMap::new())),
            policy_weights: Arc::new(RwLock::new(HashMap::new())),
            policy_history: Arc::new(RwLock::new(VecDeque::new())),
            adaptive_parameters: Arc::new(RwLock::new(AdaptivePolicyParameters {
                learning_rate: 0.1,
                exploration_rate: 0.1,
                policy_update_frequency: Duration::from_secs(30),
                performance_window: Duration::from_secs(300),
                convergence_threshold: 0.01,
                max_iterations: 100,
            })),
            policy_metrics: Arc::new(RwLock::new(PolicyMetrics {
                total_decisions: 0,
                successful_decisions: 0,
                average_priority: 0.0,
                policy_effectiveness: HashMap::new(),
                decision_time: Duration::from_millis(0),
                convergence_rate: 0.0,
            })),
        };
        
        // Initialize policies
        manager.initialize_policies()?;
        
        Ok(manager)
    }
    
    /// Check if a task should be scheduled on a device
    pub fn should_schedule(&self, task: &Task<T>, device: &Device) -> Result<bool> {
        let start_time = Instant::now();
        
        // Get all policies
        let policies = self.policies.read().unwrap();
        let weights = self.policy_weights.read().unwrap();
        
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        let mut decisions = Vec::new();
        
        // Evaluate each policy
        for (policy_type, policy) in policies.iter() {
            let weight = weights.get(policy_type).copied().unwrap_or(1.0);
            
            if let Ok(should_schedule) = policy.should_schedule(&self.convert_task(task), device) {
                let priority = policy.get_priority(&self.convert_task(task), device).unwrap_or(0.5);
                let score = if should_schedule { priority } else { 0.0 };
                
                total_score += score * weight;
                total_weight += weight;
                
                decisions.push(PolicyDecision {
                    policy_type: policy_type.clone(),
                    task_id: TaskId::new(), // This should be passed from the caller
                    device: device.clone(),
                    decision: should_schedule,
                    priority,
                    timestamp: Instant::now(),
                    reasoning: format!("Policy {} decision: {}", policy.get_policy_name(), should_schedule),
                });
            }
        }
        
        // Make final decision
        let final_decision = if total_weight > 0.0 {
            total_score / total_weight > 0.5
        } else {
            true // Default to allowing scheduling
        };
        
        // Record decision
        let decision_time = start_time.elapsed();
        self.record_decision(decisions, final_decision, decision_time)?;
        
        Ok(final_decision)
    }
    
    /// Get task priority based on policies
    pub fn get_task_priority(&self, task: &Task<T>, device: &Device) -> Result<f32> {
        let policies = self.policies.read().unwrap();
        let weights = self.policy_weights.read().unwrap();
        
        let mut total_priority = 0.0;
        let mut total_weight = 0.0;
        
        for (policy_type, policy) in policies.iter() {
            let weight = weights.get(policy_type).copied().unwrap_or(1.0);
            
            if let Ok(priority) = policy.get_priority(&self.convert_task(task), device) {
                total_priority += priority * weight;
                total_weight += weight;
            }
        }
        
        Ok(if total_weight > 0.0 { total_priority / total_weight } else { 0.5 })
    }
    
    /// Update policy weights
    pub fn update_policy_weights(&self, weights: HashMap<PolicyType, f32>) -> Result<()> {
        let mut policy_weights = self.policy_weights.write().unwrap();
        *policy_weights = weights;
        Ok(())
    }
    
    /// Get policy metrics
    pub fn get_policy_metrics(&self) -> Result<PolicyMetrics> {
        let metrics = self.policy_metrics.read().unwrap();
        Ok(metrics.clone())
    }
    
    /// Initialize policies
    fn initialize_policies(&self) -> Result<()> {
        let mut policies = self.policies.write().unwrap();
        let mut weights = self.policy_weights.write().unwrap();
        
        // Add resource policy
        policies.insert(
            PolicyType::ResourcePolicy,
            Box::new(ResourcePolicy::new()?),
        );
        weights.insert(PolicyType::ResourcePolicy, 0.3);
        
        // Add load balancing policy
        policies.insert(
            PolicyType::LoadBalancingPolicy,
            Box::new(LoadBalancingPolicy::new()?),
        );
        weights.insert(PolicyType::LoadBalancingPolicy, 0.2);
        
        // Add priority policy
        policies.insert(
            PolicyType::PriorityPolicy,
            Box::new(PriorityPolicy::new()?),
        );
        weights.insert(PolicyType::PriorityPolicy, 0.2);
        
        // Add energy policy
        policies.insert(
            PolicyType::EnergyPolicy,
            Box::new(EnergyPolicy::new()?),
        );
        weights.insert(PolicyType::EnergyPolicy, 0.1);
        
        // Add latency policy
        policies.insert(
            PolicyType::LatencyPolicy,
            Box::new(LatencyPolicy::new()?),
        );
        weights.insert(PolicyType::LatencyPolicy, 0.1);
        
        // Add throughput policy
        policies.insert(
            PolicyType::ThroughputPolicy,
            Box::new(ThroughputPolicy::new()?),
        );
        weights.insert(PolicyType::ThroughputPolicy, 0.1);
        
        Ok(())
    }
    
    /// Convert task to generic type
    fn convert_task(&self, task: &Task<T>) -> Task<()> {
        Task {
            operation: TaskOperation::TensorOperation {
                operation: TensorOp::Add,
                inputs: vec![],
                output_shape: Shape::new(vec![1]),
            },
            priority: task.priority,
            resource_requirements: task.resource_requirements.clone(),
            device_requirements: task.device_requirements.clone(),
            timeout: task.timeout,
            retry_count: task.retry_count,
            max_retries: task.max_retries,
        }
    }
    
    /// Record policy decision
    fn record_decision(&self, decisions: Vec<PolicyDecision>, final_decision: bool, decision_time: Duration) -> Result<()> {
        // Update metrics
        {
            let mut metrics = self.policy_metrics.write().unwrap();
            metrics.total_decisions += 1;
            if final_decision {
                metrics.successful_decisions += 1;
            }
            metrics.decision_time = decision_time;
        }
        
        // Add to history
        {
            let mut history = self.policy_history.write().unwrap();
            for decision in decisions {
                history.push_back(decision);
            }
            
            // Keep only recent history
            if history.len() > 1000 {
                history.pop_front();
            }
        }
        
        Ok(())
    }
}

/// Resource policy
pub struct ResourcePolicy {
    memory_threshold: f32,
    compute_threshold: f32,
    bandwidth_threshold: f32,
}

impl ResourcePolicy {
    pub fn new() -> Result<Self> {
        Ok(Self {
            memory_threshold: 0.9,
            compute_threshold: 0.9,
            bandwidth_threshold: 0.9,
        })
    }
}

impl SchedulingPolicy for ResourcePolicy {
    fn should_schedule(&self, task: &Task<()>, device: &Device) -> Result<bool> {
        // Check if device has sufficient resources
        // This is a simplified check - in practice, you'd check actual device resources
        Ok(true)
    }
    
    fn get_priority(&self, task: &Task<()>, device: &Device) -> Result<f32> {
        // Calculate priority based on resource requirements
        let memory_priority = if task.resource_requirements.memory > 1024 * 1024 * 1024 { // 1GB
            0.5
        } else {
            1.0
        };
        
        let compute_priority = if task.resource_requirements.compute > 10.0 {
            0.5
        } else {
            1.0
        };
        
        Ok((memory_priority + compute_priority) / 2.0)
    }
    
    fn get_policy_name(&self) -> &str {
        "ResourcePolicy"
    }
    
    fn get_parameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("memory_threshold".to_string(), self.memory_threshold);
        params.insert("compute_threshold".to_string(), self.compute_threshold);
        params.insert("bandwidth_threshold".to_string(), self.bandwidth_threshold);
        params
    }
}

/// Load balancing policy
pub struct LoadBalancingPolicy {
    load_threshold: f32,
    balance_tolerance: f32,
}

impl LoadBalancingPolicy {
    pub fn new() -> Result<Self> {
        Ok(Self {
            load_threshold: 0.8,
            balance_tolerance: 0.2,
        })
    }
}

impl SchedulingPolicy for LoadBalancingPolicy {
    fn should_schedule(&self, task: &Task<()>, device: &Device) -> Result<bool> {
        // Check if device is not overloaded
        // This is a simplified check - in practice, you'd check actual device load
        Ok(true)
    }
    
    fn get_priority(&self, task: &Task<()>, device: &Device) -> Result<f32> {
        // Calculate priority based on load balancing
        // Lower load = higher priority
        0.8 // Placeholder
    }
    
    fn get_policy_name(&self) -> &str {
        "LoadBalancingPolicy"
    }
    
    fn get_parameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("load_threshold".to_string(), self.load_threshold);
        params.insert("balance_tolerance".to_string(), self.balance_tolerance);
        params
    }
}

/// Priority policy
pub struct PriorityPolicy {
    priority_weights: HashMap<TaskPriority, f32>,
}

impl PriorityPolicy {
    pub fn new() -> Result<Self> {
        let mut priority_weights = HashMap::new();
        priority_weights.insert(TaskPriority::Low, 0.2);
        priority_weights.insert(TaskPriority::Normal, 0.5);
        priority_weights.insert(TaskPriority::High, 0.8);
        priority_weights.insert(TaskPriority::Critical, 1.0);
        
        Ok(Self {
            priority_weights,
        })
    }
}

impl SchedulingPolicy for PriorityPolicy {
    fn should_schedule(&self, task: &Task<()>, device: &Device) -> Result<bool> {
        // Always allow scheduling based on priority
        Ok(true)
    }
    
    fn get_priority(&self, task: &Task<()>, device: &Device) -> Result<f32> {
        Ok(self.priority_weights.get(&task.priority).copied().unwrap_or(0.5))
    }
    
    fn get_policy_name(&self) -> &str {
        "PriorityPolicy"
    }
    
    fn get_parameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        for (priority, weight) in &self.priority_weights {
            params.insert(format!("priority_{:?}", priority), *weight);
        }
        params
    }
}

/// Energy policy
pub struct EnergyPolicy {
    energy_threshold: f32,
    efficiency_weight: f32,
}

impl EnergyPolicy {
    pub fn new() -> Result<Self> {
        Ok(Self {
            energy_threshold: 0.8,
            efficiency_weight: 0.7,
        })
    }
}

impl SchedulingPolicy for EnergyPolicy {
    fn should_schedule(&self, task: &Task<()>, device: &Device) -> Result<bool> {
        // Check if device is energy efficient
        // This is a simplified check - in practice, you'd check actual energy consumption
        Ok(true)
    }
    
    fn get_priority(&self, task: &Task<()>, device: &Device) -> Result<f32> {
        // Calculate priority based on energy efficiency
        // More efficient devices get higher priority
        0.6 // Placeholder
    }
    
    fn get_policy_name(&self) -> &str {
        "EnergyPolicy"
    }
    
    fn get_parameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("energy_threshold".to_string(), self.energy_threshold);
        params.insert("efficiency_weight".to_string(), self.efficiency_weight);
        params
    }
}

/// Latency policy
pub struct LatencyPolicy {
    latency_threshold: Duration,
    latency_weight: f32,
}

impl LatencyPolicy {
    pub fn new() -> Result<Self> {
        Ok(Self {
            latency_threshold: Duration::from_millis(100),
            latency_weight: 0.8,
        })
    }
}

impl SchedulingPolicy for LatencyPolicy {
    fn should_schedule(&self, task: &Task<()>, device: &Device) -> Result<bool> {
        // Check if device can meet latency requirements
        // This is a simplified check - in practice, you'd check actual device latency
        Ok(true)
    }
    
    fn get_priority(&self, task: &Task<()>, device: &Device) -> Result<f32> {
        // Calculate priority based on latency requirements
        // Lower latency devices get higher priority
        0.7 // Placeholder
    }
    
    fn get_policy_name(&self) -> &str {
        "LatencyPolicy"
    }
    
    fn get_parameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("latency_threshold".to_string(), self.latency_threshold.as_secs_f32());
        params.insert("latency_weight".to_string(), self.latency_weight);
        params
    }
}

/// Throughput policy
pub struct ThroughputPolicy {
    throughput_threshold: f32,
    throughput_weight: f32,
}

impl ThroughputPolicy {
    pub fn new() -> Result<Self> {
        Ok(Self {
            throughput_threshold: 100.0,
            throughput_weight: 0.9,
        })
    }
}

impl SchedulingPolicy for ThroughputPolicy {
    fn should_schedule(&self, task: &Task<()>, device: &Device) -> Result<bool> {
        // Check if device can meet throughput requirements
        // This is a simplified check - in practice, you'd check actual device throughput
        Ok(true)
    }
    
    fn get_priority(&self, task: &Task<()>, device: &Device) -> Result<f32> {
        // Calculate priority based on throughput requirements
        // Higher throughput devices get higher priority
        0.8 // Placeholder
    }
    
    fn get_policy_name(&self) -> &str {
        "ThroughputPolicy"
    }
    
    fn get_parameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("throughput_threshold".to_string(), self.throughput_threshold);
        params.insert("throughput_weight".to_string(), self.throughput_weight);
        params
    }
}
