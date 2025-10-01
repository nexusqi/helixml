//! 🌀 HelixML Adaptive Scheduler
//! 
//! Multi-device orchestration with topological awareness and learning capabilities.

use crate::{DeviceType, OperationType, Result, HalError};
use crate::backend::{ComputeBackend, BackendRegistry};
use crate::operations::{Operation, ComputeGraph};
// use crate::memory::UnifiedTensor;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Adaptive scheduler for multi-device execution
pub struct AdaptiveScheduler {
    /// Available backends
    backends: BackendRegistry,
    /// Execution history for learning
    execution_history: ExecutionHistory,
    /// Scheduling strategy
    strategy: SchedulingStrategy,
    /// Topological awareness
    topology_aware: bool,
}

impl AdaptiveScheduler {
    /// Create new adaptive scheduler
    pub fn new() -> Self {
        Self {
            backends: BackendRegistry::new(),
            execution_history: ExecutionHistory::new(),
            strategy: SchedulingStrategy::Performance,
            topology_aware: true,
        }
    }
    
    /// Register backend
    pub fn register_backend(&mut self, backend: Box<dyn ComputeBackend>) {
        self.backends.register(backend);
    }
    
    /// Schedule computation graph
    pub fn schedule(&mut self, graph: &ComputeGraph) -> Result<ExecutionPlan> {
        let mut plan = ExecutionPlan::new();
        
        // Analyze semantic regions
        let regions = self.analyze_semantic_regions(graph)?;
        
        // Schedule each operation
        for (node_id, op) in graph.nodes().iter().enumerate() {
            let region = regions.get(&node_id);
            let backend = self.select_backend(op, region)?;
            let stage = self.create_execution_stage(op, backend)?;
            plan.add_stage(stage);
        }
        
        // Optimize plan
        self.optimize_plan(&mut plan)?;
        
        Ok(plan)
    }
    
    /// Analyze semantic regions in graph
    fn analyze_semantic_regions(&self, graph: &ComputeGraph) -> Result<HashMap<usize, SemanticRegion>> {
        let mut regions = HashMap::new();
        
        for (node_id, op) in graph.nodes().iter().enumerate() {
            let region = self.classify_operation(op)?;
            regions.insert(node_id, region);
        }
        
        Ok(regions)
    }
    
    /// Classify operation into semantic region
    fn classify_operation(&self, op: &Operation) -> Result<SemanticRegion> {
        match op.op_type {
            OperationType::SSMForward | OperationType::SSMBackward => {
                Ok(SemanticRegion::StateSpace)
            },
            OperationType::Conv1D | OperationType::Conv2D | OperationType::Conv3D => {
                Ok(SemanticRegion::Convolution)
            },
            OperationType::FFT | OperationType::IFFT => {
                Ok(SemanticRegion::Spectral)
            },
            OperationType::TopologicalAnalysis | OperationType::MotifDetection => {
                Ok(SemanticRegion::Topological)
            },
            _ => Ok(SemanticRegion::General),
        }
    }
    
    /// Select optimal backend for operation
    fn select_backend(&self, op: &Operation, region: Option<&SemanticRegion>) -> Result<&dyn ComputeBackend> {
        // Get backends that support the operation
        let supporting: Vec<_> = self.backends.find_supporting(op.op_type);
        
        if supporting.is_empty() {
            return Err(HalError::SchedulerError {
                message: format!("No backend supports operation {:?}", op.op_type),
            });
        }
        
        // Select based on strategy and region
        match self.strategy {
            SchedulingStrategy::Performance => {
                self.select_by_performance(&supporting, op, region)
            },
            SchedulingStrategy::Memory => {
                self.select_by_memory(&supporting, op, region)
            },
            SchedulingStrategy::Balanced => {
                self.select_balanced(&supporting, op, region)
            },
        }
    }
    
    /// Select backend by performance
    fn select_by_performance<'a>(&self, backends: &[&'a dyn ComputeBackend], _op: &Operation, region: Option<&SemanticRegion>) -> Result<&'a dyn ComputeBackend> {
        // Prefer specialized backends for specific regions
        if let Some(region) = region {
            if let Some(specialized) = self.find_specialized_backend(backends, region) {
                return Ok(specialized);
            }
        }
        
        // Fall back to highest performance
        backends.iter()
            .max_by_key(|b| b.capabilities().peak_flops as u64)
            .map(|b| *b)
            .ok_or_else(|| HalError::SchedulerError {
                message: "No suitable backend found".to_string(),
            })
    }
    
    /// Select backend by memory efficiency
    fn select_by_memory<'a>(&self, backends: &[&'a dyn ComputeBackend], _op: &Operation, _region: Option<&SemanticRegion>) -> Result<&'a dyn ComputeBackend> {
        // Prefer backend with most available memory
        backends.iter()
            .max_by_key(|b| b.capabilities().available_memory)
            .map(|b| *b)
            .ok_or_else(|| HalError::SchedulerError {
                message: "No suitable backend found".to_string(),
            })
    }
    
    /// Select balanced backend
    fn select_balanced<'a>(&self, backends: &[&'a dyn ComputeBackend], _op: &Operation, _region: Option<&SemanticRegion>) -> Result<&'a dyn ComputeBackend> {
        // Consider both performance and memory
        let scores: Vec<_> = backends.iter()
            .map(|b| {
                let perf_score = b.capabilities().peak_flops;
                let mem_score = b.capabilities().available_memory as f64;
                let util_score = 1.0 - b.utilization();
                perf_score * mem_score * util_score
            })
            .collect();
        
        let max_idx = scores.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| HalError::SchedulerError {
                message: "No suitable backend found".to_string(),
            })?;
        
        Ok(backends[max_idx])
    }
    
    /// Find specialized backend for region
    fn find_specialized_backend<'a>(&self, backends: &[&'a dyn ComputeBackend], region: &SemanticRegion) -> Option<&'a dyn ComputeBackend> {
        match region {
            SemanticRegion::StateSpace => {
                // Prefer NPU/TPU for state-space operations
                backends.iter().find(|b| matches!(b.device_type(), DeviceType::NPU | DeviceType::TPU)).map(|v| &**v)
            },
            SemanticRegion::Spectral => {
                // Prefer GPU for FFT operations
                backends.iter().find(|b| matches!(b.device_type(), DeviceType::CUDA | DeviceType::Metal)).map(|v| &**v)
            },
            SemanticRegion::Topological => {
                // Prefer CPU for topological analysis
                backends.iter().find(|b| b.device_type() == DeviceType::CPU).map(|v| &**v)
            },
            _ => None,
        }
    }
    
    /// Create execution stage
    fn create_execution_stage(&self, op: &Operation, backend: &dyn ComputeBackend) -> Result<ExecutionStage> {
        Ok(ExecutionStage {
            operations: vec![ScheduledOperation {
                op: op.clone(),
                device: backend.device_type(),
                precision: op.precision.unwrap_or(crate::DataType::F32),
                batch_size: op.batch_size.unwrap_or(1),
                fusion_group: op.fusion_group,
            }],
            device: backend.device_type(),
            can_parallelize: backend.capabilities().compute_units > 1,
            dependencies: Vec::new(),
        })
    }
    
    /// Optimize execution plan
    fn optimize_plan(&self, _plan: &mut ExecutionPlan) -> Result<()> {
        // TODO: Implement plan optimization
        // - Kernel fusion
        // - Memory transfer optimization
        // - Parallel execution opportunities
        Ok(())
    }
    
    /// Record execution results for learning
    pub fn record_execution(&mut self, plan: &ExecutionPlan, metrics: &ExecutionMetrics) {
        self.execution_history.record(plan, metrics);
    }
    
    /// Set scheduling strategy
    pub fn set_strategy(&mut self, strategy: SchedulingStrategy) {
        self.strategy = strategy;
    }
    
    /// Enable/disable topological awareness
    pub fn set_topology_aware(&mut self, aware: bool) {
        self.topology_aware = aware;
    }
}

/// Scheduling strategy
#[derive(Debug, Clone, Copy)]
pub enum SchedulingStrategy {
    /// Optimize for performance
    Performance,
    /// Optimize for memory efficiency
    Memory,
    /// Balanced approach
    Balanced,
}

/// Semantic region classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticRegion {
    /// State-space model operations
    StateSpace,
    /// Convolution operations
    Convolution,
    /// Spectral/FFT operations
    Spectral,
    /// Topological analysis
    Topological,
    /// General purpose
    General,
}

/// Execution plan
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Execution stages
    stages: Vec<ExecutionStage>,
    /// Data transfers
    transfers: Vec<DataTransfer>,
    /// Estimated metrics
    estimated_time: Duration,
    estimated_memory: usize,
}

impl ExecutionPlan {
    /// Create new execution plan
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            transfers: Vec::new(),
            estimated_time: Duration::from_secs(0),
            estimated_memory: 0,
        }
    }
    
    /// Add execution stage
    pub fn add_stage(&mut self, stage: ExecutionStage) {
        self.stages.push(stage);
    }
    
    /// Add data transfer
    pub fn add_transfer(&mut self, transfer: DataTransfer) {
        self.transfers.push(transfer);
    }
    
    /// Get stages
    pub fn stages(&self) -> &[ExecutionStage] {
        &self.stages
    }
    
    /// Get transfers
    pub fn transfers(&self) -> &[DataTransfer] {
        &self.transfers
    }
}

/// Execution stage
#[derive(Debug, Clone)]
pub struct ExecutionStage {
    /// Operations in this stage
    operations: Vec<ScheduledOperation>,
    /// Target device
    device: DeviceType,
    /// Can parallelize operations
    can_parallelize: bool,
    /// Stage dependencies
    dependencies: Vec<usize>,
}

/// Scheduled operation
#[derive(Debug, Clone)]
pub struct ScheduledOperation {
    /// Operation to execute
    op: Operation,
    /// Target device
    device: DeviceType,
    /// Precision
    precision: crate::DataType,
    /// Batch size
    batch_size: usize,
    /// Fusion group
    fusion_group: Option<usize>,
}

/// Data transfer between devices
#[derive(Debug, Clone)]
pub struct DataTransfer {
    /// Source device
    source: DeviceType,
    /// Destination device
    destination: DeviceType,
    /// Data size
    size: usize,
    /// Transfer time
    estimated_time: Duration,
}

/// Execution history for learning
pub struct ExecutionHistory {
    records: Vec<ExecutionRecord>,
    max_records: usize,
}

/// Execution record
#[derive(Debug, Clone)]
struct ExecutionRecord {
    plan: ExecutionPlan,
    metrics: ExecutionMetrics,
    timestamp: Instant,
}

/// Execution metrics
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Actual execution time
    execution_time: Duration,
    /// Memory used
    memory_used: usize,
    /// Throughput (operations per second)
    throughput: f64,
    /// Energy consumed (if available)
    energy_consumed: Option<f64>,
}

impl ExecutionHistory {
    /// Create new execution history
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            max_records: 1000,
        }
    }
    
    /// Record execution
    pub fn record(&mut self, plan: &ExecutionPlan, metrics: &ExecutionMetrics) {
        let record = ExecutionRecord {
            plan: plan.clone(),
            metrics: metrics.clone(),
            timestamp: Instant::now(),
        };
        
        self.records.push(record);
        
        // Keep only recent records
        if self.records.len() > self.max_records {
            self.records.remove(0);
        }
    }
}
