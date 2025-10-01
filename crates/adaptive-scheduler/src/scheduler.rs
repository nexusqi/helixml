//! 🎯 Core Scheduler Implementation
//! 
//! Core scheduling algorithms and task management for the adaptive scheduler

use tensor_core::{Tensor, Shape, DType, Device, Result};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision, TensorStats, TensorReduce};
use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use anyhow::Context;
use petgraph::{Graph, Directed, NodeIndex};
use petgraph::algo::toposort;
use priority_queue::PriorityQueue;

use super::*;

/// Core scheduler implementation
#[derive(Debug)]
pub struct CoreScheduler<T: Tensor> {
    task_graph: Arc<RwLock<Graph<TaskNode<T>, TaskDependency>>>,
    execution_queue: Arc<Mutex<PriorityQueue<TaskId, TaskPriority>>>,
    running_tasks: Arc<Mutex<HashMap<TaskId, RunningTaskInfo<T>>>>,
    completed_tasks: Arc<Mutex<HashMap<TaskId, TaskResult<T>>>>,
    failed_tasks: Arc<Mutex<HashMap<TaskId, String>>>,
    scheduler_state: Arc<Mutex<SchedulerState>>,
}

/// Task node in the execution graph
#[derive(Debug, Clone)]
pub struct TaskNode<T: Tensor> {
    pub task_id: TaskId,
    pub task: Task<T>,
    pub dependencies: Vec<TaskId>,
    pub dependents: Vec<TaskId>,
    pub status: TaskStatus,
    pub created_at: Instant,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
}

/// Task dependency information
#[derive(Debug, Clone)]
pub struct TaskDependency {
    pub dependency_type: DependencyType,
    pub weight: f32,
    pub critical_path: bool,
}

/// Dependency types
#[derive(Debug, Clone)]
pub enum DependencyType {
    DataDependency,
    ResourceDependency,
    TemporalDependency,
    LogicalDependency,
}

/// Running task information
#[derive(Debug, Clone)]
pub struct RunningTaskInfo<T: Tensor> {
    pub task: Task<T>,
    pub device: Device,
    pub started_at: Instant,
    pub estimated_completion: Instant,
    pub resource_allocation: ResourceAllocation,
}

/// Resource allocation
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub memory: usize,
    pub compute: f32,
    pub bandwidth: f32,
    pub storage: usize,
}

/// Scheduler state
#[derive(Debug, Clone)]
pub struct SchedulerState {
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub running_tasks: usize,
    pub pending_tasks: usize,
    pub scheduler_start_time: Instant,
    pub last_optimization: Instant,
    pub optimization_count: usize,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorStats + TensorReduce> CoreScheduler<T> {
    pub fn new() -> Result<Self> {
        let task_graph = Arc::new(RwLock::new(Graph::new()));
        let execution_queue = Arc::new(Mutex::new(PriorityQueue::new()));
        let running_tasks = Arc::new(Mutex::new(HashMap::new()));
        let completed_tasks = Arc::new(Mutex::new(HashMap::new()));
        let failed_tasks = Arc::new(Mutex::new(HashMap::new()));
        let scheduler_state = Arc::new(Mutex::new(SchedulerState {
            total_tasks: 0,
            completed_tasks: 0,
            failed_tasks: 0,
            running_tasks: 0,
            pending_tasks: 0,
            scheduler_start_time: Instant::now(),
            last_optimization: Instant::now(),
            optimization_count: 0,
        }));
        
        Ok(Self {
            task_graph,
            execution_queue,
            running_tasks,
            completed_tasks,
            failed_tasks,
            scheduler_state,
        })
    }
    
    /// Add a task to the scheduler
    pub fn add_task(&self, task: Task<T>) -> Result<TaskId> {
        let task_id = TaskId::new();
        let task_node = TaskNode {
            task_id: task_id.clone(),
            task: task.clone(),
            dependencies: vec![],
            dependents: vec![],
            status: TaskStatus::Pending,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
        };
        
        // Add to task graph
        {
            let mut graph = self.task_graph.write().unwrap();
            let node_index = graph.add_node(task_node);
            // Store node index mapping if needed
        }
        
        // Add to execution queue
        {
            let mut queue = self.execution_queue.lock().unwrap();
            queue.push(task_id.clone(), task.priority);
        }
        
        // Update scheduler state
        {
            let mut state = self.scheduler_state.lock().unwrap();
            state.total_tasks += 1;
            state.pending_tasks += 1;
        }
        
        Ok(task_id)
    }
    
    /// Add a task with dependencies
    pub fn add_task_with_dependencies(&self, task: Task<T>, dependencies: Vec<TaskId>) -> Result<TaskId> {
        let task_id = self.add_task(task)?;
        
        // Add dependencies to the graph
        {
            let mut graph = self.task_graph.write().unwrap();
            // Find the node for this task
            let task_node_index = self.find_node_index(&graph, &task_id)?;
            
            // Add dependency edges
            for dep_id in dependencies {
                if let Some(dep_node_index) = self.find_node_index(&graph, &dep_id) {
                    graph.add_edge(dep_node_index, task_node_index, TaskDependency {
                        dependency_type: DependencyType::DataDependency,
                        weight: 1.0,
                        critical_path: false,
                    });
                }
            }
        }
        
        Ok(task_id)
    }
    
    /// Get next task to execute
    pub fn get_next_task(&self) -> Result<Option<(TaskId, Task<T>)>> {
        let mut queue = self.execution_queue.lock().unwrap();
        let graph = self.task_graph.read().unwrap();
        
        // Find tasks that are ready to execute (dependencies satisfied)
        let ready_tasks = self.find_ready_tasks(&graph)?;
        
        if ready_tasks.is_empty() {
            return Ok(None);
        }
        
        // Select highest priority ready task
        let mut best_task = None;
        let mut best_priority = TaskPriority::Low;
        
        for task_id in ready_tasks {
            if let Some(priority) = queue.get_priority(&task_id) {
                if *priority > best_priority {
                    best_priority = priority.clone();
                    best_task = Some(task_id);
                }
            }
        }
        
        if let Some(task_id) = best_task {
            queue.remove(&task_id);
            
            // Get task from graph
            if let Some(task_node) = self.get_task_node(&graph, &task_id)? {
                return Ok(Some((task_id, task_node.task)));
            }
        }
        
        Ok(None)
    }
    
    /// Mark task as running
    pub fn mark_task_running(&self, task_id: &TaskId, device: Device, resource_allocation: ResourceAllocation) -> Result<()> {
        // Update task status in graph
        {
            let mut graph = self.task_graph.write().unwrap();
            if let Some(node_index) = self.find_node_index(&graph, task_id) {
                if let Some(node) = graph.node_weight_mut(node_index) {
                    node.status = TaskStatus::Running;
                    node.started_at = Some(Instant::now());
                }
            }
        }
        
        // Add to running tasks
        {
            let mut running_tasks = self.running_tasks.lock().unwrap();
            if let Some(task_node) = self.get_task_node(&self.task_graph.read().unwrap(), task_id)? {
                let running_info = RunningTaskInfo {
                    task: task_node.task,
                    device,
                    started_at: Instant::now(),
                    estimated_completion: Instant::now() + Duration::from_secs(60), // Placeholder
                    resource_allocation,
                };
                running_tasks.insert(task_id.clone(), running_info);
            }
        }
        
        // Update scheduler state
        {
            let mut state = self.scheduler_state.lock().unwrap();
            state.running_tasks += 1;
            state.pending_tasks -= 1;
        }
        
        Ok(())
    }
    
    /// Mark task as completed
    pub fn mark_task_completed(&self, task_id: &TaskId, result: TaskResult<T>) -> Result<()> {
        // Update task status in graph
        {
            let mut graph = self.task_graph.write().unwrap();
            if let Some(node_index) = self.find_node_index(&graph, task_id) {
                if let Some(node) = graph.node_weight_mut(node_index) {
                    node.status = TaskStatus::Completed;
                    node.completed_at = Some(Instant::now());
                }
            }
        }
        
        // Remove from running tasks
        {
            let mut running_tasks = self.running_tasks.lock().unwrap();
            running_tasks.remove(task_id);
        }
        
        // Add to completed tasks
        {
            let mut completed_tasks = self.completed_tasks.lock().unwrap();
            completed_tasks.insert(task_id.clone(), result);
        }
        
        // Update scheduler state
        {
            let mut state = self.scheduler_state.lock().unwrap();
            state.completed_tasks += 1;
            state.running_tasks -= 1;
        }
        
        Ok(())
    }
    
    /// Mark task as failed
    pub fn mark_task_failed(&self, task_id: &TaskId, error: String) -> Result<()> {
        // Update task status in graph
        {
            let mut graph = self.task_graph.write().unwrap();
            if let Some(node_index) = self.find_node_index(&graph, task_id) {
                if let Some(node) = graph.node_weight_mut(node_index) {
                    node.status = TaskStatus::Failed;
                    node.completed_at = Some(Instant::now());
                }
            }
        }
        
        // Remove from running tasks
        {
            let mut running_tasks = self.running_tasks.lock().unwrap();
            running_tasks.remove(task_id);
        }
        
        // Add to failed tasks
        {
            let mut failed_tasks = self.failed_tasks.lock().unwrap();
            failed_tasks.insert(task_id.clone(), error);
        }
        
        // Update scheduler state
        {
            let mut state = self.scheduler_state.lock().unwrap();
            state.failed_tasks += 1;
            state.running_tasks -= 1;
        }
        
        Ok(())
    }
    
    /// Get task status
    pub fn get_task_status(&self, task_id: &TaskId) -> Result<TaskStatus> {
        let graph = self.task_graph.read().unwrap();
        if let Some(task_node) = self.get_task_node(&graph, task_id)? {
            Ok(task_node.status)
        } else {
            Err(anyhow::anyhow!("Task not found"))
        }
    }
    
    /// Get task result
    pub fn get_task_result(&self, task_id: &TaskId) -> Result<Option<TaskResult<T>>> {
        let completed_tasks = self.completed_tasks.lock().unwrap();
        Ok(completed_tasks.get(task_id).cloned())
    }
    
    /// Get scheduler state
    pub fn get_scheduler_state(&self) -> Result<SchedulerState> {
        let state = self.scheduler_state.lock().unwrap();
        Ok(state.clone())
    }
    
    /// Get running tasks
    pub fn get_running_tasks(&self) -> Result<HashMap<TaskId, RunningTaskInfo<T>>> {
        let running_tasks = self.running_tasks.lock().unwrap();
        Ok(running_tasks.clone())
    }
    
    /// Find ready tasks (dependencies satisfied)
    fn find_ready_tasks(&self, graph: &Graph<TaskNode<T>, TaskDependency>) -> Result<Vec<TaskId>> {
        let mut ready_tasks = Vec::new();
        
        for node_index in graph.node_indices() {
            if let Some(node) = graph.node_weight(node_index) {
                if node.status == TaskStatus::Pending {
                    // Check if all dependencies are completed
                    let mut all_deps_completed = true;
                    for dep_edge in graph.edges_directed(node_index, petgraph::Direction::Incoming) {
                        if let Some(dep_node) = graph.node_weight(dep_edge.source()) {
                            if dep_node.status != TaskStatus::Completed {
                                all_deps_completed = false;
                                break;
                            }
                        }
                    }
                    
                    if all_deps_completed {
                        ready_tasks.push(node.task_id.clone());
                    }
                }
            }
        }
        
        Ok(ready_tasks)
    }
    
    /// Find node index for a task ID
    fn find_node_index(&self, graph: &Graph<TaskNode<T>, TaskDependency>, task_id: &TaskId) -> Result<Option<NodeIndex>> {
        for node_index in graph.node_indices() {
            if let Some(node) = graph.node_weight(node_index) {
                if node.task_id == *task_id {
                    return Ok(Some(node_index));
                }
            }
        }
        Ok(None)
    }
    
    /// Get task node by ID
    fn get_task_node(&self, graph: &Graph<TaskNode<T>, TaskDependency>, task_id: &TaskId) -> Result<Option<TaskNode<T>>> {
        if let Some(node_index) = self.find_node_index(graph, task_id)? {
            if let Some(node) = graph.node_weight(node_index) {
                return Ok(Some(node.clone()));
            }
        }
        Ok(None)
    }
    
    /// Optimize task scheduling
    pub fn optimize_scheduling(&self) -> Result<OptimizationResult> {
        let graph = self.task_graph.read().unwrap();
        let state = self.scheduler_state.lock().unwrap();
        
        // Analyze critical path
        let critical_path = self.analyze_critical_path(&graph)?;
        
        // Identify bottlenecks
        let bottlenecks = self.identify_bottlenecks(&graph)?;
        
        // Suggest optimizations
        let optimizations = self.suggest_optimizations(&graph, &critical_path, &bottlenecks)?;
        
        Ok(OptimizationResult {
            critical_path,
            bottlenecks,
            optimizations,
            estimated_improvement: self.estimate_improvement(&optimizations)?,
        })
    }
    
    /// Analyze critical path
    fn analyze_critical_path(&self, graph: &Graph<TaskNode<T>, TaskDependency>) -> Result<Vec<TaskId>> {
        // Find topological order
        let topo_order = toposort(&graph, None)
            .map_err(|_| anyhow::anyhow!("Graph contains cycles"))?;
        
        // Calculate longest path
        let mut distances = HashMap::new();
        let mut critical_path = Vec::new();
        
        for node_index in topo_order {
            let mut max_distance = 0.0;
            let mut predecessor = None;
            
            for edge in graph.edges_directed(node_index, petgraph::Direction::Incoming) {
                if let Some(pred_node) = graph.node_weight(edge.source()) {
                    let pred_distance = distances.get(&edge.source()).unwrap_or(&0.0);
                    let edge_weight = edge.weight().weight;
                    let total_distance = pred_distance + edge_weight;
                    
                    if total_distance > max_distance {
                        max_distance = total_distance;
                        predecessor = Some(edge.source());
                    }
                }
            }
            
            distances.insert(node_index, max_distance);
            
            if max_distance > 0.0 {
                if let Some(pred) = predecessor {
                    critical_path.push(pred);
                }
            }
        }
        
        // Convert node indices to task IDs
        let mut critical_task_ids = Vec::new();
        for node_index in critical_path {
            if let Some(node) = graph.node_weight(node_index) {
                critical_task_ids.push(node.task_id.clone());
            }
        }
        
        Ok(critical_task_ids)
    }
    
    /// Identify bottlenecks
    fn identify_bottlenecks(&self, graph: &Graph<TaskNode<T>, TaskDependency>) -> Result<Vec<Bottleneck>> {
        let mut bottlenecks = Vec::new();
        
        for node_index in graph.node_indices() {
            if let Some(node) = graph.node_weight(node_index) {
                // Check for high resource requirements
                if node.task.resource_requirements.memory > 1024 * 1024 * 1024 { // 1GB
                    bottlenecks.push(Bottleneck {
                        task_id: node.task_id.clone(),
                        bottleneck_type: BottleneckType::HighMemoryUsage,
                        severity: 0.8,
                        description: "High memory usage".to_string(),
                    });
                }
                
                // Check for many dependencies
                let incoming_edges = graph.edges_directed(node_index, petgraph::Direction::Incoming).count();
                if incoming_edges > 5 {
                    bottlenecks.push(Bottleneck {
                        task_id: node.task_id.clone(),
                        bottleneck_type: BottleneckType::HighDependencyCount,
                        severity: 0.6,
                        description: "High dependency count".to_string(),
                    });
                }
            }
        }
        
        Ok(bottlenecks)
    }
    
    /// Suggest optimizations
    fn suggest_optimizations(
        &self,
        graph: &Graph<TaskNode<T>, TaskDependency>,
        critical_path: &[TaskId],
        bottlenecks: &[Bottleneck],
    ) -> Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Suggest parallelization for independent tasks
        for node_index in graph.node_indices() {
            if let Some(node) = graph.node_weight(node_index) {
                let incoming_edges = graph.edges_directed(node_index, petgraph::Direction::Incoming).count();
                if incoming_edges == 0 && node.status == TaskStatus::Pending {
                    suggestions.push(OptimizationSuggestion {
                        suggestion_type: OptimizationType::Parallelization,
                        task_id: Some(node.task_id.clone()),
                        description: "Task can be parallelized".to_string(),
                        estimated_improvement: 0.3,
                    });
                }
            }
        }
        
        // Suggest resource optimization for bottlenecks
        for bottleneck in bottlenecks {
            if bottleneck.bottleneck_type == BottleneckType::HighMemoryUsage {
                suggestions.push(OptimizationSuggestion {
                    suggestion_type: OptimizationType::ResourceOptimization,
                    task_id: Some(bottleneck.task_id.clone()),
                    description: "Optimize memory usage".to_string(),
                    estimated_improvement: 0.2,
                });
            }
        }
        
        Ok(suggestions)
    }
    
    /// Estimate improvement from optimizations
    fn estimate_improvement(&self, optimizations: &[OptimizationSuggestion]) -> Result<f32> {
        let total_improvement: f32 = optimizations.iter()
            .map(|opt| opt.estimated_improvement)
            .sum();
        
        Ok(total_improvement.min(1.0))
    }
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub critical_path: Vec<TaskId>,
    pub bottlenecks: Vec<Bottleneck>,
    pub optimizations: Vec<OptimizationSuggestion>,
    pub estimated_improvement: f32,
}

/// Bottleneck information
#[derive(Debug, Clone)]
pub struct Bottleneck {
    pub task_id: TaskId,
    pub bottleneck_type: BottleneckType,
    pub severity: f32,
    pub description: String,
}

/// Bottleneck types
#[derive(Debug, Clone)]
pub enum BottleneckType {
    HighMemoryUsage,
    HighComputeUsage,
    HighDependencyCount,
    ResourceContention,
    NetworkLatency,
}

/// Optimization suggestion
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    pub suggestion_type: OptimizationType,
    pub task_id: Option<TaskId>,
    pub description: String,
    pub estimated_improvement: f32,
}

/// Optimization types
#[derive(Debug, Clone)]
pub enum OptimizationType {
    Parallelization,
    ResourceOptimization,
    DependencyOptimization,
    LoadBalancing,
    Caching,
    Prefetching,
}
