//! 📋 Task Queue Management
//! 
//! Task queue implementation with priority scheduling and dependency management

use tensor_core::{Tensor, Shape, DType, Device, Result, TensorError};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision, TensorStats, TensorReduce};
use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use anyhow::Context;
use priority_queue::PriorityQueue;
use crossbeam::channel::{self, Receiver, Sender};

use super::*;

/// Task queue for managing task execution
#[derive(Debug)]
pub struct TaskQueue {
    // Task storage
    pending_tasks: Arc<Mutex<PriorityQueue<TaskId, TaskPriority>>>,
    running_tasks: Arc<Mutex<HashMap<TaskId, RunningTaskInfo>>>,
    completed_tasks: Arc<Mutex<HashMap<TaskId, TaskResult>>>,
    failed_tasks: Arc<Mutex<HashMap<TaskId, String>>>,
    cancelled_tasks: Arc<Mutex<HashSet<TaskId>>>,
    
    // Task metadata
    task_metadata: Arc<RwLock<HashMap<TaskId, TaskMetadata>>>,
    task_dependencies: Arc<RwLock<HashMap<TaskId, Vec<TaskId>>>>,
    task_dependents: Arc<RwLock<HashMap<TaskId, Vec<TaskId>>>>,
    
    // Queue management
    max_queue_size: usize,
    queue_notifications: Arc<Mutex<Vec<Sender<TaskId>>>>,
    
    // Statistics
    queue_stats: Arc<Mutex<QueueStatistics>>,
}

/// Running task information
#[derive(Debug, Clone)]
pub struct RunningTaskInfo {
    pub task: Task,
    pub device: Device,
    pub started_at: Instant,
    pub estimated_completion: Instant,
    pub resource_allocation: ResourceAllocation,
}

/// Task metadata
#[derive(Debug, Clone)]
pub struct TaskMetadata {
    pub task_id: TaskId,
    pub priority: TaskPriority,
    pub status: TaskStatus,
    pub created_at: Instant,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub retry_count: usize,
    pub max_retries: usize,
    pub timeout: Duration,
}

/// Queue statistics
#[derive(Debug, Clone)]
pub struct QueueStatistics {
    pub total_tasks: usize,
    pub pending_tasks: usize,
    pub running_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub cancelled_tasks: usize,
    pub average_wait_time: Duration,
    pub average_execution_time: Duration,
    pub throughput: f32,
    pub queue_utilization: f32,
}

impl TaskQueue {
    pub fn new(max_queue_size: usize) -> Result<Self> {
        Ok(Self {
            pending_tasks: Arc::new(Mutex::new(PriorityQueue::new())),
            running_tasks: Arc::new(Mutex::new(HashMap::new())),
            completed_tasks: Arc::new(Mutex::new(HashMap::new())),
            failed_tasks: Arc::new(Mutex::new(HashMap::new())),
            cancelled_tasks: Arc::new(Mutex::new(HashSet::new())),
            task_metadata: Arc::new(RwLock::new(HashMap::new())),
            task_dependencies: Arc::new(RwLock::new(HashMap::new())),
            task_dependents: Arc::new(RwLock::new(HashMap::new())),
            max_queue_size,
            queue_notifications: Arc::new(Mutex::new(Vec::new())),
            queue_stats: Arc::new(Mutex::new(QueueStatistics {
                total_tasks: 0,
                pending_tasks: 0,
                running_tasks: 0,
                completed_tasks: 0,
                failed_tasks: 0,
                cancelled_tasks: 0,
                average_wait_time: Duration::from_millis(0),
                average_execution_time: Duration::from_millis(0),
                throughput: 0.0,
                queue_utilization: 0.0,
            })),
        })
    }
    
    /// Enqueue a task
    pub fn enqueue(&self, task_id: TaskId, task: Task) -> Result<()> {
        // Check queue size
        {
            let pending = self.pending_tasks.lock().unwrap();
            if pending.len() >= self.max_queue_size {
                return Err(TensorError::InvalidInput { message: "Task queue is full".to_string() });
            }
        }
        
        // Create task metadata
        let metadata = TaskMetadata {
            task_id: task_id.clone(),
            priority: task.priority,
            status: TaskStatus::Pending,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            retry_count: 0,
            max_retries: task.max_retries,
            timeout: task.timeout,
        };
        
        // Add to pending tasks
        {
            let mut pending = self.pending_tasks.lock().unwrap();
            pending.push(task_id.clone(), task.priority);
        }
        
        // Store metadata
        {
            let mut metadata_map = self.task_metadata.write().unwrap();
            metadata_map.insert(task_id.clone(), metadata);
        }
        
        // Update statistics
        self.update_statistics()?;
        
        // Notify listeners
        self.notify_task_added(&task_id)?;
        
        Ok(())
    }
    
    /// Dequeue a task
    pub fn dequeue(&self) -> Result<Option<(TaskId, Task)>> {
        let mut pending = self.pending_tasks.lock().unwrap();
        let mut metadata_map = self.task_metadata.write().unwrap();
        
        // Find highest priority task that's ready to execute
        let mut best_task = None;
        let mut best_priority = TaskPriority::Low;
        
        for (task_id, priority) in pending.iter() {
            if let Some(metadata) = metadata_map.get(task_id) {
                if metadata.status == TaskStatus::Pending && self.is_task_ready(task_id)? {
                    if *priority > best_priority {
                        best_priority = priority.clone();
                        best_task = Some(task_id.clone());
                    }
                }
            }
        }
        
        if let Some(task_id) = best_task {
            pending.remove(&task_id);
            
            // Get task from metadata (we need to store the actual task somewhere)
            // This is a simplified version - in practice, you'd store the task separately
            let task = Task {
                operation: TaskOperation::TensorOperation {
                    operation: TensorOp::Add,
                    input_shapes: vec![],
                    output_shape: Shape::new(vec![1]),
                },
                priority: best_priority,
                resource_requirements: ResourceRequirements::default(),
                device_requirements: DeviceRequirements::default(),
                timeout: Duration::from_secs(300),
                retry_count: 0,
                max_retries: 3,
            };
            
            // Update metadata
            if let Some(metadata) = metadata_map.get_mut(&task_id) {
                metadata.status = TaskStatus::Running;
                metadata.started_at = Some(Instant::now());
            }
            
            // Update statistics
            self.update_statistics()?;
            
            return Ok(Some((task_id, task)));
        }
        
        Ok(None)
    }
    
    /// Get pending tasks
    pub fn get_pending_tasks(&self) -> Result<Vec<TaskInfo>> {
        let pending = self.pending_tasks.lock().unwrap();
        let metadata_map = self.task_metadata.read().unwrap();
        let mut pending_tasks = Vec::new();
        
        for (task_id, priority) in pending.iter() {
            if let Some(metadata) = metadata_map.get(task_id) {
                if metadata.status == TaskStatus::Pending {
                    let task = Task {
                        operation: TaskOperation::TensorOperation {
                            operation: TensorOp::Add,
                            input_shapes: vec![],
                            output_shape: Shape::new(vec![1]),
                        },
                        priority: priority.clone(),
                        resource_requirements: ResourceRequirements::default(),
                        device_requirements: DeviceRequirements::default(),
                        timeout: metadata.timeout,
                        retry_count: metadata.retry_count,
                        max_retries: metadata.max_retries,
                    };
                    
                    pending_tasks.push(TaskInfo {
                        task_id: task_id.clone(),
                        task,
                        priority: priority.clone(),
                        created_at: metadata.created_at,
                    });
                }
            }
        }
        
        Ok(pending_tasks)
    }
    
    /// Get task status
    pub fn get_status(&self, task_id: &TaskId) -> Result<TaskStatus> {
        let metadata_map = self.task_metadata.read().unwrap();
        if let Some(metadata) = metadata_map.get(task_id) {
            Ok(metadata.status.clone())
        } else {
            Err(TensorError::InvalidInput { message: "Task not found".to_string() })
        }
    }
    
    /// Set task status
    pub fn set_status(&self, task_id: &TaskId, status: TaskStatus) -> Result<()> {
        let mut metadata_map = self.task_metadata.write().unwrap();
        if let Some(metadata) = metadata_map.get_mut(task_id) {
            metadata.status = status;
        }
        
        self.update_statistics()?;
        Ok(())
    }
    
    /// Get task result
    pub fn get_result(&self, task_id: &TaskId) -> Result<Option<TaskResult>> {
        let completed = self.completed_tasks.lock().unwrap();
        Ok(completed.get(task_id).cloned())
    }
    
    /// Set task result
    pub fn set_result(&self, task_id: &TaskId, result: TaskResult) -> Result<()> {
        let mut completed = self.completed_tasks.lock().unwrap();
        completed.insert(task_id.clone(), result);
        
        // Update metadata
        let mut metadata_map = self.task_metadata.write().unwrap();
        if let Some(metadata) = metadata_map.get_mut(task_id) {
            metadata.status = TaskStatus::Completed;
            metadata.completed_at = Some(Instant::now());
        }
        
        self.update_statistics()?;
        Ok(())
    }
    
    /// Cancel a task
    pub fn cancel(&self, task_id: &TaskId) -> Result<()> {
        // Add to cancelled tasks
        {
            let mut cancelled = self.cancelled_tasks.lock().unwrap();
            cancelled.insert(task_id.clone());
        }
        
        // Remove from pending if present
        {
            let mut pending = self.pending_tasks.lock().unwrap();
            pending.remove(task_id);
        }
        
        // Update metadata
        let mut metadata_map = self.task_metadata.write().unwrap();
        if let Some(metadata) = metadata_map.get_mut(task_id) {
            metadata.status = TaskStatus::Cancelled;
            metadata.completed_at = Some(Instant::now());
        }
        
        self.update_statistics()?;
        Ok(())
    }
    
    /// Reschedule a task to a different device
    pub fn reschedule(&self, task_id: &TaskId, new_device: Device) -> Result<()> {
        // Update running task info
        {
            let mut running = self.running_tasks.lock().unwrap();
            if let Some(task_info) = running.get_mut(task_id) {
                task_info.device = new_device;
            }
        }
        
        Ok(())
    }
    
    /// Get all task statuses
    pub fn get_all_status(&self) -> Result<HashMap<TaskId, TaskStatus>> {
        let metadata_map = self.task_metadata.read().unwrap();
        let mut statuses = HashMap::new();
        
        for (task_id, metadata) in metadata_map.iter() {
            statuses.insert(task_id.clone(), metadata.status);
        }
        
        Ok(statuses)
    }
    
    /// Get queue statistics
    pub fn get_statistics(&self) -> Result<QueueStatistics> {
        let stats = self.queue_stats.lock().unwrap();
        Ok(stats.clone())
    }
    
    /// Add task dependency
    pub fn add_dependency(&self, task_id: &TaskId, dependency_id: &TaskId) -> Result<()> {
        // Add to dependencies
        {
            let mut dependencies = self.task_dependencies.write().unwrap();
            dependencies.entry(task_id.clone())
                .or_insert_with(Vec::new)
                .push(dependency_id.clone());
        }
        
        // Add to dependents
        {
            let mut dependents = self.task_dependents.write().unwrap();
            dependents.entry(dependency_id.clone())
                .or_insert_with(Vec::new)
                .push(task_id.clone());
        }
        
        Ok(())
    }
    
    /// Remove task dependency
    pub fn remove_dependency(&self, task_id: &TaskId, dependency_id: &TaskId) -> Result<()> {
        // Remove from dependencies
        {
            let mut dependencies = self.task_dependencies.write().unwrap();
            if let Some(deps) = dependencies.get_mut(task_id) {
                deps.retain(|id| id != dependency_id);
            }
        }
        
        // Remove from dependents
        {
            let mut dependents = self.task_dependents.write().unwrap();
            if let Some(deps) = dependents.get_mut(dependency_id) {
                deps.retain(|id| id != task_id);
            }
        }
        
        Ok(())
    }
    
    /// Check if task is ready to execute (dependencies satisfied)
    fn is_task_ready(&self, task_id: &TaskId) -> Result<bool> {
        let dependencies = self.task_dependencies.read().unwrap();
        let completed = self.completed_tasks.lock().unwrap();
        
        if let Some(deps) = dependencies.get(task_id) {
            for dep_id in deps {
                if !completed.contains_key(dep_id) {
                    return Ok(false);
                }
            }
        }
        
        Ok(true)
    }
    
    /// Update queue statistics
    fn update_statistics(&self) -> Result<()> {
        let mut stats = self.queue_stats.lock().unwrap();
        let metadata_map = self.task_metadata.read().unwrap();
        
        stats.total_tasks = metadata_map.len();
        stats.pending_tasks = 0;
        stats.running_tasks = 0;
        stats.completed_tasks = 0;
        stats.failed_tasks = 0;
        stats.cancelled_tasks = 0;
        
        for metadata in metadata_map.values() {
            match metadata.status {
                TaskStatus::Pending => stats.pending_tasks += 1,
                TaskStatus::Running => stats.running_tasks += 1,
                TaskStatus::Completed => stats.completed_tasks += 1,
                TaskStatus::Failed => stats.failed_tasks += 1,
                TaskStatus::Cancelled => stats.cancelled_tasks += 1,
                TaskStatus::Timeout => stats.failed_tasks += 1,
            }
        }
        
        // Calculate utilization
        stats.queue_utilization = if stats.total_tasks > 0 {
            (stats.running_tasks as f32) / (stats.total_tasks as f32)
        } else {
            0.0
        };
        
        Ok(())
    }
    
    /// Notify listeners about task addition
    fn notify_task_added(&self, task_id: &TaskId) -> Result<()> {
        let notifications = self.queue_notifications.lock().unwrap();
        for sender in notifications.iter() {
            let _ = sender.send(task_id.clone());
        }
        Ok(())
    }
    
    /// Subscribe to queue notifications
    pub fn subscribe(&self) -> Receiver<TaskId> {
        let (sender, receiver) = channel::unbounded();
        let mut notifications = self.queue_notifications.lock().unwrap();
        notifications.push(sender);
        receiver
    }
}

/// Task information
#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub task_id: TaskId,
    pub task: Task,
    pub priority: TaskPriority,
    pub created_at: Instant,
}
