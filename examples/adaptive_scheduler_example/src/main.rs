//! 🎯 Adaptive Scheduler Example
//! 
//! Comprehensive example demonstrating the adaptive scheduler for
//! multi-device orchestration in HelixML

use adaptive_scheduler::*;
use tensor_core::{Tensor, Shape, DType, Device, Result};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision, TensorStats, TensorReduce};
use anyhow::Context;
use tracing::{info, error, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🎯 Starting Adaptive Scheduler Example");
    
    // Initialize scheduler configuration
    let config = SchedulerConfig::default();
    let mut scheduler = AdaptiveScheduler::new(config)?;
    info!("✅ Adaptive scheduler initialized");
    
    // Start the scheduler
    scheduler.start()?;
    info!("🚀 Scheduler started");
    
    // Run various examples
    run_basic_scheduling_example(&scheduler).await?;
    run_load_balancing_example(&scheduler).await?;
    run_optimization_example(&scheduler).await?;
    run_policy_example(&scheduler).await?;
    run_metrics_example(&scheduler).await?;
    run_adaptive_example(&scheduler).await?;
    
    // Stop the scheduler
    scheduler.stop()?;
    info!("🛑 Scheduler stopped");
    
    info!("🎉 Adaptive Scheduler Example completed successfully!");
    Ok(())
}

async fn run_basic_scheduling_example<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorStats + TensorReduce>(
    scheduler: &AdaptiveScheduler<T>
) -> Result<()> {
    info!("📋 Running basic scheduling example...");
    
    // Create sample tasks
    let tasks = create_sample_tasks()?;
    info!("✅ Created {} sample tasks", tasks.len());
    
    // Submit tasks to scheduler
    let mut task_ids = Vec::new();
    for task in tasks {
        let task_id = scheduler.submit_task(task)?;
        task_ids.push(task_id);
        info!("📤 Submitted task: {}", task_id.id);
    }
    
    // Wait for tasks to complete
    for task_id in &task_ids {
        let mut attempts = 0;
        loop {
            let status = scheduler.get_task_status(task_id)?;
            match status {
                TaskStatus::Completed => {
                    info!("✅ Task {} completed", task_id.id);
                    break;
                }
                TaskStatus::Failed => {
                    error!("❌ Task {} failed", task_id.id);
                    break;
                }
                TaskStatus::Running => {
                    info!("🔄 Task {} is running", task_id.id);
                }
                TaskStatus::Pending => {
                    info!("⏳ Task {} is pending", task_id.id);
                }
                _ => {}
            }
            
            attempts += 1;
            if attempts > 100 {
                warn!("⏰ Task {} timed out", task_id.id);
                break;
            }
            
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }
    
    info!("📊 Basic scheduling example completed");
    Ok(())
}

async fn run_load_balancing_example<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorStats + TensorReduce>(
    scheduler: &AdaptiveScheduler<T>
) -> Result<()> {
    info!("⚖️ Running load balancing example...");
    
    // Get device status
    let device_status = scheduler.get_device_status()?;
    info!("📱 Available devices: {}", device_status.len());
    
    for (device, status) in &device_status {
        info!("🖥️ Device {:?}: load={:.2}, memory={:.2}, compute={:.2}", 
              device, status.current_load, status.memory_usage, status.compute_usage);
    }
    
    // Create tasks with different resource requirements
    let heavy_tasks = create_heavy_tasks()?;
    let light_tasks = create_light_tasks()?;
    
    info!("📦 Created {} heavy tasks and {} light tasks", heavy_tasks.len(), light_tasks.len());
    
    // Submit heavy tasks
    for task in heavy_tasks {
        let task_id = scheduler.submit_task(task)?;
        info!("📤 Submitted heavy task: {}", task_id.id);
    }
    
    // Submit light tasks
    for task in light_tasks {
        let task_id = scheduler.submit_task(task)?;
        info!("📤 Submitted light task: {}", task_id.id);
    }
    
    // Monitor load distribution
    for i in 0..10 {
        let device_status = scheduler.get_device_status()?;
        let total_load: f32 = device_status.values().map(|s| s.current_load).sum();
        let average_load = total_load / device_status.len() as f32;
        
        info!("📊 Load distribution (iteration {}): average={:.2}", i, average_load);
        
        for (device, status) in &device_status {
            if status.current_load > average_load * 1.2 {
                warn!("⚠️ Device {:?} is overloaded: {:.2}", device, status.current_load);
            }
        }
        
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }
    
    info!("📊 Load balancing example completed");
    Ok(())
}

async fn run_optimization_example<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorStats + TensorReduce>(
    scheduler: &AdaptiveScheduler<T>
) -> Result<()> {
    info!("🚀 Running optimization example...");
    
    // Get scheduler metrics
    let metrics = scheduler.get_metrics()?;
    info!("📈 Current metrics: throughput={:.2}, load_factor={:.2}", 
          metrics.throughput, metrics.load_factor);
    
    // Create tasks with dependencies
    let tasks_with_deps = create_tasks_with_dependencies()?;
    info!("🔗 Created {} tasks with dependencies", tasks_with_deps.len());
    
    // Submit tasks
    for task in tasks_with_deps {
        let task_id = scheduler.submit_task(task)?;
        info!("📤 Submitted task with dependencies: {}", task_id.id);
    }
    
    // Monitor optimization
    for i in 0..5 {
        let metrics = scheduler.get_metrics()?;
        info!("📊 Optimization iteration {}: throughput={:.2}, utilization={:.2}", 
              i, metrics.throughput, metrics.resource_utilization.memory_usage);
        
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }
    
    info!("📊 Optimization example completed");
    Ok(())
}

async fn run_policy_example<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorStats + TensorReduce>(
    scheduler: &AdaptiveScheduler<T>
) -> Result<()> {
    info!("📋 Running policy example...");
    
    // Create tasks with different priorities
    let critical_tasks = create_critical_tasks()?;
    let normal_tasks = create_normal_tasks()?;
    let low_priority_tasks = create_low_priority_tasks()?;
    
    info!("🎯 Created {} critical, {} normal, {} low-priority tasks", 
          critical_tasks.len(), normal_tasks.len(), low_priority_tasks.len());
    
    // Submit tasks in order of priority
    for task in critical_tasks {
        let task_id = scheduler.submit_task(task)?;
        info!("🚨 Submitted critical task: {}", task_id.id);
    }
    
    for task in normal_tasks {
        let task_id = scheduler.submit_task(task)?;
        info!("📝 Submitted normal task: {}", task_id.id);
    }
    
    for task in low_priority_tasks {
        let task_id = scheduler.submit_task(task)?;
        info!("📄 Submitted low-priority task: {}", task_id.id);
    }
    
    // Monitor task execution order
    let mut completed_critical = 0;
    let mut completed_normal = 0;
    let mut completed_low = 0;
    
    for _ in 0..50 {
        let metrics = scheduler.get_metrics()?;
        info!("📊 Task completion: critical={}, normal={}, low={}", 
              completed_critical, completed_normal, completed_low);
        
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }
    
    info!("📊 Policy example completed");
    Ok(())
}

async fn run_metrics_example<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorStats + TensorReduce>(
    scheduler: &AdaptiveScheduler<T>
) -> Result<()> {
    info!("📊 Running metrics example...");
    
    // Get comprehensive metrics
    let metrics = scheduler.get_metrics()?;
    info!("📈 Scheduler metrics:");
    info!("   Total tasks: {}", metrics.total_tasks);
    info!("   Completed tasks: {}", metrics.completed_tasks);
    info!("   Failed tasks: {}", metrics.failed_tasks);
    info!("   Running tasks: {}", metrics.running_tasks);
    info!("   Pending tasks: {}", metrics.pending_tasks);
    info!("   Average execution time: {:?}", metrics.average_execution_time);
    info!("   Throughput: {:.2}", metrics.throughput);
    info!("   Load factor: {:.2}", metrics.load_factor);
    
    // Get device utilization
    for (device, utilization) in &metrics.device_utilization {
        info!("🖥️ Device {:?} utilization: {:.2}%", device, utilization * 100.0);
    }
    
    // Get resource utilization
    let resource_util = &metrics.resource_utilization;
    info!("💾 Resource utilization:");
    info!("   Memory: {:.2}%", resource_util.memory_usage * 100.0);
    info!("   Compute: {:.2}%", resource_util.compute_usage * 100.0);
    info!("   Bandwidth: {:.2}%", resource_util.bandwidth_usage * 100.0);
    info!("   Storage: {:.2}%", resource_util.storage_usage * 100.0);
    
    info!("📊 Metrics example completed");
    Ok(())
}

async fn run_adaptive_example<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorStats + TensorReduce>(
    scheduler: &AdaptiveScheduler<T>
) -> Result<()> {
    info!("🧠 Running adaptive example...");
    
    // Create a mix of different task types
    let mixed_tasks = create_mixed_tasks()?;
    info!("🎲 Created {} mixed tasks", mixed_tasks.len());
    
    // Submit tasks
    for task in mixed_tasks {
        let task_id = scheduler.submit_task(task)?;
        info!("📤 Submitted mixed task: {}", task_id.id);
    }
    
    // Monitor adaptive behavior
    for i in 0..20 {
        let metrics = scheduler.get_metrics()?;
        let device_status = scheduler.get_device_status()?;
        
        info!("🧠 Adaptive iteration {}:", i);
        info!("   Throughput: {:.2}", metrics.throughput);
        info!("   Load factor: {:.2}", metrics.load_factor);
        info!("   Device count: {}", device_status.len());
        
        // Check for load balancing
        let loads: Vec<f32> = device_status.values().map(|s| s.current_load).collect();
        if !loads.is_empty() {
            let average_load = loads.iter().sum::<f32>() / loads.len() as f32;
            let max_load = loads.iter().fold(0.0, |a, &b| a.max(b));
            let load_variance = max_load - average_load;
            
            if load_variance > 0.3 {
                warn!("⚠️ Load imbalance detected: variance={:.2}", load_variance);
            } else {
                info!("✅ Load is well balanced: variance={:.2}", load_variance);
            }
        }
        
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }
    
    info!("📊 Adaptive example completed");
    Ok(())
}

/// Create sample tasks
fn create_sample_tasks() -> Result<Vec<Task<()>>> {
    let mut tasks = Vec::new();
    
    for i in 0..10 {
        let task = Task {
            operation: TaskOperation::TensorOperation {
                operation: TensorOp::Add,
                inputs: vec![],
                output_shape: Shape::new(vec![100, 100]),
            },
            priority: TaskPriority::Normal,
            resource_requirements: ResourceRequirements {
                memory: 1024 * 1024, // 1MB
                compute: 1.0,
                bandwidth: 1.0,
                storage: 0,
            },
            device_requirements: DeviceRequirements::default(),
            timeout: Duration::from_secs(30),
            retry_count: 0,
            max_retries: 3,
        };
        
        tasks.push(task);
    }
    
    Ok(tasks)
}

/// Create heavy tasks
fn create_heavy_tasks() -> Result<Vec<Task<()>>> {
    let mut tasks = Vec::new();
    
    for _ in 0..5 {
        let task = Task {
            operation: TaskOperation::TensorOperation {
                operation: TensorOp::MatrixMultiply,
                inputs: vec![],
                output_shape: Shape::new(vec![1000, 1000]),
            },
            priority: TaskPriority::High,
            resource_requirements: ResourceRequirements {
                memory: 1024 * 1024 * 1024, // 1GB
                compute: 10.0,
                bandwidth: 10.0,
                storage: 1024 * 1024, // 1MB
            },
            device_requirements: DeviceRequirements {
                device_types: vec![Device::CPU, Device::CUDA(0)],
                min_memory: 1024 * 1024 * 1024,
                min_compute_capability: 5.0,
                special_features: vec!["cuda".to_string()],
            },
            timeout: Duration::from_secs(60),
            retry_count: 0,
            max_retries: 2,
        };
        
        tasks.push(task);
    }
    
    Ok(tasks)
}

/// Create light tasks
fn create_light_tasks() -> Result<Vec<Task<()>>> {
    let mut tasks = Vec::new();
    
    for _ in 0..15 {
        let task = Task {
            operation: TaskOperation::TensorOperation {
                operation: TensorOp::Add,
                inputs: vec![],
                output_shape: Shape::new(vec![10, 10]),
            },
            priority: TaskPriority::Low,
            resource_requirements: ResourceRequirements {
                memory: 1024, // 1KB
                compute: 0.1,
                bandwidth: 0.1,
                storage: 0,
            },
            device_requirements: DeviceRequirements::default(),
            timeout: Duration::from_secs(10),
            retry_count: 0,
            max_retries: 5,
        };
        
        tasks.push(task);
    }
    
    Ok(tasks)
}

/// Create tasks with dependencies
fn create_tasks_with_dependencies() -> Result<Vec<Task<()>>> {
    let mut tasks = Vec::new();
    
    // Create a chain of dependent tasks
    for i in 0..5 {
        let task = Task {
            operation: TaskOperation::TensorOperation {
                operation: TensorOp::MatrixMultiply,
                inputs: vec![],
                output_shape: Shape::new(vec![100, 100]),
            },
            priority: TaskPriority::Normal,
            resource_requirements: ResourceRequirements {
                memory: 1024 * 1024, // 1MB
                compute: 2.0,
                bandwidth: 2.0,
                storage: 0,
            },
            device_requirements: DeviceRequirements::default(),
            timeout: Duration::from_secs(30),
            retry_count: 0,
            max_retries: 3,
        };
        
        tasks.push(task);
    }
    
    Ok(tasks)
}

/// Create critical tasks
fn create_critical_tasks() -> Result<Vec<Task<()>>> {
    let mut tasks = Vec::new();
    
    for _ in 0..3 {
        let task = Task {
            operation: TaskOperation::TensorOperation {
                operation: TensorOp::MatrixMultiply,
                inputs: vec![],
                output_shape: Shape::new(vec![500, 500]),
            },
            priority: TaskPriority::Critical,
            resource_requirements: ResourceRequirements {
                memory: 512 * 1024 * 1024, // 512MB
                compute: 5.0,
                bandwidth: 5.0,
                storage: 0,
            },
            device_requirements: DeviceRequirements {
                device_types: vec![Device::CUDA(0)],
                min_memory: 512 * 1024 * 1024,
                min_compute_capability: 7.0,
                special_features: vec!["cuda".to_string()],
            },
            timeout: Duration::from_secs(120),
            retry_count: 0,
            max_retries: 1,
        };
        
        tasks.push(task);
    }
    
    Ok(tasks)
}

/// Create normal tasks
fn create_normal_tasks() -> Result<Vec<Task<()>>> {
    let mut tasks = Vec::new();
    
    for _ in 0..7 {
        let task = Task {
            operation: TaskOperation::TensorOperation {
                operation: TensorOp::Add,
                inputs: vec![],
                output_shape: Shape::new(vec![100, 100]),
            },
            priority: TaskPriority::Normal,
            resource_requirements: ResourceRequirements {
                memory: 1024 * 1024, // 1MB
                compute: 1.0,
                bandwidth: 1.0,
                storage: 0,
            },
            device_requirements: DeviceRequirements::default(),
            timeout: Duration::from_secs(60),
            retry_count: 0,
            max_retries: 3,
        };
        
        tasks.push(task);
    }
    
    Ok(tasks)
}

/// Create low priority tasks
fn create_low_priority_tasks() -> Result<Vec<Task<()>>> {
    let mut tasks = Vec::new();
    
    for _ in 0..10 {
        let task = Task {
            operation: TaskOperation::TensorOperation {
                operation: TensorOp::Add,
                inputs: vec![],
                output_shape: Shape::new(vec![50, 50]),
            },
            priority: TaskPriority::Low,
            resource_requirements: ResourceRequirements {
                memory: 256 * 1024, // 256KB
                compute: 0.5,
                bandwidth: 0.5,
                storage: 0,
            },
            device_requirements: DeviceRequirements::default(),
            timeout: Duration::from_secs(300),
            retry_count: 0,
            max_retries: 5,
        };
        
        tasks.push(task);
    }
    
    Ok(tasks)
}

/// Create mixed tasks
fn create_mixed_tasks() -> Result<Vec<Task<()>>> {
    let mut tasks = Vec::new();
    
    // Add some of each type
    tasks.extend(create_critical_tasks()?);
    tasks.extend(create_normal_tasks()?);
    tasks.extend(create_low_priority_tasks()?);
    tasks.extend(create_heavy_tasks()?);
    tasks.extend(create_light_tasks()?);
    
    Ok(tasks)
}
