//! 🌀 HelixML Advanced Autograd Example
//! 
//! Example demonstrating advanced autograd features for training large models

use tensor_core::*;
use tensor_core::tensor::{TensorOps, TensorRandom, TensorReduce, TensorStats};
use backend_cpu::CpuTensor;
use autograd::*;
use autograd::advanced::*;
use autograd::optimizer::*;
use autograd::memory::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🌀 HelixML Advanced Autograd Example");
    println!("====================================");
    
    let device = Device::cpu();
    
    // 1. Basic gradient accumulation
    println!("\n📊 1. Gradient Accumulation Demo");
    gradient_accumulation_demo(&device)?;
    
    // 2. Gradient clipping
    println!("\n✂️  2. Gradient Clipping Demo");
    gradient_clipping_demo(&device)?;
    
    // 3. Mixed precision training
    println!("\n🎯 3. Mixed Precision Demo");
    mixed_precision_demo(&device)?;
    
    // 4. Memory optimization
    println!("\n💾 4. Memory Optimization Demo");
    memory_optimization_demo(&device)?;
    
    // 5. Advanced training loop
    println!("\n🚀 5. Advanced Training Loop Demo");
    advanced_training_demo(&device)?;
    
    println!("\n✅ All demos completed successfully!");
    Ok(())
}

fn gradient_accumulation_demo(device: &Device) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut ctx = AutogradContext::<CpuTensor>::new();
    
    // Create parameters
    let w = CpuTensor::random_uniform(Shape::new(vec![10, 1]), -0.1, 0.1, device)?;
    let b = CpuTensor::random_uniform(Shape::new(vec![1]), -0.1, 0.1, device)?;
    
    let w_id = ctx.tensor(w, true);
    let b_id = ctx.tensor(b, true);
    
    // Create gradient accumulator
    let mut accumulator = GradientAccumulator::new(4); // Accumulate over 4 steps
    
    println!("  📈 Accumulating gradients over 4 steps...");
    
    for step in 0..4 {
        // Simulate forward pass
        let x = CpuTensor::random_uniform(Shape::new(vec![10, 1]), -1.0, 1.0, device)?;
        let x_id = ctx.tensor(x, false);
        
        // Forward pass: y = x * w + b
        let xw = ctx.get_tensor(x_id).unwrap().tensor().matmul(ctx.get_tensor(w_id).unwrap().tensor())?;
        let xw_id = ctx.tensor(xw.clone(), true);
        
        let y = xw.add(ctx.get_tensor(b_id).unwrap().tensor())?;
        let y_id = ctx.tensor(y.clone(), true);
        
        // Simulate loss
        let target = CpuTensor::random_uniform(Shape::new(vec![1]), -1.0, 1.0, device)?;
        let target_id = ctx.tensor(target, false);
        
        let loss = y.sub(ctx.get_tensor(target_id).unwrap().tensor())?.pow(2.0)?;
        let loss_id = ctx.tensor(loss.clone(), true);
        
        // Simulate gradient computation
        let grad_w = CpuTensor::random_uniform(Shape::new(vec![10, 1]), -0.1, 0.1, device)?;
        let grad_b = CpuTensor::random_uniform(Shape::new(vec![1]), -0.1, 0.1, device)?;
        
        // Accumulate gradients
        accumulator.accumulate_gradient(w_id, grad_w)?;
        accumulator.accumulate_gradient(b_id, grad_b)?;
        
        println!("    Step {}: Progress {:.1}%", step + 1, accumulator.progress() * 100.0);
    }
    
    println!("  ✅ Gradient accumulation completed!");
    println!("  📊 Should apply gradients: {}", accumulator.should_apply());
    
    Ok(())
}

fn gradient_clipping_demo(device: &Device) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut ctx = AutogradContext::<CpuTensor>::new();
    
    // Create parameters with large gradients
    let w = CpuTensor::random_uniform(Shape::new(vec![5, 5]), -0.1, 0.1, device)?;
    let w_id = ctx.tensor(w, true);
    
    // Simulate large gradients
    let large_grad = CpuTensor::random_uniform(Shape::new(vec![5, 5]), -10.0, 10.0, device)?;
    ctx.get_tensor_mut(w_id).unwrap().set_grad(large_grad);
    
    // Create gradient clipper
    let clipper = GradientClipper::new(1.0, 2.0); // Max L2 norm of 1.0
    
    println!("  📏 Clipping gradients with max norm 1.0...");
    
    // Apply gradient clipping
    let grad_norm = clipper.clip_gradients(&mut ctx)?;
    
    println!("  📊 Gradient norm after clipping: {:.4}", grad_norm);
    println!("  ✅ Gradient clipping completed!");
    
    Ok(())
}

fn mixed_precision_demo(device: &Device) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut ctx = AutogradContext::<CpuTensor>::new();
    
    // Create mixed precision trainer
    let mut mp_trainer = MixedPrecisionTrainer::new(
        65536.0,  // initial loss scale
        16777216.0, // max loss scale
        1.0,      // min loss scale
        2.0,      // scale factor
        2000,     // scale window
        2000,     // max consecutive skips
    );
    
    // Create loss tensor
    let loss = CpuTensor::random_uniform(Shape::new(vec![1]), 0.0, 1.0, device)?;
    let loss_id = ctx.tensor(loss, true);
    
    println!("  🎯 Mixed precision training simulation...");
    
    // Simulate training steps
    for step in 0..5 {
        let has_overflow = step == 2; // Simulate overflow at step 2
        
        if has_overflow {
            println!("    Step {}: Gradient overflow detected! 🚨", step + 1);
            mp_trainer.update_loss_scale(true);
        } else {
            println!("    Step {}: Normal training step ✅", step + 1);
            mp_trainer.update_loss_scale(false);
        }
        
        println!("    Current loss scale: {:.0}", mp_trainer.current_loss_scale());
    }
    
    println!("  ✅ Mixed precision training completed!");
    
    Ok(())
}

fn memory_optimization_demo(device: &Device) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create memory pool
    let mut memory_pool = TensorMemoryPool::<CpuTensor>::new(100);
    
    println!("  💾 Memory pool optimization demo...");
    
    // Allocate tensors from pool
    let shapes = vec![
        Shape::new(vec![100, 100]),
        Shape::new(vec![50, 50]),
        Shape::new(vec![100, 100]), // Same as first
    ];
    
    for (i, shape) in shapes.iter().enumerate() {
        let tensor = memory_pool.get_tensor(shape.clone(), DType::F32, device)?;
        println!("    Allocated tensor {} with shape {:?}", i + 1, shape);
        
        // Return tensor to pool
        memory_pool.return_tensor(tensor);
    }
    
    let stats = memory_pool.stats();
    println!("  📊 Memory pool stats:");
    println!("    Total allocated: {}", stats.total_allocated);
    println!("    Peak memory: {}", stats.peak_memory);
    println!("    Pool size: {}", stats.pool_size);
    
    // Create memory monitor
    let mut monitor = MemoryMonitor::new();
    
    // Simulate memory tracking
    for i in 0..5 {
        let size = (i + 1) * 1000;
        monitor.track_allocation(i, size);
        println!("    Tracked allocation {}: {} bytes", i, size);
    }
    
    let (current, peak) = monitor.get_usage();
    println!("  📊 Memory usage: current={}, peak={}", current, peak);
    
    println!("  ✅ Memory optimization completed!");
    
    Ok(())
}

fn advanced_training_demo(device: &Device) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create advanced trainer
    let lr_scheduler = LRScheduler::WarmupCosine {
        initial_lr: 0.001,
        max_lr: 0.01,
        warmup_steps: 100,
        max_steps: 1000,
    };
    
    let mut trainer = AdvancedTrainer::new(
        0.001,  // learning rate
        0.01,   // weight decay
        0.9,    // momentum
        0.9,    // beta1
        0.999,  // beta2
        1e-8,   // epsilon
        lr_scheduler,
        Some(1.0), // gradient clipping
        true,   // mixed precision
        Some(100), // checkpoint every 100 steps
    );
    
    println!("  🚀 Advanced training loop simulation...");
    
    // Add parameters
    let w = CpuTensor::random_uniform(Shape::new(vec![10, 1]), -0.1, 0.1, device)?;
    let b = CpuTensor::random_uniform(Shape::new(vec![1]), -0.1, 0.1, device)?;
    
    let w_id = trainer.add_parameter(w.clone(), true);
    let b_id = trainer.add_parameter(b.clone(), true);
    
    println!("  📊 Training parameters:");
    println!("    Weight shape: {:?}", w.shape());
    println!("    Bias shape: {:?}", b.shape());
    
    // Simulate training steps
    for step in 0..5 {
        let result = trainer.training_step(
            |ctx| {
                // Forward pass
                let x = CpuTensor::random_uniform(Shape::new(vec![10, 1]), -1.0, 1.0, device)?;
                let x_id = ctx.tensor(x.clone(), false);
                
                let w = ctx.get_tensor(w_id).unwrap().tensor();
                let b = ctx.get_tensor(b_id).unwrap().tensor();
                
                let y = x.matmul(w)?.add(b)?;
                let y_id = ctx.tensor(y.clone(), true);
                
                let target = CpuTensor::random_uniform(Shape::new(vec![1]), -1.0, 1.0, device)?;
                let target_id = ctx.tensor(target, false);
                
                let loss = y.sub(ctx.get_tensor(target_id).unwrap().tensor())?.pow(2.0)?;
                let loss_id = ctx.tensor(loss, true);
                
                Ok(loss_id)
            },
            |ctx, loss_id| {
                // Backward pass (simplified)
                println!("    Step {}: Backward pass for loss {}", step + 1, loss_id);
                Ok(())
            },
        )?;
        
        println!("    Step {}: LR={:.6}, Step={}", 
                 step + 1, result.learning_rate, result.step_count);
    }
    
    let stats = trainer.get_stats();
    println!("  📊 Final training stats:");
    println!("    Total steps: {}", stats.step_count);
    println!("    Total parameters: {}", stats.total_params);
    println!("    Parameters with gradients: {}", stats.params_with_grad);
    
    println!("  ✅ Advanced training completed!");
    
    Ok(())
}
