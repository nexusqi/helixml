//! 🌀 HelixML CUDA Example
//! 
//! Example demonstrating CUDA backend usage

use tensor_core::*;
use tensor_core::tensor::{TensorOps, TensorRandom};
use backend_cpu::CpuTensor;
use std::env;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🌀 HelixML CUDA Backend Example");
    println!("================================");
    
    // Check if CUDA is available
    if env::var("CUDA_ROOT").is_err() && env::var("CUDA_HOME").is_err() {
        println!("⚠️  CUDA not found, falling back to CPU backend");
        run_cpu_example()?;
        return Ok(());
    }
    
    println!("🚀 CUDA backend detected!");
    
    // TODO: Implement CUDA example when CUDA is available
    println!("📝 CUDA example will be implemented when CUDA is available");
    
    Ok(())
}

fn run_cpu_example() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🖥️  Running CPU backend example...");
    
    let device = Device::cpu();
    
    // Create tensors
    let a = CpuTensor::random_uniform(
        Shape::new(vec![100, 100]), 
        -1.0, 1.0, &device
    )?;
    
    let b = CpuTensor::random_uniform(
        Shape::new(vec![100, 100]), 
        -1.0, 1.0, &device
    )?;
    
    // Perform operations
    println!("➕ Adding tensors...");
    let c = a.add(&b)?;
    println!("✅ Addition completed");
    
    println!("✖️  Multiplying tensors...");
    let d = a.mul(&b)?;
    println!("✅ Multiplication completed");
    
    println!("🔢 Matrix multiplication...");
    let e = a.matmul(&b)?;
    println!("✅ Matrix multiplication completed");
    
    println!("🎯 Results:");
    println!("  - Addition shape: {:?}", c.shape());
    println!("  - Multiplication shape: {:?}", d.shape());
    println!("  - Matrix multiplication shape: {:?}", e.shape());
    
    Ok(())
}
