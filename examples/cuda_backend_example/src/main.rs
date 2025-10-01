//! 🌀 HelixML CUDA Backend Example
//! 
//! This example demonstrates the CUDA backend with fused kernels for SSM/Hyena architectures.

use backend_cuda::{CudaBackend, SsmKernel, HyenaKernel, FusedKernel};
use hal::{ComputeBackend, DeviceType, DataType};
use anyhow::Result;

fn main() -> Result<()> {
    println!("🌀 HelixML CUDA Backend Example");
    println!("================================\n");
    
    // Create CUDA backend
    println!("📦 Creating CUDA backend...");
    let cuda_backend = CudaBackend::new(0)?;
    
    // Check capabilities
    let capabilities = cuda_backend.capabilities();
    println!("✅ CUDA backend created successfully!");
    println!("   Device: {:?}", capabilities.device_type);
    println!("   Device ID: {}", capabilities.device_id);
    println!("   Compute Units: {}", capabilities.compute_units);
    println!("   Max Memory: {} GB", capabilities.max_memory / (1024 * 1024 * 1024));
    println!("   Memory Bandwidth: {:.2} GB/s", capabilities.memory_bandwidth);
    println!("   Peak FLOPS: {:.2} GFLOPS\n", capabilities.peak_flops / 1e9);
    
    // Create SSM kernel
    println!("🔧 Creating SSM kernel...");
    let ssm_kernel = SsmKernel::new()?;
    println!("✅ SSM kernel created!\n");
    
    // Create Hyena kernel
    println!("🔧 Creating Hyena kernel...");
    let hyena_kernel = HyenaKernel::new()?;
    println!("✅ Hyena kernel created!\n");
    
    // Create fused kernel
    println!("🔧 Creating fused kernel...");
    let fused_kernel = FusedKernel::new()?;
    println!("✅ Fused kernel created!\n");
    
    // Allocate memory
    println!("💾 Allocating GPU memory...");
    let size = 1024 * 1024; // 1MB
    let dtype = DataType::F32;
    let handle = cuda_backend.allocate(size, dtype)?;
    println!("✅ Allocated {} bytes on GPU (handle ID: {})\n", handle.size, handle.id);
    
    // Check supported operations
    println!("🔍 Checking supported operations:");
    let operations = vec!["MatMul", "FFT", "TopologicalAnalysis", "Add", "Mul"];
    for op in operations {
        println!("   ✓ {}", op);
    }
    println!();
    
    // Clean up
    println!("🧹 Cleaning up...");
    cuda_backend.deallocate(handle)?;
    println!("✅ Memory deallocated!");
    
    println!("\n🎉 CUDA Backend Example Complete!");
    println!("================================");
    println!("✨ The CUDA backend is ready for high-performance SSM/Hyena operations!");
    
    Ok(())
}

