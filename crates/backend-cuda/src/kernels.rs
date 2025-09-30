//! 🌀 HelixML CUDA Kernels Module
//! 
//! Module for loading and managing CUDA kernels

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;

/// CUDA kernels manager
pub struct CudaKernels {
    device: CudaDevice,
    kernels: HashMap<String, Ptx>,
}

impl CudaKernels {
    /// Create a new CUDA kernels manager
    pub fn new(device: CudaDevice) -> Result<Self, Box<dyn std::error::Error>> {
        let mut kernels = HashMap::new();
        
        // Load kernels from PTX
        let ptx_code = include_str!("kernels.ptx");
        let ptx = Ptx::from_str(ptx_code)?;
        
        // Register kernels
        kernels.insert("elementwise_add".to_string(), ptx.clone());
        kernels.insert("elementwise_sub".to_string(), ptx.clone());
        kernels.insert("elementwise_mul".to_string(), ptx.clone());
        kernels.insert("elementwise_div".to_string(), ptx.clone());
        kernels.insert("elementwise_max".to_string(), ptx.clone());
        kernels.insert("elementwise_min".to_string(), ptx.clone());
        kernels.insert("elementwise_pow".to_string(), ptx.clone());
        kernels.insert("elementwise_sqrt".to_string(), ptx.clone());
        kernels.insert("elementwise_exp".to_string(), ptx.clone());
        kernels.insert("elementwise_log".to_string(), ptx.clone());
        kernels.insert("elementwise_sin".to_string(), ptx.clone());
        kernels.insert("elementwise_cos".to_string(), ptx.clone());
        kernels.insert("elementwise_tan".to_string(), ptx.clone());
        kernels.insert("elementwise_abs".to_string(), ptx.clone());
        kernels.insert("elementwise_sign".to_string(), ptx.clone());
        kernels.insert("elementwise_clamp".to_string(), ptx.clone());
        kernels.insert("elementwise_relu".to_string(), ptx.clone());
        kernels.insert("elementwise_gelu".to_string(), ptx.clone());
        kernels.insert("elementwise_silu".to_string(), ptx.clone());
        kernels.insert("elementwise_sigmoid".to_string(), ptx.clone());
        kernels.insert("elementwise_tanh".to_string(), ptx.clone());
        kernels.insert("elementwise_leaky_relu".to_string(), ptx.clone());
        kernels.insert("transpose_2d".to_string(), ptx.clone());
        kernels.insert("sum_reduce".to_string(), ptx.clone());
        kernels.insert("max_reduce".to_string(), ptx.clone());
        kernels.insert("min_reduce".to_string(), ptx.clone());
        
        Ok(Self { device, kernels })
    }
    
    /// Get a kernel function
    pub fn get_func(&self, name: &str) -> Option<&Ptx> {
        self.kernels.get(name)
    }
    
    /// Launch a kernel
    pub fn launch_kernel<Args>(&self, name: &str, config: LaunchConfig, args: Args) -> Result<(), Box<dyn std::error::Error>>
    where
        Args: LaunchAsync,
    {
        if let Some(kernel) = self.kernels.get(name) {
            unsafe {
                kernel.launch(config, args)?;
            }
        } else {
            return Err(format!("Kernel '{}' not found", name).into());
        }
        Ok(())
    }
}

/// CUDA kernel launch configurations
pub mod configs {
    use cudarc::driver::LaunchConfig;
    
    /// Get launch config for element-wise operations
    pub fn elementwise(n: u32) -> LaunchConfig {
        LaunchConfig::for_num_elems(n)
    }
    
    /// Get launch config for matrix operations
    pub fn matrix(rows: u32, cols: u32) -> LaunchConfig {
        LaunchConfig::for_num_elems(rows * cols)
    }
    
    /// Get launch config for reduction operations
    pub fn reduction(n: u32, block_size: u32) -> LaunchConfig {
        let grid_size = (n + block_size - 1) / block_size;
        LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: block_size * 4, // 4 bytes per float
        }
    }
}
