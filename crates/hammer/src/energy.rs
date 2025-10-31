//! ⚡ Energy Optimizer
//! 
//! Advanced energy optimization for minimal power consumption.

use tensor_core::{Tensor, Result, Shape, DType, Device};
use std::collections::HashMap;

/// Energy optimizer for minimal power consumption
pub struct EnergyOptimizer {
    /// Energy budget in joules
    energy_budget: f32,
    /// Energy consumption history
    energy_history: Vec<f32>,
    /// Operation energy profiles (operation_type -> joules per FLOP)
    energy_profiles: HashMap<String, f32>,
    /// Device energy efficiency (device_type -> efficiency factor)
    device_efficiency: HashMap<String, f32>,
}

/// Energy profile for a computation
#[derive(Debug, Clone)]
pub struct EnergyProfile {
    /// Estimated energy consumption in joules
    pub estimated_energy: f32,
    /// Energy per FLOP (joules/FLOP)
    pub energy_per_flop: f32,
    /// Total FLOPs
    pub total_flops: u64,
    /// Recommended optimizations
    pub recommendations: Vec<EnergyRecommendation>,
}

/// Energy optimization recommendations
#[derive(Debug, Clone)]
pub enum EnergyRecommendation {
    /// Use lower precision (FP32 -> FP16)
    LowerPrecision(DType),
    /// Increase batch size for better efficiency
    IncreaseBatchSize(usize),
    /// Use different device
    ChangeDevice(String),
    /// Fuse operations
    FuseOperations(Vec<String>),
    /// Use gradient checkpointing
    UseCheckpointing,
    /// Reduce sequence length
    ReduceSequenceLength(usize),
}

impl EnergyOptimizer {
    /// Create new energy optimizer with default budget
    pub fn new(budget: f32) -> Self {
        let mut energy_profiles = HashMap::new();
        // Energy per FLOP in joules (typical values)
        energy_profiles.insert("MatMul".to_string(), 1.0e-12); // 1 pJ/FLOP
        energy_profiles.insert("Add".to_string(), 0.5e-12);
        energy_profiles.insert("Mul".to_string(), 0.7e-12);
        energy_profiles.insert("Conv".to_string(), 2.0e-12);
        energy_profiles.insert("Attention".to_string(), 3.0e-12);
        energy_profiles.insert("SSM".to_string(), 1.5e-12);
        energy_profiles.insert("Mamba".to_string(), 1.2e-12);
        energy_profiles.insert("Hyena".to_string(), 1.8e-12);
        
        let mut device_efficiency = HashMap::new();
        device_efficiency.insert("CPU".to_string(), 1.0);
        device_efficiency.insert("GPU".to_string(), 0.3); // GPU is more energy efficient per FLOP
        device_efficiency.insert("TPU".to_string(), 0.1);
        device_efficiency.insert("NPU".to_string(), 0.2);
        
        Self {
            energy_budget: budget,
            energy_history: Vec::new(),
            energy_profiles,
            device_efficiency,
        }
    }
    
    /// Estimate energy consumption for an operation
    pub fn estimate_energy(
        &self,
        operation_type: &str,
        input_shapes: &[Shape],
        output_shape: &Shape,
        dtype: &DType,
        device: &Device,
    ) -> Result<f32> {
        // Estimate FLOPs for the operation
        let flops = self.estimate_flops(operation_type, input_shapes, output_shape)?;
        
        // Get base energy per FLOP
        let base_energy_per_flop = self.energy_profiles
            .get(operation_type)
            .copied()
            .unwrap_or(1.0e-12); // Default: 1 pJ/FLOP
        
        // Apply precision multiplier (FP16/BF16 use less energy than FP32)
        let precision_multiplier = match dtype {
            DType::F32 => 1.0,
            DType::F16 => 0.5,
            DType::F64 => 2.0,
            _ => 1.0,
        };
        
        // Apply device efficiency
        let device_name = device.name();
        let device_type = if device.is_cpu() {
            "CPU".to_string()
        } else if device.is_gpu() {
            "GPU".to_string()
        } else if device.is_tpu() {
            "TPU".to_string()
        } else if device.is_npu() {
            "NPU".to_string()
        } else {
            "CPU".to_string() // Default
        };
        let efficiency = self.device_efficiency
            .get(&device_type)
            .copied()
            .unwrap_or(1.0);
        
        // Calculate total energy: FLOPs * energy_per_flop * precision_mult * efficiency
        let total_energy = flops as f32 * base_energy_per_flop * precision_multiplier / efficiency;
        
        Ok(total_energy)
    }
    
    /// Estimate energy for a batch of operations
    pub fn estimate_batch_energy(
        &self,
        operations: &[(String, Vec<Shape>, Shape, DType, Device)],
    ) -> Result<f32> {
        let mut total_energy = 0.0;
        
        for (op_type, input_shapes, output_shape, dtype, device) in operations {
            let energy = self.estimate_energy(op_type, input_shapes, output_shape, dtype, device)?;
            total_energy += energy;
        }
        
        Ok(total_energy)
    }
    
    /// Get energy profile with recommendations
    pub fn analyze_energy(
        &self,
        operation_type: &str,
        input_shapes: &[Shape],
        output_shape: &Shape,
        dtype: &DType,
        device: &Device,
    ) -> Result<EnergyProfile> {
        let flops = self.estimate_flops(operation_type, input_shapes, output_shape)?;
        let estimated_energy = self.estimate_energy(operation_type, input_shapes, output_shape, dtype, device)?;
        let energy_per_flop = estimated_energy / flops as f32;
        
        let mut recommendations = Vec::new();
        
        // Recommend lower precision if using FP32
        if *dtype == DType::F32 {
            recommendations.push(EnergyRecommendation::LowerPrecision(DType::F16));
        }
        
        // Recommend device change if CPU and large operation
        if device.is_cpu() && flops > 1_000_000_000 {
            recommendations.push(EnergyRecommendation::ChangeDevice("GPU".to_string()));
        }
        
        // Recommend gradient checkpointing for large models
        if flops > 10_000_000_000 {
            recommendations.push(EnergyRecommendation::UseCheckpointing);
        }
        
        // Recommend batch size optimization
        if let Some(batch_dim) = output_shape.as_slice().first() {
            if *batch_dim < 32 {
                recommendations.push(EnergyRecommendation::IncreaseBatchSize(32));
            }
        }
        
        Ok(EnergyProfile {
            estimated_energy,
            energy_per_flop,
            total_flops: flops,
            recommendations,
        })
    }
    
    /// Estimate FLOPs for an operation
    fn estimate_flops(
        &self,
        operation_type: &str,
        input_shapes: &[Shape],
        output_shape: &Shape,
    ) -> Result<u64> {
        match operation_type {
            "MatMul" => {
                if input_shapes.len() >= 2 {
                    let a_shape = &input_shapes[0];
                    let b_shape = &input_shapes[1];
                    // MatMul: A[m,k] * B[k,n] = C[m,n]
                    // FLOPs = m * n * k
                    // For 2D matrices: shape = [m, k] and [k, n]
                    // For output shape [m, n], we can infer dimensions
                    let a_dims = a_shape.as_slice();
                    let b_dims = b_shape.as_slice();
                    let out_dims = output_shape.as_slice();
                    
                    // Handle 2D case: [m, k] * [k, n] = [m, n]
                    if a_dims.len() >= 2 && b_dims.len() >= 2 && out_dims.len() >= 2 {
                        let m = out_dims[out_dims.len() - 2];
                        let n = out_dims[out_dims.len() - 1];
                        let k = a_dims[a_dims.len() - 1];
                        Ok((m * n * k) as u64)
                    } else {
                        // Fallback: use output size multiplied by estimated inner dimension
                        let m = out_dims.get(0).copied().unwrap_or(10);
                        let n = out_dims.get(1).copied().unwrap_or(10);
                        let k = a_dims.get(1).copied().unwrap_or(10);
                        Ok((m * n * k) as u64)
                    }
                } else {
                    // Fallback: estimate from output shape
                    let out_size = output_shape.numel();
                    Ok(out_size as u64 * 10) // Rough estimate
                }
            },
            "Add" | "Mul" | "Sub" | "Div" => {
                // Element-wise operations: FLOPs = number of output elements
                Ok(output_shape.numel() as u64)
            },
            "Conv" => {
                // Convolution: approximate as MatMul equivalent
                // FLOPs ≈ output_size * kernel_size * channels_in * channels_out
                if input_shapes.len() >= 1 {
                    let input_size: usize = input_shapes[0].as_slice().iter().product();
                    Ok((input_size * 100) as u64) // Rough approximation
                } else {
                    Ok(0)
                }
            },
            "Attention" => {
                // Attention: FLOPs ≈ 2 * seq_len^2 * d_model
                if let Some(seq_len) = output_shape.as_slice().get(0) {
                    let d_model = output_shape.as_slice().get(1).copied().unwrap_or(1);
                    Ok((2 * seq_len * seq_len * d_model) as u64)
                } else {
                    Ok(0)
                }
            },
            "SSM" | "Mamba" | "Hyena" => {
                // State-space models: FLOPs ≈ seq_len * d_model * d_state
                if let Some(seq_len) = output_shape.as_slice().get(0) {
                    let d_model = output_shape.as_slice().get(1).copied().unwrap_or(1);
                    Ok((seq_len * d_model * 32) as u64) // Assume d_state=32
                } else {
                    Ok(0)
                }
            },
            _ => {
                // Default: approximate as output size
                Ok(output_shape.numel() as u64)
            }
        }
    }
    
    /// Check if operation fits within energy budget
    pub fn fits_budget(&self, estimated_energy: f32) -> bool {
        estimated_energy <= self.energy_budget
    }
    
    /// Record energy consumption
    pub fn record_energy(&mut self, energy: f32) {
        self.energy_history.push(energy);
        // Keep only last 1000 entries
        if self.energy_history.len() > 1000 {
            self.energy_history.remove(0);
        }
    }
    
    /// Get average energy consumption
    pub fn average_energy(&self) -> f32 {
        if self.energy_history.is_empty() {
            return 0.0;
        }
        self.energy_history.iter().sum::<f32>() / self.energy_history.len() as f32
    }
    
    /// Get total energy consumed
    pub fn total_energy(&self) -> f32 {
        self.energy_history.iter().sum()
    }
    
    /// Reset energy tracking
    pub fn reset(&mut self) {
        self.energy_history.clear();
    }
    
    /// Set energy budget
    pub fn set_budget(&mut self, budget: f32) {
        self.energy_budget = budget;
    }
    
    /// Get current energy budget
    pub fn budget(&self) -> f32 {
        self.energy_budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_core::Shape;
    
    #[test]
    fn test_energy_estimation() {
        let optimizer = EnergyOptimizer::new(1.0);
        
        let input_shape = Shape::new(vec![128, 768]);
        let output_shape = Shape::new(vec![128, 3072]);
        
        let energy = optimizer.estimate_energy(
            "MatMul",
            &[input_shape.clone()],
            &output_shape,
            &DType::F32,
            &Device::cpu(),
        ).unwrap();
        
        // Energy should be positive
        assert!(energy > 0.0);
    }
    
    #[test]
    fn test_energy_profile() {
        let optimizer = EnergyOptimizer::new(1.0);
        
        let input_shape = Shape::new(vec![128, 768]);
        let output_shape = Shape::new(vec![128, 3072]);
        
        let profile = optimizer.analyze_energy(
            "MatMul",
            &[input_shape],
            &output_shape,
            &DType::F32,
            &Device::cpu(),
        ).unwrap();
        
        assert!(profile.estimated_energy > 0.0);
        assert!(profile.total_flops > 0);
    }
}
