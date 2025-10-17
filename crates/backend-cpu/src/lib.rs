// Minimal backend-cpu stub for Docker build
use anyhow::Result;
use tensor_core::{Tensor, Device, DeviceType};

pub struct CpuBackend {
    pub device: Device,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            device: Device::new(DeviceType::Cpu),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}
