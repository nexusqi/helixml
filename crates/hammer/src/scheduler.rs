//! ⚡ Device-Agnostic Scheduler

use tensor_core::Device;

/// Device assignment for operations
#[derive(Debug, Clone)]
pub struct DeviceAssignment {
    pub device: Device,
    pub priority: f32,
}

/// Hammer scheduler for device-agnostic execution
pub struct HammerScheduler {
    available_devices: Vec<Device>,
}

impl HammerScheduler {
    pub fn new() -> Self {
        Self {
            available_devices: vec![Device::cpu()],
        }
    }
    
    pub fn detect_devices(&mut self) {
        // TODO: Auto-detect all available devices
    }
    
    pub fn assign_device(&self) -> DeviceAssignment {
        DeviceAssignment {
            device: self.available_devices.first().cloned().unwrap_or(Device::cpu()),
            priority: 1.0,
        }
    }
}

impl Default for HammerScheduler {
    fn default() -> Self {
        Self::new()
    }
}

