//! 🌀 HelixML Checkpoint Manager
//! 
//! Advanced checkpointing system for training state management.

use tensor_core::tensor::Tensor;
use nn::{Module, CheckpointableModule};
use crate::metrics::Metrics;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use serde::{Serialize, Deserialize};
use anyhow::Result as AnyResult;

/// Checkpoint manager
pub struct CheckpointManager<T: Tensor> {
    _phantom: std::marker::PhantomData<T>,
    /// Checkpoint directory
    checkpoint_dir: PathBuf,
    /// Checkpoint metadata
    metadata: Arc<std::sync::Mutex<HashMap<String, CheckpointMetadata>>>,
}

/// Checkpoint metadata
#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    /// Epoch
    pub epoch: usize,
    /// Step
    pub step: usize,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// File path
    pub file_path: String,
}

impl<T: Tensor> CheckpointManager<T> {
    /// Create new checkpoint manager
    pub fn new() -> AnyResult<Self> {
        let checkpoint_dir = PathBuf::from("checkpoints");
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&checkpoint_dir)?;
        
        Ok(Self {
            _phantom: std::marker::PhantomData,
            checkpoint_dir,
            metadata: Arc::new(std::sync::Mutex::new(HashMap::new())),
        })
    }
    
    /// Set checkpoint directory
    pub fn set_checkpoint_dir(&mut self, dir: &str) {
        self.checkpoint_dir = PathBuf::from(dir);
        std::fs::create_dir_all(&self.checkpoint_dir).ok();
    }
    
    /// Save checkpoint
    pub async fn save_checkpoint<M: Module<T> + CheckpointableModule<T>>(
        &self,
        epoch: usize,
        model: &M,
        metrics: &Metrics,
    ) -> AnyResult<()> {
        let checkpoint_path = self.checkpoint_dir.join(format!("checkpoint_epoch_{}.pt", epoch));
        
        // Get model state using checkpoint method
        let model_params = model.checkpoint()
            .map_err(|e| anyhow::anyhow!("Failed to get model checkpoint: {}", e))?;
        
        // Convert model parameters to serializable format
        // For now, we'll serialize the checkpoint directly
        // In a production system, we'd want to serialize tensor data more efficiently
        
        // Create checkpoint data
        let checkpoint_data = CheckpointData {
            epoch,
            step: metrics.step,
            training_loss: metrics.training_loss,
            validation_loss: metrics.validation_loss,
            learning_rate: metrics.learning_rate,
            timestamp: std::time::SystemTime::now(),
            model_params_count: model_params.len(),
            metrics: metrics.clone(),
        };
        
        // Save to file (serialize checkpoint data and model parameters)
        self.save_checkpoint_to_file(&checkpoint_path, &checkpoint_data, &model_params).await?;
        
        // Update metadata
        let mut metadata = self.metadata.lock().unwrap();
        metadata.insert(
            epoch.to_string(),
            CheckpointMetadata {
                epoch,
                step: metrics.step,
                training_loss: metrics.training_loss,
                validation_loss: metrics.validation_loss,
                learning_rate: metrics.learning_rate,
                timestamp: std::time::SystemTime::now(),
                file_path: checkpoint_path.to_string_lossy().to_string(),
            },
        );
        
        Ok(())
    }
    
    /// Load checkpoint
    pub async fn load_checkpoint<M: Module<T> + CheckpointableModule<T>>(
        &self,
        epoch: usize,
        model: &mut M,
        metrics: &mut Metrics,
    ) -> AnyResult<()> {
        let checkpoint_path = self.checkpoint_dir.join(format!("checkpoint_epoch_{}.pt", epoch));
        
        // Load from file
        let (checkpoint_data, model_params) = self.load_checkpoint_from_file(&checkpoint_path).await?;
        
        // Restore model state
        model.restore(model_params)
            .map_err(|e| anyhow::anyhow!("Failed to restore model from checkpoint: {}", e))?;
        
        // Restore metrics
        *metrics = checkpoint_data.metrics;
        
        Ok(())
    }
    
    /// Save checkpoint to file
    async fn save_checkpoint_to_file(
        &self,
        path: &Path,
        data: &CheckpointData,
        model_params: &[T],
    ) -> AnyResult<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        
        // Serialize checkpoint metadata to JSON
        let metadata_json = serde_json::to_string(data)?;
        
        // For model parameters, we'll serialize them using bincode for efficiency
        // Since tensors implement Serialize/Deserialize, we can use bincode
        let params_bytes = bincode::serialize(model_params)
            .map_err(|e| anyhow::anyhow!("Failed to serialize model parameters: {}", e))?;
        
        // Create a combined checkpoint file:
        // - First 4 bytes: length of metadata JSON (u32)
        // - Next N bytes: metadata JSON
        // - Rest: serialized model parameters
        
        let mut file = fs::File::create(path).await?;
        
        // Write metadata length
        let metadata_len = metadata_json.len() as u32;
        file.write_u32(metadata_len).await?;
        
        // Write metadata JSON
        file.write_all(metadata_json.as_bytes()).await?;
        
        // Write model parameters
        file.write_all(&params_bytes).await?;
        
        file.sync_all().await?;
        
        Ok(())
    }
    
    /// Load checkpoint from file
    async fn load_checkpoint_from_file(&self, path: &Path) -> AnyResult<(CheckpointData, Vec<T>)> {
        let mut file = fs::File::open(path).await?;
        
        // Read metadata length
        let metadata_len = file.read_u32().await? as usize;
        
        // Read metadata JSON
        let mut metadata_buf = vec![0u8; metadata_len];
        file.read_exact(&mut metadata_buf).await?;
        let metadata_json = String::from_utf8(metadata_buf)?;
        let checkpoint_data: CheckpointData = serde_json::from_str(&metadata_json)?;
        
        // Read remaining bytes (model parameters)
        let mut params_buf = Vec::new();
        file.read_to_end(&mut params_buf).await?;
        let model_params: Vec<T> = bincode::deserialize(&params_buf)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize model parameters: {}", e))?;
        
        Ok((checkpoint_data, model_params))
    }
    
    /// List available checkpoints
    pub fn list_checkpoints(&self) -> Vec<usize> {
        let metadata = self.metadata.lock().unwrap();
        metadata.keys()
            .filter_map(|k| k.parse::<usize>().ok())
            .collect()
    }
    
    /// Get checkpoint metadata
    pub fn get_checkpoint_metadata(&self, epoch: usize) -> Option<CheckpointMetadata> {
        let metadata = self.metadata.lock().unwrap();
        metadata.get(&epoch.to_string()).cloned()
    }
    
    /// Delete checkpoint
    pub async fn delete_checkpoint(&self, epoch: usize) -> AnyResult<()> {
        let checkpoint_path = self.checkpoint_dir.join(format!("checkpoint_epoch_{}.pt", epoch));
        
        // Delete file
        if checkpoint_path.exists() {
            fs::remove_file(&checkpoint_path).await?;
        }
        
        // Remove from metadata
        let mut metadata = self.metadata.lock().unwrap();
        metadata.remove(&epoch.to_string());
        
        Ok(())
    }
    
    /// Clean old checkpoints
    pub async fn clean_old_checkpoints(&self, keep_last: usize) -> AnyResult<()> {
        let mut epochs: Vec<usize> = self.list_checkpoints();
        epochs.sort();
        
        if epochs.len() > keep_last {
            let to_delete = epochs.len() - keep_last;
            for epoch in epochs.iter().take(to_delete) {
                self.delete_checkpoint(*epoch).await?;
            }
        }
        
        Ok(())
    }
}

/// Checkpoint data (metadata only - model params are stored separately)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    /// Epoch
    pub epoch: usize,
    /// Step
    pub step: usize,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Timestamp
    #[serde(with = "serde_system_time")]
    pub timestamp: std::time::SystemTime,
    /// Number of model parameters (for validation)
    pub model_params_count: usize,
    /// Metrics
    pub metrics: Metrics,
}

// Helper module for SystemTime serialization
mod serde_system_time {
    use serde::{Serialize, Serializer, Deserialize, Deserializer};
    use std::time::{SystemTime, UNIX_EPOCH};
    
    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap();
        duration.as_secs().serialize(serializer)
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + std::time::Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend_cpu::CpuTensor;
    
    #[test]
    fn test_checkpoint_manager_creation() {
        let manager: CheckpointManager<CpuTensor> = CheckpointManager::new().unwrap();
        assert_eq!(manager.checkpoint_dir, std::path::PathBuf::from("checkpoints"));
    }
    
    #[test]
    fn test_checkpoint_metadata() {
        let metadata = CheckpointMetadata {
            epoch: 10,
            step: 1000,
            training_loss: 0.5,
            validation_loss: 0.6,
            learning_rate: 0.001,
            timestamp: std::time::SystemTime::now(),
            file_path: "checkpoint_epoch_10.pt".to_string(),
        };
        
        assert_eq!(metadata.epoch, 10);
        assert_eq!(metadata.step, 1000);
    }
}
