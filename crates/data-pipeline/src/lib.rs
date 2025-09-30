//! 🌀 HelixML Data Pipeline
//!
//! Efficient data loading and preprocessing for SSM/Hyena architectures.
//! Features async data loading, preprocessing, batching, and caching.

use tensor_core::{Tensor, Shape, DType, Device, Result, TensorError};
use backend_cpu::CpuTensor;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};

pub mod dataset;
pub mod loader;
pub mod preprocessor;
pub mod batcher;
pub mod cache;

// Re-export key types
pub use dataset::{Dataset, DatasetItem, DatasetIterator};
pub use loader::{DataLoader, DataLoaderConfig, LoaderError};
pub use preprocessor::{Preprocessor, TextPreprocessor, ImagePreprocessor, AudioPreprocessor};
pub use batcher::{Batcher, BatchConfig, BatchItem};
pub use cache::{DataCache, CacheConfig, CacheStats};

/// Data pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub batch_size: usize,
    pub num_workers: usize,
    pub prefetch_factor: usize,
    pub shuffle: bool,
    pub cache_enabled: bool,
    pub cache_size: usize,
    pub device: Device,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_workers: 4,
            prefetch_factor: 2,
            shuffle: true,
            cache_enabled: false,
            cache_size: 1000,
            device: Device::cpu(),
        }
    }
}

/// Main data pipeline for training
pub struct DataPipeline<T: Tensor> {
    config: PipelineConfig,
    dataset: Arc<dyn Dataset<T>>,
    loader: DataLoader<T>,
    preprocessor: Arc<dyn Preprocessor<T>>,
    batcher: Batcher<T>,
    cache: Option<DataCache<T>>,
}

impl<T: Tensor + tensor_core::tensor::TensorRandom + 'static> DataPipeline<T> {
    /// Create a new data pipeline
    pub fn new(
        config: PipelineConfig,
        dataset: Arc<dyn Dataset<T>>,
        preprocessor: Arc<dyn Preprocessor<T>>,
    ) -> Result<Self> {
        let loader_config = DataLoaderConfig {
            batch_size: config.batch_size,
            num_workers: config.num_workers,
            prefetch_factor: config.prefetch_factor,
            shuffle: config.shuffle,
        };

        let loader = DataLoader::new(dataset.clone(), loader_config)?;
        
        let batch_config = BatchConfig {
            batch_size: config.batch_size,
            device: config.device.clone(),
        };
        
        let batcher = Batcher::<T>::new(batch_config)?;
        
        let cache = if config.cache_enabled {
            let cache_config = CacheConfig {
                max_size: config.cache_size,
                device: config.device.clone(),
            };
            Some(DataCache::new(cache_config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            dataset,
            loader,
            preprocessor,
            batcher,
            cache,
        })
    }

    /// Get the next batch of data
    pub async fn next_batch(&mut self) -> Result<BatchItem<T>> {
        // Load raw data
        let raw_batch = self.loader.next_batch().await?;
        
        // Preprocess data
        let processed_batch = self.preprocessor.process_batch(raw_batch).await?;
        
        // Check cache
        if let Some(cache) = &mut self.cache {
            if let Some(cached_batch) = cache.get(&processed_batch.id).await? {
                return Ok(cached_batch);
            }
        }
        
        // Create batch
        let batch = self.batcher.create_batch(processed_batch.items).await?;
        
        // Cache if enabled
        if let Some(cache) = &mut self.cache {
            cache.put(processed_batch.id, batch.clone()).await?;
        }
        
        Ok(batch)
    }

    /// Get pipeline statistics
    pub fn get_stats(&self) -> PipelineStats {
        PipelineStats {
            total_samples: self.dataset.len(),
            batch_size: self.config.batch_size,
            num_workers: self.config.num_workers,
            cache_stats: self.cache.as_ref().map(|c| c.get_stats()),
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub total_samples: usize,
    pub batch_size: usize,
    pub num_workers: usize,
    pub cache_stats: Option<CacheStats>,
}

/// Error types for data pipeline
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Dataset error: {0}")]
    Dataset(String),
    
    #[error("Loader error: {0}")]
    Loader(#[from] LoaderError),
    
    #[error("Preprocessing error: {0}")]
    Preprocessing(String),
    
    #[error("Batching error: {0}")]
    Batching(String),
    
    #[error("Cache error: {0}")]
    Cache(String),
    
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),
}

impl From<PipelineError> for TensorError {
    fn from(err: PipelineError) -> Self {
        match err {
            PipelineError::Tensor(e) => e,
            _ => TensorError::UnsupportedOperation {
                op: format!("Pipeline error: {}", err),
            },
        }
    }
}

/// Result type for data pipeline operations
pub type PipelineResult<T> = std::result::Result<T, PipelineError>;
