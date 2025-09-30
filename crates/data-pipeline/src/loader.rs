//! Data loader for efficient data loading

use tensor_core::{Tensor, Result, TensorError};
use crate::dataset::{Dataset, DatasetItem};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use std::collections::VecDeque;
use rand::seq::SliceRandom;

/// Configuration for data loader
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub batch_size: usize,
    pub num_workers: usize,
    pub prefetch_factor: usize,
    pub shuffle: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_workers: 4,
            prefetch_factor: 2,
            shuffle: true,
        }
    }
}

/// Data loader for efficient data loading
pub struct DataLoader<T: Tensor> {
    dataset: Arc<dyn Dataset<T>>,
    config: DataLoaderConfig,
    batch_queue: Arc<tokio::sync::RwLock<VecDeque<Vec<DatasetItem<T>>>>>,
    workers: Vec<JoinHandle<()>>,
    sender: Option<mpsc::UnboundedSender<()>>,
    receiver: Option<mpsc::UnboundedReceiver<()>>,
}

impl<T: Tensor + 'static> DataLoader<T> {
    /// Create a new data loader
    pub fn new(dataset: Arc<dyn Dataset<T>>, config: DataLoaderConfig) -> Result<Self> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let batch_queue = Arc::new(tokio::sync::RwLock::new(VecDeque::new()));
        
        Ok(Self {
            dataset,
            config,
            batch_queue,
            workers: Vec::new(),
            sender: Some(sender),
            receiver: Some(receiver),
        })
    }
    
    /// Start the data loader workers
    pub async fn start(&mut self) -> Result<()> {
        let dataset = self.dataset.clone();
        let config = self.config.clone();
        let batch_queue = self.batch_queue.clone();
        // Start worker tasks
        for _worker_id in 0..self.config.num_workers {
            let dataset = dataset.clone();
            let config = config.clone();
            let batch_queue = batch_queue.clone();
            let (_, mut receiver) = mpsc::unbounded_channel::<()>();
            
            let worker = tokio::spawn(async move {
                let mut indices = (0..dataset.len()).collect::<Vec<_>>();
                let mut current_batch = Vec::new();
                
                loop {
                    // Check for shutdown signal
                    if receiver.try_recv().is_ok() {
                        break;
                    }
                    
                    // Shuffle if enabled
                    if config.shuffle {
                        use rand::seq::SliceRandom;
                        indices.shuffle(&mut rand::thread_rng());
                    }
                    
                    // Create batches
                    for &index in &indices {
                        if let Ok(item) = dataset.get(index).await {
                            current_batch.push(item);
                            
                            if current_batch.len() == config.batch_size {
                                // Add batch to queue
                                let mut queue = batch_queue.write().await;
                                queue.push_back(current_batch.clone());
                                current_batch.clear();
                            }
                        }
                    }
                    
                    // Add remaining items as final batch
                    if !current_batch.is_empty() {
                        let mut queue = batch_queue.write().await;
                        queue.push_back(current_batch.clone());
                    }
                    
                    // Small delay to prevent busy waiting
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                }
            });
            
            self.workers.push(worker);
        }
        
        Ok(())
    }
    
    /// Get the next batch
    pub async fn next_batch(&mut self) -> Result<Vec<DatasetItem<T>>> {
        let mut queue = self.batch_queue.write().await;
        
        if let Some(batch) = queue.pop_front() {
            Ok(batch)
        } else {
            // If no batch available, create one on demand
            self.create_batch_on_demand().await
        }
    }
    
    /// Create a batch on demand
    async fn create_batch_on_demand(&self) -> Result<Vec<DatasetItem<T>>> {
        let mut batch = Vec::new();
        let mut indices = (0..self.dataset.len()).collect::<Vec<_>>();
        
        if self.config.shuffle {
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());
        }
        
        for &index in indices.iter().take(self.config.batch_size) {
            if let Ok(item) = self.dataset.get(index).await {
                batch.push(item);
            }
        }
        
        Ok(batch)
    }
    
    /// Stop the data loader
    pub async fn stop(&mut self) -> Result<()> {
        // Send shutdown signal to all workers
        if let Some(sender) = self.sender.take() {
            drop(sender);
        }
        
        // Wait for all workers to finish
        for worker in self.workers.drain(..) {
            let _ = worker.await;
        }
        
        Ok(())
    }
    
    /// Get loader statistics
    pub fn get_stats(&self) -> LoaderStats {
        LoaderStats {
            batch_size: self.config.batch_size,
            num_workers: self.config.num_workers,
            prefetch_factor: self.config.prefetch_factor,
            shuffle: self.config.shuffle,
        }
    }
}

/// Data loader statistics
#[derive(Debug, Clone)]
pub struct LoaderStats {
    pub batch_size: usize,
    pub num_workers: usize,
    pub prefetch_factor: usize,
    pub shuffle: bool,
}

/// Error types for data loader
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("Dataset error: {0}")]
    Dataset(String),
    
    #[error("Worker error: {0}")]
    Worker(String),
    
    #[error("Channel error: {0}")]
    Channel(String),
    
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),
}

impl<T: Tensor> Drop for DataLoader<T> {
    fn drop(&mut self) {
        // Try to stop workers gracefully
        if let Some(sender) = self.sender.take() {
            drop(sender);
        }
    }
}
