//! 🌀 HelixML Data Loader
//! 
//! Efficient data loading and batching for training.

use tensor_core::tensor::Tensor;
use std::collections::VecDeque;
use anyhow::Result as AnyResult;

/// Data loader
pub struct DataLoader<T: Tensor> {
    /// Number of workers
    num_workers: usize,
    /// Pin memory
    pin_memory: bool,
    /// Data queue
    data_queue: VecDeque<T>,
    /// Batch queue
    batch_queue: VecDeque<Vec<T>>,
}

impl<T: Tensor> DataLoader<T> {
    /// Create new data loader
    pub fn new(num_workers: usize, pin_memory: bool) -> AnyResult<Self> {
        Ok(Self {
            num_workers,
            pin_memory,
            data_queue: VecDeque::new(),
            batch_queue: VecDeque::new(),
        })
    }
    
    /// Create batches from data
    pub fn create_batches(&self, data: &[T], batch_size: usize) -> AnyResult<Vec<Vec<T>>> {
        let mut batches = Vec::new();
        
        for chunk in data.chunks(batch_size) {
            batches.push(chunk.to_vec());
        }
        
        Ok(batches)
    }
    
    /// Load data
    pub fn load_data(&mut self, data: &[T]) -> AnyResult<()> {
        self.data_queue.clear();
        for tensor in data {
            self.data_queue.push_back(tensor.clone());
        }
        Ok(())
    }
    
    /// Get next batch
    pub fn get_next_batch(&mut self, batch_size: usize) -> Option<Vec<T>> {
        if self.data_queue.len() < batch_size {
            return None;
        }
        
        let mut batch = Vec::new();
        for _ in 0..batch_size {
            if let Some(tensor) = self.data_queue.pop_front() {
                batch.push(tensor);
            }
        }
        
        Some(batch)
    }
    
    /// Check if data is available
    pub fn has_data(&self) -> bool {
        !self.data_queue.is_empty()
    }
    
    /// Get data size
    pub fn get_data_size(&self) -> usize {
        self.data_queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_loader_creation() {
        let loader = DataLoader::new(4, true).unwrap();
        assert_eq!(loader.num_workers, 4);
        assert!(loader.pin_memory);
    }
    
    #[test]
    fn test_create_batches() {
        let loader = DataLoader::new(4, true).unwrap();
        let data = vec![Tensor::from(1.0), Tensor::from(2.0), Tensor::from(3.0), Tensor::from(4.0)];
        let batches = loader.create_batches(&data, 2).unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 2);
    }
}
