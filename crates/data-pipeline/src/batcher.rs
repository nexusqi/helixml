//! Batching utilities for HelixML

use tensor_core::{Tensor, Shape, DType, Device, Result, TensorError};
use crate::dataset::DatasetItem;
use std::collections::HashMap;

/// Configuration for batching
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub batch_size: usize,
    pub device: Device,
}

/// Batch item containing tensors
#[derive(Debug, Clone)]
pub struct BatchItem<T: Tensor> {
    pub inputs: T,
    pub targets: Option<T>,
    pub attention_masks: Option<T>,
    pub metadata: HashMap<String, String>,
}

/// Batcher for creating batches from items
#[derive(Debug)]
pub struct Batcher<T: Tensor> {
    config: BatchConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor + tensor_core::tensor::TensorRandom> Batcher<T> {
    /// Create a new batcher
    pub fn new(config: BatchConfig) -> Result<Self> {
        Ok(Self { 
            config,
            _phantom: std::marker::PhantomData,
        })
    }
    
    /// Create a batch from items
    pub async fn create_batch(&self, items: Vec<DatasetItem<T>>) -> Result<BatchItem<T>> {
        if items.is_empty() {
            return Err(TensorError::UnsupportedOperation {
                op: "Cannot create batch from empty items".to_string(),
            });
        }
        
        // Stack input tensors
        let inputs = self.stack_tensors(items.iter().map(|item| &item.data).collect())?;
        
        // Create targets if available
        let targets = if let Some(_target_data) = items[0].metadata.get("target") {
            // Parse target data (placeholder)
            let target_tensor = T::random_uniform(
                Shape::new(vec![items.len()]),
                0.0,
                1.0,
                &self.config.device,
            )?;
            Some(target_tensor)
        } else {
            None
        };
        
        // Create attention masks if available
        let attention_masks = if let Some(_mask_data) = items[0].metadata.get("attention_mask") {
            // Parse attention mask (placeholder)
            let mask_tensor = T::random_uniform(
                Shape::new(vec![items.len()]),
                0.0,
                1.0,
                &self.config.device,
            )?;
            Some(mask_tensor)
        } else {
            None
        };
        
        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("batch_size".to_string(), items.len().to_string());
        metadata.insert("device".to_string(), format!("{:?}", self.config.device));
        
        Ok(BatchItem {
            inputs,
            targets,
            attention_masks,
            metadata,
        })
    }
    
    /// Stack tensors into a batch
    fn stack_tensors(&self, tensors: Vec<&T>) -> Result<T> {
        if tensors.is_empty() {
            return Err(TensorError::UnsupportedOperation {
                op: "Cannot stack empty tensor list".to_string(),
            });
        }
        
        // Convert to owned tensors
        let owned_tensors: Vec<T> = tensors.into_iter().cloned().collect();
        
        // Stack along batch dimension (dimension 0)
        T::stack(owned_tensors, 0)
    }
    
    /// Pad sequences to the same length
    fn pad_sequences(&self, sequences: &[&T], max_length: usize) -> Result<Vec<T>> {
        let mut padded = Vec::new();
        
        for &sequence in sequences {
            let shape = sequence.shape();
            let current_length = shape.dim(0).unwrap_or(0);
            
            if current_length < max_length {
                // Create padding tensor
                let pad_length = max_length - current_length;
                let mut pad_shape_vec = vec![pad_length];
                pad_shape_vec.extend_from_slice(&shape.as_slice()[1..]);
                let pad_shape = Shape::new(pad_shape_vec);
                let pad_tensor = T::random_uniform(pad_shape, 0.0, 0.0, sequence.device())?;
                
                // Concatenate sequence and padding
                let padded_sequence = T::cat(vec![sequence.clone(), pad_tensor], 0)?;
                padded.push(padded_sequence);
            } else {
                // Truncate if too long (placeholder)
                let truncated = sequence.clone(); // TODO: Implement proper slicing
                padded.push(truncated);
            }
        }
        
        Ok(padded)
    }
    
    /// Create attention masks for padded sequences
    fn create_attention_masks(&self, sequences: &[&T], max_length: usize) -> Result<Vec<T>> {
        let mut masks = Vec::new();
        
        for &sequence in sequences {
            let shape = sequence.shape();
            let current_length = shape.dim(0).unwrap_or(0);
            
            // Create mask: 1 for real tokens, 0 for padding
            let mut mask_data = vec![1.0; current_length.min(max_length)];
            while mask_data.len() < max_length {
                mask_data.push(0.0); // Padding
            }
            
            let mask_tensor = T::random_uniform(
                Shape::new(vec![max_length]),
                0.0,
                1.0,
                sequence.device(),
            )?;
            
            masks.push(mask_tensor);
        }
        
        Ok(masks)
    }
    
    /// Get batch statistics
    pub fn get_stats(&self) -> BatchStats {
        BatchStats {
            batch_size: self.config.batch_size,
            device: self.config.device.clone(),
        }
    }
}

/// Batch statistics
#[derive(Debug, Clone)]
pub struct BatchStats {
    pub batch_size: usize,
    pub device: Device,
}

/// Batch collator for specific data types
pub struct BatchCollator<T: Tensor> {
    max_length: usize,
    device: Device,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor + tensor_core::tensor::TensorRandom> BatchCollator<T> {
    /// Create a new batch collator
    pub fn new(max_length: usize, device: Device) -> Self {
        Self { 
            max_length, 
            device,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Collate text sequences
    pub fn collate_text(&self, items: Vec<DatasetItem<T>>) -> Result<BatchItem<T>> {
        if items.is_empty() {
            return Err(TensorError::UnsupportedOperation {
                op: "Cannot collate empty items".to_string(),
            });
        }
        
        // Extract sequences
        let sequences: Vec<&T> = items.iter().map(|item| &item.data).collect();
        
        // Pad sequences to max_length
        let batcher = Batcher::<T>::new(BatchConfig {
            batch_size: items.len(),
            device: self.device.clone(),
        })?;
        
        let padded_sequences = batcher.pad_sequences(&sequences, self.max_length)?;
        
        // Stack into batch
        let inputs = T::stack(padded_sequences, 0)?;
        
        // Create attention masks
        let attention_masks = batcher.create_attention_masks(&sequences, self.max_length)?;
        let stacked_masks = T::stack(attention_masks, 0)?;
        
        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("batch_size".to_string(), items.len().to_string());
        metadata.insert("max_length".to_string(), self.max_length.to_string());
        metadata.insert("data_type".to_string(), "text".to_string());
        
        Ok(BatchItem {
            inputs,
            targets: None,
            attention_masks: Some(stacked_masks),
            metadata,
        })
    }
    
    /// Collate image data
    pub fn collate_images(&self, items: Vec<DatasetItem<T>>) -> Result<BatchItem<T>> {
        if items.is_empty() {
            return Err(TensorError::UnsupportedOperation {
                op: "Cannot collate empty items".to_string(),
            });
        }
        
        // Stack image tensors
        let images: Vec<T> = items.iter().map(|item| item.data.clone()).collect();
        let inputs = T::stack(images, 0)?;
        
        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("batch_size".to_string(), items.len().to_string());
        metadata.insert("data_type".to_string(), "image".to_string());
        
        Ok(BatchItem {
            inputs,
            targets: None,
            attention_masks: None,
            metadata,
        })
    }
}
