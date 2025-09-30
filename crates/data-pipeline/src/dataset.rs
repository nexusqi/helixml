//! Dataset abstraction for HelixML

use tensor_core::{Tensor, Result, TensorError};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// A single item in a dataset
#[derive(Debug, Clone, Serialize)]
pub struct DatasetItem<T: Tensor> {
    pub id: String,
    pub data: T,
    pub metadata: HashMap<String, String>,
}

/// Trait for datasets
#[async_trait::async_trait]
pub trait Dataset<T: Tensor>: Send + Sync {
    /// Get the length of the dataset
    fn len(&self) -> usize;
    
    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get an item by index
    async fn get(&self, index: usize) -> Result<DatasetItem<T>>;
    
    /// Get multiple items by indices
    async fn get_batch(&self, indices: &[usize]) -> Result<Vec<DatasetItem<T>>>;
    
    /// Get all items (for small datasets)
    async fn get_all(&self) -> Result<Vec<DatasetItem<T>>>;
}

/// In-memory dataset
#[derive(Debug)]
pub struct InMemoryDataset<T: Tensor> {
    items: Vec<DatasetItem<T>>,
}

impl<T: Tensor> InMemoryDataset<T> {
    /// Create a new in-memory dataset
    pub fn new(items: Vec<DatasetItem<T>>) -> Self {
        Self { items }
    }
    
    /// Add an item to the dataset
    pub fn add_item(&mut self, item: DatasetItem<T>) {
        self.items.push(item);
    }
    
    /// Get items by indices
    pub fn get_items(&self, indices: &[usize]) -> Result<Vec<DatasetItem<T>>> {
        let mut result = Vec::new();
        for &index in indices {
            if index >= self.items.len() {
                return Err(TensorError::UnsupportedOperation {
                    op: format!("Index {} out of bounds for dataset of size {}", index, self.items.len()),
                });
            }
            result.push(self.items[index].clone());
        }
        Ok(result)
    }
}

#[async_trait::async_trait]
impl<T: Tensor> Dataset<T> for InMemoryDataset<T> {
    fn len(&self) -> usize {
        self.items.len()
    }
    
    async fn get(&self, index: usize) -> Result<DatasetItem<T>> {
        if index >= self.items.len() {
            return Err(TensorError::UnsupportedOperation {
                op: format!("Index {} out of bounds for dataset of size {}", index, self.items.len()),
            });
        }
        Ok(self.items[index].clone())
    }
    
    async fn get_batch(&self, indices: &[usize]) -> Result<Vec<DatasetItem<T>>> {
        self.get_items(indices)
    }
    
    async fn get_all(&self) -> Result<Vec<DatasetItem<T>>> {
        Ok(self.items.clone())
    }
}

/// Dataset iterator for sequential access
pub struct DatasetIterator<T: Tensor> {
    dataset: std::sync::Arc<dyn Dataset<T>>,
    current_index: usize,
    end_index: usize,
}

impl<T: Tensor> DatasetIterator<T> {
    /// Create a new dataset iterator
    pub fn new(dataset: std::sync::Arc<dyn Dataset<T>>, start: usize, end: usize) -> Self {
        Self {
            dataset,
            current_index: start,
            end_index: end,
        }
    }
    
    /// Get the next item
    pub async fn next(&mut self) -> Option<Result<DatasetItem<T>>> {
        if self.current_index >= self.end_index {
            return None;
        }
        
        let result = self.dataset.get(self.current_index).await;
        self.current_index += 1;
        Some(result)
    }
    
    /// Check if there are more items
    pub fn has_next(&self) -> bool {
        self.current_index < self.end_index
    }
}

/// Dataset for text data
pub struct TextDataset<T: Tensor> {
    texts: Vec<String>,
    tokenizer: Box<dyn TextTokenizer>,
    max_length: usize,
    device: tensor_core::Device,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> TextDataset<T> {
    /// Create a new text dataset
    pub fn new(
        texts: Vec<String>,
        tokenizer: Box<dyn TextTokenizer>,
        max_length: usize,
        device: tensor_core::Device,
    ) -> Self {
        Self {
            texts,
            tokenizer,
            max_length,
            device,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait::async_trait]
impl<T: Tensor + tensor_core::tensor::TensorRandom> Dataset<T> for TextDataset<T> {
    fn len(&self) -> usize {
        self.texts.len()
    }
    
    async fn get(&self, index: usize) -> Result<DatasetItem<T>> {
        if index >= self.texts.len() {
            return Err(TensorError::UnsupportedOperation {
                op: format!("Index {} out of bounds for dataset of size {}", index, self.texts.len()),
            });
        }
        
        let text = &self.texts[index];
        let tokens = self.tokenizer.tokenize(text, self.max_length)?;
        
        // Convert tokens to tensor (placeholder)
        let tensor = T::random_uniform(
            tensor_core::Shape::new(vec![tokens.len()]),
            0.0,
            1.0,
            &self.device,
        )?;
        
        let mut metadata = HashMap::new();
        metadata.insert("text".to_string(), text.clone());
        metadata.insert("length".to_string(), tokens.len().to_string());
        
        Ok(DatasetItem {
            id: format!("text_{}", index),
            data: tensor,
            metadata,
        })
    }
    
    async fn get_batch(&self, indices: &[usize]) -> Result<Vec<DatasetItem<T>>> {
        let mut items = Vec::new();
        for &index in indices {
            items.push(self.get(index).await?);
        }
        Ok(items)
    }
    
    async fn get_all(&self) -> Result<Vec<DatasetItem<T>>> {
        let mut items = Vec::new();
        for i in 0..self.len() {
            items.push(self.get(i).await?);
        }
        Ok(items)
    }
}

/// Trait for text tokenization
pub trait TextTokenizer: Send + Sync {
    /// Tokenize text into tokens
    fn tokenize(&self, text: &str, max_length: usize) -> Result<Vec<i32>>;
    
    /// Get vocabulary size
    fn vocab_size(&self) -> usize;
    
    /// Get special tokens
    fn get_special_tokens(&self) -> HashMap<String, i32>;
}

/// Simple character-level tokenizer
pub struct CharTokenizer {
    vocab: HashMap<String, i32>,
    reverse_vocab: HashMap<i32, String>,
    vocab_size: usize,
}

impl CharTokenizer {
    /// Create a new character tokenizer
    pub fn new(texts: &[String]) -> Self {
        let mut chars = std::collections::HashSet::new();
        for text in texts {
            for ch in text.chars() {
                chars.insert(ch);
            }
        }
        
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        
        // Add special tokens
        vocab.insert("<PAD>".to_string(), 0);
        reverse_vocab.insert(0, "<PAD>".to_string());
        vocab.insert("<UNK>".to_string(), 1);
        reverse_vocab.insert(1, "<UNK>".to_string());
        
        let mut next_id = 2;
        for ch in chars {
            vocab.insert(ch.to_string(), next_id);
            reverse_vocab.insert(next_id, ch.to_string());
            next_id += 1;
        }
        
        Self {
            vocab,
            reverse_vocab,
            vocab_size: next_id as usize,
        }
    }
}

impl TextTokenizer for CharTokenizer {
    fn tokenize(&self, text: &str, max_length: usize) -> Result<Vec<i32>> {
        let mut tokens = Vec::new();
        for ch in text.chars().take(max_length) {
            let token = self.vocab.get(&ch.to_string()).copied().unwrap_or(1); // 1 is UNK
            tokens.push(token);
        }
        
        // Pad to max_length
        while tokens.len() < max_length {
            tokens.push(0); // 0 is PAD
        }
        
        Ok(tokens)
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    fn get_special_tokens(&self) -> HashMap<String, i32> {
        let mut special = HashMap::new();
        special.insert("<PAD>".to_string(), 0);
        special.insert("<UNK>".to_string(), 1);
        special
    }
}
