//! Data caching for HelixML

use tensor_core::{Tensor, Result, TensorError};
use crate::batcher::BatchItem;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use rand::seq::SliceRandom;

/// Configuration for data cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_size: usize,
    pub device: tensor_core::Device,
}

/// Data cache for storing processed batches
#[derive(Debug)]
pub struct DataCache<T: Tensor> {
    cache: Arc<RwLock<HashMap<String, BatchItem<T>>>>,
    config: CacheConfig,
    stats: CacheStats,
}

impl<T: Tensor> DataCache<T> {
    /// Create a new data cache
    pub fn new(config: CacheConfig) -> Result<Self> {
        Ok(Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: CacheStats::default(),
        })
    }
    
    /// Get a batch from cache
    pub async fn get(&mut self, key: &str) -> Result<Option<BatchItem<T>>> {
        let cache = self.cache.read().await;
        if let Some(batch) = cache.get(key) {
            self.stats.hits += 1;
            Ok(Some(batch.clone()))
        } else {
            self.stats.misses += 1;
            Ok(None)
        }
    }
    
    /// Put a batch into cache
    pub async fn put(&mut self, key: String, batch: BatchItem<T>) -> Result<()> {
        let mut cache = self.cache.write().await;
        
        // Check if cache is full
        if cache.len() >= self.config.max_size {
            // Remove oldest entry (simple LRU)
            if let Some(oldest_key) = cache.keys().next().cloned() {
                cache.remove(&oldest_key);
            }
        }
        
        cache.insert(key, batch);
        self.stats.inserts += 1;
        Ok(())
    }
    
    /// Remove a batch from cache
    pub async fn remove(&mut self, key: &str) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.remove(key);
        self.stats.removes += 1;
        Ok(())
    }
    
    /// Clear the cache
    pub async fn clear(&mut self) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.clear();
        self.stats.clears += 1;
        Ok(())
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.stats.clone()
    }
    
    /// Get cache size
    pub async fn size(&self) -> usize {
        let cache = self.cache.read().await;
        cache.len()
    }
    
    /// Check if cache is full
    pub async fn is_full(&self) -> bool {
        self.size().await >= self.config.max_size
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub inserts: usize,
    pub removes: usize,
    pub clears: usize,
}

impl CacheStats {
    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
    
    /// Get total operations
    pub fn total_operations(&self) -> usize {
        self.hits + self.misses + self.inserts + self.removes + self.clears
    }
}

/// LRU cache implementation
#[derive(Debug)]
pub struct LRUCache<T: Tensor> {
    cache: Arc<RwLock<HashMap<String, (BatchItem<T>, usize)>>>,
    access_order: Arc<RwLock<Vec<String>>>,
    config: CacheConfig,
    stats: CacheStats,
}

impl<T: Tensor> LRUCache<T> {
    /// Create a new LRU cache
    pub fn new(config: CacheConfig) -> Result<Self> {
        Ok(Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(RwLock::new(Vec::new())),
            config,
            stats: CacheStats::default(),
        })
    }
    
    /// Get a batch from cache
    pub async fn get(&mut self, key: &str) -> Result<Option<BatchItem<T>>> {
        let mut cache = self.cache.write().await;
        let mut access_order = self.access_order.write().await;
        
        if let Some((batch, _)) = cache.get_mut(key) {
            // Update access order
            if let Some(pos) = access_order.iter().position(|k| k == key) {
                access_order.remove(pos);
            }
            access_order.push(key.to_string());
            
            self.stats.hits += 1;
            Ok(Some(batch.clone()))
        } else {
            self.stats.misses += 1;
            Ok(None)
        }
    }
    
    /// Put a batch into cache
    pub async fn put(&mut self, key: String, batch: BatchItem<T>) -> Result<()> {
        let mut cache = self.cache.write().await;
        let mut access_order = self.access_order.write().await;
        
        // Check if cache is full
        if cache.len() >= self.config.max_size {
            // Remove least recently used item
            if let Some(lru_key) = access_order.first().cloned() {
                cache.remove(&lru_key);
                access_order.remove(0);
            }
        }
        
        cache.insert(key.clone(), (batch, 0));
        access_order.push(key);
        self.stats.inserts += 1;
        Ok(())
    }
    
    /// Remove a batch from cache
    pub async fn remove(&mut self, key: &str) -> Result<()> {
        let mut cache = self.cache.write().await;
        let mut access_order = self.access_order.write().await;
        
        cache.remove(key);
        if let Some(pos) = access_order.iter().position(|k| k == key) {
            access_order.remove(pos);
        }
        
        self.stats.removes += 1;
        Ok(())
    }
    
    /// Clear the cache
    pub async fn clear(&mut self) -> Result<()> {
        let mut cache = self.cache.write().await;
        let mut access_order = self.access_order.write().await;
        
        cache.clear();
        access_order.clear();
        
        self.stats.clears += 1;
        Ok(())
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.stats.clone()
    }
}

/// Cache eviction strategies
#[derive(Debug, Clone)]
pub enum EvictionStrategy {
    LRU,    // Least Recently Used
    LFU,    // Least Frequently Used
    FIFO,   // First In, First Out
    Random, // Random eviction
}

/// Advanced cache with configurable eviction strategy
#[derive(Debug)]
pub struct AdvancedCache<T: Tensor> {
    cache: Arc<RwLock<HashMap<String, (BatchItem<T>, usize, usize)>>>,
    config: CacheConfig,
    strategy: EvictionStrategy,
    stats: CacheStats,
}

impl<T: Tensor> AdvancedCache<T> {
    /// Create a new advanced cache
    pub fn new(config: CacheConfig, strategy: EvictionStrategy) -> Result<Self> {
        Ok(Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            strategy,
            stats: CacheStats::default(),
        })
    }
    
    /// Get a batch from cache
    pub async fn get(&mut self, key: &str) -> Result<Option<BatchItem<T>>> {
        let mut cache = self.cache.write().await;
        
        if let Some((batch, access_count, last_access)) = cache.get_mut(key) {
            *access_count += 1;
            *last_access = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as usize;
            
            self.stats.hits += 1;
            Ok(Some(batch.clone()))
        } else {
            self.stats.misses += 1;
            Ok(None)
        }
    }
    
    /// Put a batch into cache
    pub async fn put(&mut self, key: String, batch: BatchItem<T>) -> Result<()> {
        let mut cache = self.cache.write().await;
        
        // Check if cache is full
        if cache.len() >= self.config.max_size {
            self.evict_entry(&mut cache).await?;
        }
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as usize;
        
        cache.insert(key, (batch, 1, now));
        self.stats.inserts += 1;
        Ok(())
    }
    
    /// Evict an entry based on strategy
    async fn evict_entry(&self, cache: &mut HashMap<String, (BatchItem<T>, usize, usize)>) -> Result<()> {
        let key_to_evict = match self.strategy {
            EvictionStrategy::LRU => {
                cache.iter()
                    .min_by_key(|(_, (_, _, last_access))| *last_access)
                    .map(|(k, _)| k.clone())
            }
            EvictionStrategy::LFU => {
                cache.iter()
                    .min_by_key(|(_, (_, access_count, _))| *access_count)
                    .map(|(k, _)| k.clone())
            }
            EvictionStrategy::FIFO => {
                cache.keys().next().cloned()
            }
            EvictionStrategy::Random => {
                use rand::seq::SliceRandom;
                cache.keys().collect::<Vec<_>>().choose(&mut rand::thread_rng()).cloned().cloned()
            }
        };
        
        if let Some(key) = key_to_evict {
            cache.remove(&key);
        }
        
        Ok(())
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.stats.clone()
    }
}
