//! 📊 Universal Compute Graph

use tensor_core::Tensor;
use serde::{Serialize, Deserialize};

/// Architecture types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Architecture {
    Transformer,
    Mamba,
    SSM,
    Hyena,
    CNN,
    RNN,
    GNN,
    Custom,
    Auto, // Auto-select
}

/// Compute node in the universal graph
#[derive(Debug, Clone)]
pub enum ComputeNode {
    Linear,
    Conv,
    Attention,
    SSM,
    Mamba,
    Hyena,
    Graph,
    Custom(String),
}

/// Universal compute graph
pub struct UniversalGraph<T: Tensor> {
    pub nodes: Vec<ComputeNode>,
    pub architecture: Architecture,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor> UniversalGraph<T> {
    pub fn new(architecture: Architecture) -> Self {
        Self {
            nodes: Vec::new(),
            architecture,
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn add_node(&mut self, node: ComputeNode) {
        self.nodes.push(node);
    }
}

