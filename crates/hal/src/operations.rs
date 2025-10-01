//! 🌀 HelixML Operations and Compute Graph
//! 
//! Operation definitions and computation graph for hardware-agnostic execution.

use crate::{DeviceType, DataType, OperationType, Result, HalError};
use crate::memory::MemoryHandle;
use std::collections::HashMap;

/// Operation definition
#[derive(Debug, Clone)]
pub struct Operation {
    /// Operation type
    pub op_type: OperationType,
    /// Input memory handles
    pub inputs: Vec<MemoryHandle>,
    /// Output memory handle
    pub output: Option<MemoryHandle>,
    /// Operation parameters
    pub params: HashMap<String, serde_json::Value>,
    /// Device where operation should execute
    pub device: Option<DeviceType>,
    /// Precision for operation
    pub precision: Option<DataType>,
    /// Batch size
    pub batch_size: Option<usize>,
    /// Fusion group ID (for kernel fusion)
    pub fusion_group: Option<usize>,
}

impl Operation {
    /// Create new operation
    pub fn new(op_type: OperationType) -> Self {
        Self {
            op_type,
            inputs: Vec::new(),
            output: None,
            params: HashMap::new(),
            device: None,
            precision: None,
            batch_size: None,
            fusion_group: None,
        }
    }
    
    /// Add input
    pub fn add_input(mut self, input: MemoryHandle) -> Self {
        self.inputs.push(input);
        self
    }
    
    /// Set output
    pub fn set_output(mut self, output: MemoryHandle) -> Self {
        self.output = Some(output);
        self
    }
    
    /// Set parameter
    pub fn set_param(mut self, key: String, value: serde_json::Value) -> Self {
        self.params.insert(key, value);
        self
    }
    
    /// Set device
    pub fn set_device(mut self, device: DeviceType) -> Self {
        self.device = Some(device);
        self
    }
    
    /// Set precision
    pub fn set_precision(mut self, precision: DataType) -> Self {
        self.precision = Some(precision);
        self
    }
    
    /// Set batch size
    pub fn set_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }
    
    /// Set fusion group
    pub fn set_fusion_group(mut self, group: usize) -> Self {
        self.fusion_group = Some(group);
        self
    }
}

/// Computation graph for complex operations
#[derive(Debug, Clone)]
pub struct ComputeGraph {
    /// Graph nodes (operations)
    nodes: Vec<Operation>,
    /// Node dependencies (node_id -> [dependent_node_ids])
    dependencies: HashMap<usize, Vec<usize>>,
    /// Input nodes (no dependencies)
    input_nodes: Vec<usize>,
    /// Output nodes (no dependents)
    output_nodes: Vec<usize>,
}

impl ComputeGraph {
    /// Create new computation graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            dependencies: HashMap::new(),
            input_nodes: Vec::new(),
            output_nodes: Vec::new(),
        }
    }
    
    /// Add operation to graph
    pub fn add_operation(&mut self, op: Operation) -> usize {
        let node_id = self.nodes.len();
        self.nodes.push(op);
        node_id
    }
    
    /// Add dependency between nodes
    pub fn add_dependency(&mut self, from: usize, to: usize) -> Result<()> {
        if from >= self.nodes.len() || to >= self.nodes.len() {
            return Err(HalError::OperationError {
                message: "Invalid node ID".to_string(),
            });
        }
        
        self.dependencies.entry(from).or_insert_with(Vec::new).push(to);
        Ok(())
    }
    
    /// Get operation by ID
    pub fn get_operation(&self, node_id: usize) -> Option<&Operation> {
        self.nodes.get(node_id)
    }
    
    /// Get operation mutably by ID
    pub fn get_operation_mut(&mut self, node_id: usize) -> Option<&mut Operation> {
        self.nodes.get_mut(node_id)
    }
    
    /// Get dependencies for node
    pub fn get_dependencies(&self, node_id: usize) -> Option<&Vec<usize>> {
        self.dependencies.get(&node_id)
    }
    
    /// Get all nodes
    pub fn nodes(&self) -> &[Operation] {
        &self.nodes
    }
    
    /// Get input nodes
    pub fn input_nodes(&self) -> &[usize] {
        &self.input_nodes
    }
    
    /// Get output nodes
    pub fn output_nodes(&self) -> &[usize] {
        &self.output_nodes
    }
    
    /// Build graph structure (identify inputs/outputs)
    pub fn build(&mut self) -> Result<()> {
        self.input_nodes.clear();
        self.output_nodes.clear();
        
        // Find input nodes (no dependencies)
        for i in 0..self.nodes.len() {
            if !self.dependencies.values().any(|deps| deps.contains(&i)) {
                self.input_nodes.push(i);
            }
        }
        
        // Find output nodes (no dependents)
        for i in 0..self.nodes.len() {
            if !self.dependencies.contains_key(&i) {
                self.output_nodes.push(i);
            }
        }
        
        Ok(())
    }
    
    /// Get topological sort of nodes
    pub fn topological_sort(&self) -> Result<Vec<usize>> {
        let mut visited = vec![false; self.nodes.len()];
        let mut temp_visited = vec![false; self.nodes.len()];
        let mut result = Vec::new();
        
        for i in 0..self.nodes.len() {
            if !visited[i] {
                self.dfs(i, &mut visited, &mut temp_visited, &mut result)?;
            }
        }
        
        result.reverse();
        Ok(result)
    }
    
    /// Depth-first search for topological sort
    fn dfs(
        &self,
        node: usize,
        visited: &mut [bool],
        temp_visited: &mut [bool],
        result: &mut Vec<usize>,
    ) -> Result<()> {
        if temp_visited[node] {
            return Err(HalError::OperationError {
                message: "Circular dependency detected".to_string(),
            });
        }
        
        if visited[node] {
            return Ok(());
        }
        
        temp_visited[node] = true;
        
        if let Some(deps) = self.dependencies.get(&node) {
            for &dep in deps {
                self.dfs(dep, visited, temp_visited, result)?;
            }
        }
        
        temp_visited[node] = false;
        visited[node] = true;
        result.push(node);
        
        Ok(())
    }
}

/// Operation builder for fluent API
pub struct OperationBuilder {
    op: Operation,
}

impl OperationBuilder {
    /// Create new operation builder
    pub fn new(op_type: OperationType) -> Self {
        Self {
            op: Operation::new(op_type),
        }
    }
    
    /// Add input
    pub fn input(mut self, input: MemoryHandle) -> Self {
        self.op.inputs.push(input);
        self
    }
    
    /// Set output
    pub fn output(mut self, output: MemoryHandle) -> Self {
        self.op.output = Some(output);
        self
    }
    
    /// Set parameter
    pub fn param(mut self, key: &str, value: serde_json::Value) -> Self {
        self.op.params.insert(key.to_string(), value);
        self
    }
    
    /// Set device
    pub fn device(mut self, device: DeviceType) -> Self {
        self.op.device = Some(device);
        self
    }
    
    /// Set precision
    pub fn precision(mut self, precision: DataType) -> Self {
        self.op.precision = Some(precision);
        self
    }
    
    /// Set batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.op.batch_size = Some(batch_size);
        self
    }
    
    /// Set fusion group
    pub fn fusion_group(mut self, group: usize) -> Self {
        self.op.fusion_group = Some(group);
        self
    }
    
    /// Build operation
    pub fn build(self) -> Operation {
        self.op
    }
}

/// Macro for creating operations
#[macro_export]
macro_rules! op {
    ($op_type:expr) => {
        OperationBuilder::new($op_type)
    };
}

