//! 🌀 HelixML Neural Networks
//! 
//! Neural network layers and modules for SSM/Hyena architectures.

use tensor_core::{Tensor, Shape, DType, Device, Result, TensorError};
use tensor_core::tensor::{TensorOps, TensorActivation, TensorRandom, TensorReduce, TensorBroadcast, TensorMixedPrecision};
use std::marker::PhantomData;

// Forward declaration for AutogradContext
pub struct AutogradContext<T: Tensor>(std::marker::PhantomData<T>);

/// Base trait for all neural network modules
pub trait Module<T: Tensor> {
    fn forward(&self, input: &T) -> Result<T>;
    fn parameters(&self) -> Vec<&T>;
    fn parameters_mut(&mut self) -> Vec<&mut T>;
    fn device(&self) -> &Device;
    fn to_device(&mut self, device: &Device) -> Result<()>;
}

/// Trait for modules that support gradient checkpointing
pub trait CheckpointableModule<T: Tensor> {
    /// Forward pass with checkpointing support
    fn forward_checkpointed(&self, input: &T, ctx: &mut AutogradContext<T>) -> Result<usize>;
    
    /// Check if this module should be checkpointed
    fn should_checkpoint(&self) -> bool {
        true // Default to checkpointing for memory efficiency
    }
}

/// Linear layer (fully connected layer)
#[derive(Debug, Clone)]
pub struct Linear<T: Tensor> {
    weight: T,
    bias: Option<T>,
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast> Linear<T> {
    pub fn new(in_features: usize, out_features: usize, device: &Device) -> Result<Self> {
        let weight_shape = Shape::new(vec![out_features, in_features]);
        let weight = T::random_normal(
            weight_shape,
            0.0,
            (2.0 / in_features as f32).sqrt(),
            device,
        )?;
        
        let bias_shape = Shape::new(vec![out_features]);
        let bias = T::zeros(bias_shape, DType::F32, device)?;
        
        Ok(Self {
            weight,
            bias: Some(bias),
            device: device.clone(),
        })
    }
    
    pub fn new_without_bias(in_features: usize, out_features: usize, device: &Device) -> Result<Self> {
        let weight_shape = Shape::new(vec![out_features, in_features]);
        let weight = T::random_normal(
            weight_shape,
            0.0,
            (2.0 / in_features as f32).sqrt(),
            device,
        )?;
        
        Ok(Self {
            weight,
            bias: None,
            device: device.clone(),
        })
    }
    
    pub fn weight(&self) -> &T {
        &self.weight
    }
    
    pub fn weight_mut(&mut self) -> &mut T {
        &mut self.weight
    }
    
    pub fn bias(&self) -> Option<&T> {
        self.bias.as_ref()
    }
    
    pub fn bias_mut(&mut self) -> Option<&mut T> {
        self.bias.as_mut()
    }
}

impl<T: Tensor + TensorOps + TensorBroadcast> Module<T> for Linear<T> {
    fn forward(&self, input: &T) -> Result<T> {
        // input: [batch_size, in_features] or [batch_size, seq_len, in_features]
        // weight: [out_features, in_features]
        // output: [batch_size, out_features] or [batch_size, seq_len, out_features]
        
        let output = input.matmul(&self.weight.transpose(0, 1)?)?;
        
        if let Some(bias) = &self.bias {
            // Use broadcasting to add bias
            let bias_broadcast = bias.broadcast_to(output.shape().clone())?;
            Ok(output.add(&bias_broadcast)?)
        } else {
            Ok(output)
        }
    }
    
    fn parameters(&self) -> Vec<&T> {
        if let Some(bias) = &self.bias {
            vec![&self.weight, bias]
        } else {
            vec![&self.weight]
        }
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut T> {
        if let Some(bias) = &mut self.bias {
            vec![&mut self.weight, bias]
        } else {
            vec![&mut self.weight]
        }
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.weight = self.weight.to_device(device)?;
        if let Some(bias) = &mut self.bias {
            *bias = bias.to_device(device)?;
        }
        self.device = device.clone();
        Ok(())
    }
}

/// RMSNorm (Root Mean Square Normalization)
#[derive(Debug, Clone)]
pub struct RMSNorm<T: Tensor> {
    weight: T,
    eps: f32,
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorReduce> RMSNorm<T> {
    pub fn new(d_model: usize, eps: f32, device: &Device) -> Result<Self> {
        let weight_shape = Shape::new(vec![d_model]);
        let weight = T::ones(weight_shape, DType::F32, device)?;
        
        Ok(Self {
            weight,
            eps,
            device: device.clone(),
        })
    }
    
    pub fn weight(&self) -> &T {
        &self.weight
    }
    
    pub fn weight_mut(&mut self) -> &mut T {
        &mut self.weight
    }
    
    pub fn eps(&self) -> f32 {
        self.eps
    }
    
    pub fn set_eps(&mut self, eps: f32) {
        self.eps = eps;
    }
}

impl<T: Tensor + TensorOps + TensorReduce + TensorRandom + TensorBroadcast> Module<T> for RMSNorm<T> {
    fn forward(&self, input: &T) -> Result<T> {
        // input: [batch_size, seq_len, d_model]
        // Compute RMS
        let mean_squared = input.mul(input)?.mean(Some(vec![input.ndim() - 1]), true)?;
        let eps_tensor = T::random_uniform(mean_squared.shape().clone(), self.eps, self.eps + 1e-8, input.device())?;
        let rms = mean_squared.add(&eps_tensor)?.sqrt()?;
        
        // Normalize - use broadcasting to expand rms to match input shape
        let rms_reshaped = rms.reshape(Shape::new(vec![rms.shape().dim(0).unwrap(), rms.shape().dim(1).unwrap(), 1]))?;
        let rms_broadcasted = rms_reshaped.broadcast_to(input.shape().clone())?;
        let normalized = input.div(&rms_broadcasted)?;
        
        // Scale by weight
        let output = normalized.mul(&self.weight)?;
        
        Ok(output)
    }
    
    fn parameters(&self) -> Vec<&T> {
        vec![&self.weight]
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut T> {
        vec![&mut self.weight]
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.weight = self.weight.to_device(device)?;
        self.device = device.clone();
        Ok(())
    }
}

/// SiLU activation function (Swish)
pub struct SiLU<T: Tensor> {
    device: Device,
    _phantom: PhantomData<T>,
}

impl<T: Tensor> SiLU<T> {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T: Tensor + TensorActivation> Module<T> for SiLU<T> {
    fn forward(&self, input: &T) -> Result<T> {
        input.silu()
    }
    
    fn parameters(&self) -> Vec<&T> {
        vec![]
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut T> {
        vec![]
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        Ok(())
    }
}

/// GELU activation function
pub struct GELU<T: Tensor> {
    device: Device,
    _phantom: PhantomData<T>,
}

impl<T: Tensor> GELU<T> {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T: Tensor + TensorActivation> Module<T> for GELU<T> {
    fn forward(&self, input: &T) -> Result<T> {
        input.gelu()
    }
    
    fn parameters(&self) -> Vec<&T> {
        vec![]
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut T> {
        vec![]
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        Ok(())
    }
}

/// ReLU activation function
pub struct ReLU<T: Tensor> {
    device: Device,
    _phantom: PhantomData<T>,
}

impl<T: Tensor> ReLU<T> {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T: Tensor + TensorActivation> Module<T> for ReLU<T> {
    fn forward(&self, input: &T) -> Result<T> {
        input.relu()
    }
    
    fn parameters(&self) -> Vec<&T> {
        vec![]
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut T> {
        vec![]
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        Ok(())
    }
}

/// Dropout layer
pub struct Dropout<T: Tensor> {
    p: f32,
    training: bool,
    device: Device,
    _phantom: PhantomData<T>,
}

impl<T: Tensor> Dropout<T> {
    pub fn new(p: f32, device: &Device) -> Self {
        Self {
            p,
            training: true,
            device: device.clone(),
            _phantom: PhantomData,
        }
    }
    
    pub fn p(&self) -> f32 {
        self.p
    }
    
    pub fn set_p(&mut self, p: f32) {
        self.p = p;
    }
    
    pub fn training(&self) -> bool {
        self.training
    }
    
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
    
    pub fn train(&mut self) {
        self.training = true;
    }
    
    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl<T: Tensor + TensorRandom + TensorOps> Module<T> for Dropout<T> {
    fn forward(&self, input: &T) -> Result<T> {
        if !self.training {
            return Ok(input.clone());
        }
        
        let mask = T::random_bernoulli(input.shape().clone(), 1.0 - self.p, input.device())?;
        let scale = 1.0 / (1.0 - self.p);
        let scaled_mask = mask.mul(&T::random_uniform(Shape::new(vec![]), scale, scale, input.device())?)?;
        
        input.mul(&scaled_mask)
    }
    
    fn parameters(&self) -> Vec<&T> {
        vec![]
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut T> {
        vec![]
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        Ok(())
    }
}

/// Sequential container for modules
pub struct Sequential<T: Tensor> {
    modules: Vec<Box<dyn Module<T> + Send + Sync>>,
    device: Device,
}

impl<T: Tensor> Sequential<T> {
    pub fn new(device: &Device) -> Self {
        Self {
            modules: Vec::new(),
            device: device.clone(),
        }
    }
    
    pub fn add<M: Module<T> + Send + Sync + 'static>(&mut self, module: M) {
        self.modules.push(Box::new(module));
    }
    
    pub fn len(&self) -> usize {
        self.modules.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl<T: Tensor> Module<T> for Sequential<T> {
    fn forward(&self, input: &T) -> Result<T> {
        let mut output = input.clone();
        
        for module in &self.modules {
            output = module.forward(&output)?;
        }
        
        Ok(output)
    }
    
    fn parameters(&self) -> Vec<&T> {
        let mut params = Vec::new();
        for module in &self.modules {
            params.extend(module.parameters());
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut T> {
        let mut params = Vec::new();
        for module in &mut self.modules {
            params.extend(module.parameters_mut());
        }
        params
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        for module in &mut self.modules {
            module.to_device(device)?;
        }
        self.device = device.clone();
        Ok(())
    }
}

/// State-Space Model (SSM) blocks for sequence modeling
/// These replace self-attention in transformers with more efficient state-space operations

/// S4 (Structured State Space) layer
/// Based on "Efficiently Modeling Long Sequences with Structured State Spaces"
#[derive(Debug, Clone)]
pub struct S4Block<T: Tensor> {
    // State space parameters
    A: T,           // State transition matrix (complex)
    B: T,           // Input matrix
    C: T,           // Output matrix
    D: T,           // Feedthrough matrix
    
    // Learnable parameters
    delta: T,       // Time step parameter
    Lambda: T,      // Diagonal state matrix
    
    // Dimensions
    d_model: usize,
    d_state: usize,
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision> S4Block<T> {
    pub fn new(d_model: usize, d_state: usize, device: &Device) -> Result<Self> {
        // Initialize state space parameters
        let A = T::random_normal(
            Shape::new(vec![d_state, d_state]),
            0.0,
            0.1,
            device,
        )?;
        
        let B = T::random_normal(
            Shape::new(vec![d_state, d_model]),
            0.0,
            0.1,
            device,
        )?;
        
        let C = T::random_normal(
            Shape::new(vec![d_model, d_state]),
            0.0,
            0.1,
            device,
        )?;
        
        let D = T::zeros(Shape::new(vec![d_model]), DType::F32, device)?;
        
        // Time step parameter
        let delta = T::random_uniform(
            Shape::new(vec![d_model]),
            0.001,
            0.1,
            device,
        )?;
        
        // Diagonal state matrix (Lambda)
        let Lambda = T::random_normal(
            Shape::new(vec![d_state]),
            0.0,
            1.0,
            device,
        )?;
        
        Ok(Self {
            A,
            B,
            C,
            D,
            delta,
            Lambda,
            d_model,
            d_state,
            device: device.clone(),
        })
    }
    
    /// Forward pass through S4 layer
    pub fn forward_ssm(&self, input: &T) -> Result<T> {
        // input: [seq_len, d_model]
        let _batch_size = input.shape().dim(0).unwrap();
        let _seq_len = input.shape().dim(1).unwrap();
        
        // For now, implement a simplified version
        // In practice, this would involve:
        // 1. Discretization of continuous-time state space
        // 2. Convolution or recurrence operations
        // 3. Complex number arithmetic for A matrix
        
        // Simplified implementation: just return input with residual connection
        // The actual SSM computation would be much more complex
        Ok(input.clone())
    }
    
    /// Get state dimension
    pub fn d_state(&self) -> usize {
        self.d_state
    }
    
    /// Get model dimension
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision> Module<T> for S4Block<T> {
    fn forward(&self, input: &T) -> Result<T> {
        self.forward_ssm(input)
    }
    
    fn parameters(&self) -> Vec<&T> {
        vec![&self.A, &self.B, &self.C, &self.D, &self.delta, &self.Lambda]
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut T> {
        vec![&mut self.A, &mut self.B, &mut self.C, &mut self.D, &mut self.delta, &mut self.Lambda]
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        // In practice, would move all tensors to the new device
        Ok(())
    }
}

/// Mamba block - efficient SSM with selective mechanisms
/// Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
#[derive(Debug, Clone)]
pub struct MambaBlock<T: Tensor> {
    // Selective state space parameters
    A: T,           // State transition matrix
    B: T,           // Input matrix  
    C: T,           // Output matrix
    D: T,           // Feedthrough matrix
    
    // Selective mechanism
    delta_proj: Linear<T>,      // Time step projection
    B_proj: Linear<T>,          // Input projection
    C_proj: Linear<T>,          // Output projection
    
    // Gating mechanism
    gate: Linear<T>,            // Gate projection
    
    // Output projection
    out_proj: Linear<T>,        // Output projection
    
    // Dimensions
    d_model: usize,
    d_state: usize,
    d_conv: usize,
    expand: usize,
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorActivation> MambaBlock<T> {
    pub fn new(d_model: usize, d_state: usize, d_conv: usize, expand: usize, device: &Device) -> Result<Self> {
        let d_inner = d_model * expand;
        
        // Initialize state space parameters
        let A = T::random_normal(
            Shape::new(vec![d_state, d_state]),
            0.0,
            0.1,
            device,
        )?;
        
        let B = T::zeros(Shape::new(vec![d_state, d_inner]), DType::F32, device)?;
        let C = T::zeros(Shape::new(vec![d_inner, d_state]), DType::F32, device)?;
        let D = T::zeros(Shape::new(vec![d_inner]), DType::F32, device)?;
        
        // Initialize projections
        let delta_proj = Linear::<T>::new(d_inner, d_inner, device)?;
        let B_proj = Linear::<T>::new(d_inner, d_state * d_inner, device)?;
        let C_proj = Linear::<T>::new(d_inner, d_state * d_inner, device)?;
        let gate = Linear::<T>::new(d_model, d_inner, device)?;
        let out_proj = Linear::<T>::new(d_inner, d_model, device)?;
        
        Ok(Self {
            A,
            B,
            C,
            D,
            delta_proj,
            B_proj,
            C_proj,
            gate,
            out_proj,
            d_model,
            d_state,
            d_conv,
            expand,
            device: device.clone(),
        })
    }
    
    /// Forward pass through Mamba block
    pub fn forward_mamba(&self, input: &T) -> Result<T> {
        // input: [seq_len, d_model]
        let seq_len = input.shape().dim(0).unwrap();
        let d_model = input.shape().dim(1).unwrap();
        
        // 1. Gate projection
        let xz = self.gate.forward(input)?;
        
        // 2. For now, simplified implementation without slicing
        // In practice, would split xz into x and z parts
        
        // 3. Apply SiLU activation
        let activated = xz.silu()?;
        
        // 4. Output projection
        let output = self.out_proj.forward(&activated)?;
        
        Ok(output)
    }
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorActivation> Module<T> for MambaBlock<T> {
    fn forward(&self, input: &T) -> Result<T> {
        self.forward_mamba(input)
    }
    
    fn parameters(&self) -> Vec<&T> {
        let mut params = vec![&self.A, &self.B, &self.C, &self.D];
        params.extend(self.delta_proj.parameters());
        params.extend(self.B_proj.parameters());
        params.extend(self.C_proj.parameters());
        params.extend(self.gate.parameters());
        params.extend(self.out_proj.parameters());
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut T> {
        let mut params = vec![&mut self.A, &mut self.B, &mut self.C, &mut self.D];
        params.extend(self.delta_proj.parameters_mut());
        params.extend(self.B_proj.parameters_mut());
        params.extend(self.C_proj.parameters_mut());
        params.extend(self.gate.parameters_mut());
        params.extend(self.out_proj.parameters_mut());
        params
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.delta_proj.to_device(device)?;
        self.B_proj.to_device(device)?;
        self.C_proj.to_device(device)?;
        self.gate.to_device(device)?;
        self.out_proj.to_device(device)?;
        Ok(())
    }
}

/// Hyena block - FFT-based long convolution architecture
/// Based on "Hyena Hierarchy: Towards Larger Convolutional Language Models"
#[derive(Debug, Clone)]
pub struct HyenaBlock<T: Tensor> {
    // Input projections
    input_proj: Linear<T>,        // Input projection
    output_proj: Linear<T>,       // Output projection
    
    // Hyena-specific components
    short_conv: Linear<T>,        // Short convolution layer
    long_conv_weights: T,         // Long convolution weights
    
    // Gating mechanism
    gate: Linear<T>,              // Gate projection
    
    // FFT parameters
    max_length: usize,            // Maximum sequence length
    num_fft_layers: usize,        // Number of FFT layers
    
    // Dimensions
    d_model: usize,
    d_ff: usize,                  // Feed-forward dimension
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorActivation> HyenaBlock<T> {
    pub fn new(d_model: usize, d_ff: usize, max_length: usize, num_fft_layers: usize, device: &Device) -> Result<Self> {
        // Initialize projections
        let input_proj = Linear::<T>::new(d_model, d_ff, device)?;
        let output_proj = Linear::<T>::new(d_ff, d_model, device)?;
        
        // Short convolution (1D conv)
        let short_conv = Linear::<T>::new(d_ff, d_ff, device)?;
        
        // Long convolution weights (simplified)
        let long_conv_weights = T::random_normal(
            Shape::new(vec![max_length, d_ff]),
            0.0,
            0.1,
            device,
        )?;
        
        // Gate projection
        let gate = Linear::<T>::new(d_model, d_ff, device)?;
        
        Ok(Self {
            input_proj,
            output_proj,
            short_conv,
            long_conv_weights,
            gate,
            max_length,
            num_fft_layers,
            d_model,
            d_ff,
            device: device.clone(),
        })
    }
    
    /// Forward pass through Hyena block
    pub fn forward_hyena(&self, input: &T) -> Result<T> {
        // input: [seq_len, d_model]
        let seq_len = input.shape().dim(0).unwrap();
        let _d_model = input.shape().dim(1).unwrap();
        
        // 1. Input projection
        let projected = self.input_proj.forward(input)?;
        
        // 2. Short convolution (simplified as linear for now)
        let short_conv_out = self.short_conv.forward(&projected)?;
        
        // 3. Long convolution via FFT (simplified)
        // In practice, this would involve:
        // - FFT of input sequence
        // - Element-wise multiplication with learned filters
        // - IFFT back to time domain
        let long_conv_out = short_conv_out; // Simplified: just pass through
        
        // 4. Gating mechanism
        let gate_values = self.gate.forward(input)?;
        let gated = long_conv_out.mul(&gate_values)?;
        
        // 5. Output projection
        let output = self.output_proj.forward(&gated)?;
        
        Ok(output)
    }
    
    /// Get model dimension
    pub fn d_model(&self) -> usize {
        self.d_model
    }
    
    /// Get feed-forward dimension
    pub fn d_ff(&self) -> usize {
        self.d_ff
    }
    
    /// Get maximum sequence length
    pub fn max_length(&self) -> usize {
        self.max_length
    }
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorActivation> Module<T> for HyenaBlock<T> {
    fn forward(&self, input: &T) -> Result<T> {
        self.forward_hyena(input)
    }
    
    fn parameters(&self) -> Vec<&T> {
        let mut params = Vec::new();
        params.extend(self.input_proj.parameters());
        params.extend(self.output_proj.parameters());
        params.extend(self.short_conv.parameters());
        params.push(&self.long_conv_weights);
        params.extend(self.gate.parameters());
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut T> {
        let mut params = Vec::new();
        params.extend(self.input_proj.parameters_mut());
        params.extend(self.output_proj.parameters_mut());
        params.extend(self.short_conv.parameters_mut());
        params.push(&mut self.long_conv_weights);
        params.extend(self.gate.parameters_mut());
        params
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        self.input_proj.to_device(device)?;
        self.output_proj.to_device(device)?;
        self.short_conv.to_device(device)?;
        self.gate.to_device(device)?;
        // In practice, would move long_conv_weights to new device
        Ok(())
    }
}

/// HyenaOperator - Core FFT-based convolution operator
/// This is the heart of the Hyena architecture
#[derive(Debug, Clone)]
pub struct HyenaOperator<T: Tensor> {
    // FFT-based convolution parameters
    filters: T,                   // Learnable filters in frequency domain
    bias: T,                      // Bias terms
    
    // FFT configuration
    fft_size: usize,              // FFT size
    num_filters: usize,           // Number of filters
    
    // Dimensions
    d_model: usize,
    device: Device,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision> HyenaOperator<T> {
    pub fn new(d_model: usize, fft_size: usize, num_filters: usize, device: &Device) -> Result<Self> {
        // Initialize filters in frequency domain
        let filters = T::random_normal(
            Shape::new(vec![fft_size, num_filters]),
            0.0,
            0.1,
            device,
        )?;
        
        // Initialize bias
        let bias = T::zeros(Shape::new(vec![num_filters]), DType::F32, device)?;
        
        Ok(Self {
            filters,
            bias,
            fft_size,
            num_filters,
            d_model,
            device: device.clone(),
        })
    }
    
    /// Apply FFT-based convolution
    pub fn forward_operator(&self, input: &T) -> Result<T> {
        // input: [seq_len, d_model]
        let seq_len = input.shape().dim(0).unwrap();
        let d_model = input.shape().dim(1).unwrap();
        
        // For now, simplified implementation
        // In practice, this would involve:
        // 1. FFT of input sequence
        // 2. Element-wise multiplication with filters
        // 3. IFFT back to time domain
        // 4. Apply bias
        
        // Simplified: just return input with some transformation
        let transformed = input.clone();
        Ok(transformed)
    }
    
    /// Get FFT size
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }
    
    /// Get number of filters
    pub fn num_filters(&self) -> usize {
        self.num_filters
    }
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision> Module<T> for HyenaOperator<T> {
    fn forward(&self, input: &T) -> Result<T> {
        self.forward_operator(input)
    }
    
    fn parameters(&self) -> Vec<&T> {
        vec![&self.filters, &self.bias]
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut T> {
        vec![&mut self.filters, &mut self.bias]
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        // In practice, would move tensors to new device
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend_cpu::CpuTensor;
    
    #[test]
    fn test_linear_layer() {
        let device = Device::cpu();
        let mut linear = Linear::<CpuTensor>::new(10, 5, &device).unwrap();
        
        let input = CpuTensor::random_uniform(Shape::new(vec![2, 10]), 0.0, 1.0, &device).unwrap();
        let output = linear.forward(&input).unwrap();
        
        assert_eq!(output.shape(), &Shape::new(vec![2, 5]));
        assert_eq!(linear.parameters().len(), 2); // weight + bias
    }

    #[test]
    fn test_rmsnorm() {
        let device = Device::cpu();
        let rmsnorm = RMSNorm::<CpuTensor>::new(512, 1e-6, &device).unwrap();
        
        let input = CpuTensor::random_uniform(Shape::new(vec![2, 10, 512]), 0.0, 1.0, &device).unwrap();
        let output = rmsnorm.forward(&input).unwrap();
        
        assert_eq!(output.shape(), input.shape());
        assert_eq!(rmsnorm.parameters().len(), 1); // weight only
    }
    
    #[test]
    fn test_silu_activation() {
        let device = Device::cpu();
        let silu = SiLU::<CpuTensor>::new(&device);
        
        let input = CpuTensor::random_uniform(Shape::new(vec![2, 10]), 0.0, 1.0, &device).unwrap();
        let output = silu.forward(&input).unwrap();
        
        assert_eq!(output.shape(), input.shape());
        assert_eq!(silu.parameters().len(), 0); // no parameters
    }
    
    #[test]
    fn test_sequential() {
        let device = Device::cpu();
        let mut seq = Sequential::<CpuTensor>::new(&device);
        
        seq.add(Linear::<CpuTensor>::new(10, 5, &device).unwrap());
        seq.add(SiLU::<CpuTensor>::new(&device));
        seq.add(Linear::<CpuTensor>::new(5, 3, &device).unwrap());
        
        let input = CpuTensor::random_uniform(Shape::new(vec![2, 10]), 0.0, 1.0, &device).unwrap();
        let output = seq.forward(&input).unwrap();
        
        assert_eq!(output.shape(), &Shape::new(vec![2, 3]));
        assert_eq!(seq.parameters().len(), 4); // 2 linear layers with bias
    }
}

// Phase synchronization utilities for SSM cores
pub mod phase_sync {
    use super::*;
    use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision, TensorReduce};

    /// Calculate phase synchronization metric between channels
    pub fn phase_sync_metric<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorReduce>(
        hidden_states: &T
    ) -> Result<f32> {
        let shape = hidden_states.shape();
        if shape.dim(0).is_none() || shape.dim(1).is_none() {
            return Ok(0.0);
        }

        let seq_len = shape.dim(0).unwrap();
        let d_model = shape.dim(1).unwrap();

        if seq_len < 2 || d_model < 2 {
            return Ok(0.0);
        }

        // Simplified phase sync calculation using tensor operations
        // Calculate variance as a proxy for phase synchronization
        let variance = hidden_states.var(Some(vec![0]), false)?;
        let mean_var = variance.mean(Some(vec![0]), false)?;
        
        // Convert to scalar approximation (simplified)
        Ok(mean_var.shape().numel() as f32 * 0.1) // Placeholder calculation
    }

    /// Calculate instantaneous phase from complex-like representation
    pub fn calculate_instantaneous_phase<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision>(
        real_part: &T, 
        imag_part: &T
    ) -> Result<T> {
        // For real tensors, we can simulate phase by using atan2-like operation
        // This is a simplified version for demonstration
        let phase = real_part.div(imag_part)?;
        
        // Apply arctan-like transformation (simplified)
        let phase_atan = phase.pow(0.33)?; // Simplified arctan approximation
        
        Ok(phase_atan)
    }

    /// Calculate phase coherence between multiple sequences
    pub fn calculate_phase_coherence<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorReduce>(
        sequences: &[T]
    ) -> Result<f32> {
        if sequences.len() < 2 {
            return Ok(0.0);
        }

        let mut total_coherence = 0.0;
        let mut comparisons = 0;

        for i in 0..sequences.len() {
            for j in (i + 1)..sequences.len() {
                let coherence = calculate_pairwise_coherence(&sequences[i], &sequences[j])?;
                total_coherence += coherence;
                comparisons += 1;
            }
        }

        Ok(if comparisons > 0 { total_coherence / comparisons as f32 } else { 0.0 })
    }

    /// Calculate pairwise coherence between two sequences
    fn calculate_pairwise_coherence<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorReduce>(
        seq_a: &T, 
        seq_b: &T
    ) -> Result<f32> {
        let shape_a = seq_a.shape();
        let shape_b = seq_b.shape();

        if shape_a.dim(0).is_none() || shape_b.dim(0).is_none() {
            return Ok(0.0);
        }

        let len_a = shape_a.dim(0).unwrap();
        let len_b = shape_b.dim(0).unwrap();

        if len_a != len_b {
            return Ok(0.0);
        }

        // Calculate coherence using tensor operations
        let correlation = seq_a.mul(seq_b)?;
        
        // Simplified coherence calculation
        let coherence_sum = correlation.sum(Some(vec![0]), false)?;
        
        Ok(coherence_sum.shape().numel() as f32 * 0.1) // Placeholder
    }

    /// Phase synchronization index (PSI) for SSM states
    pub fn phase_synchronization_index<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision>(
        states: &[T]
    ) -> Result<f32> {
        if states.len() < 2 {
            return Ok(0.0);
        }

        let mut psi_sum = 0.0;
        let mut comparisons = 0;

        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                let psi = calculate_psi_pair(&states[i], &states[j])?;
                psi_sum += psi;
                comparisons += 1;
            }
        }

        Ok(if comparisons > 0 { psi_sum / comparisons as f32 } else { 0.0 })
    }

    /// Calculate PSI between two state vectors
    fn calculate_psi_pair<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision>(
        state_a: &T, 
        state_b: &T
    ) -> Result<f32> {
        let shape_a = state_a.shape();
        let shape_b = state_b.shape();

        if shape_a.dim(0).is_none() || shape_b.dim(0).is_none() {
            return Ok(0.0);
        }

        let len_a = shape_a.dim(0).unwrap();
        let len_b = shape_b.dim(0).unwrap();

        if len_a != len_b {
            return Ok(0.0);
        }

        // Calculate PSI using tensor operations
        let diff = state_a.sub(state_b)?;
        let abs_diff = diff.abs()?;
        let one_tensor = T::ones(state_a.shape().clone(), state_a.dtype(), state_a.device())?;
        let psi_tensor = one_tensor.sub(&abs_diff)?;
        let zero_tensor = T::zeros(state_a.shape().clone(), state_a.dtype(), state_a.device())?;
        let psi_max = psi_tensor.max(&zero_tensor)?;
        
        Ok(psi_max.shape().numel() as f32 * 0.1) // Placeholder
    }
}