//! 🛠️ Multimodal Utilities
//!
//! Helper functions and utilities for multimodal processing

use tensor_core::{Tensor, Shape, Result};
use tensor_core::tensor::TensorOps;
use crate::{Modality, DataMetadata};

/// Detect modality from shape and metadata
pub fn detect_modality(shape: &Shape, metadata: &DataMetadata) -> Modality {
    let ndim = shape.ndim();
    
    // Check metadata first
    if metadata.sample_rate.is_some() {
        return Modality::Audio;
    }
    
    if metadata.resolution.is_some() {
        return if ndim >= 4 {
            Modality::Video
        } else {
            Modality::Image
        };
    }
    
    // Fallback to shape-based detection
    match ndim {
        1 | 2 => Modality::Text,           // Sequence or batch of sequences
        3 => Modality::Image,              // [C, H, W] or [B, seq_len, hidden]
        4 => Modality::Video,              // [B, T, H, W] or [B, C, H, W]
        _ => Modality::Mixed,
    }
}

/// Validate modality data
pub fn validate_modality_data<T: Tensor>(
    data: &T,
    modality: &Modality,
) -> Result<bool> {
    let shape = data.shape();
    let ndim = shape.ndim();
    
    match modality {
        Modality::Text => {
            // Text should be 1D or 2D (batch)
            Ok(ndim == 1 || ndim == 2)
        }
        Modality::Image => {
            // Images should be 3D [C,H,W] or 4D [B,C,H,W]
            Ok(ndim == 3 || ndim == 4)
        }
        Modality::Audio => {
            // Audio should be 1D (waveform) or 2D (spectrogram)
            Ok(ndim == 1 || ndim == 2)
        }
        Modality::Video => {
            // Video should be 4D [B,T,H,W] or 5D [B,C,T,H,W]
            Ok(ndim == 4 || ndim == 5)
        }
        Modality::PointCloud3D => {
            // Point cloud should be 2D [N,3] or 3D [B,N,3]
            Ok(ndim == 2 || ndim == 3)
        }
        Modality::Mixed => {
            // Mixed modality accepts any shape
            Ok(true)
        }
    }
}

/// Compute modality compatibility score
pub fn modality_compatibility(mod1: &Modality, mod2: &Modality) -> f32 {
    if mod1 == mod2 {
        return 1.0;
    }
    
    match (mod1, mod2) {
        (Modality::Text, Modality::Image) | (Modality::Image, Modality::Text) => 0.8,
        (Modality::Audio, Modality::Video) | (Modality::Video, Modality::Audio) => 0.9,
        (Modality::Text, Modality::Audio) | (Modality::Audio, Modality::Text) => 0.7,
        (Modality::Image, Modality::Video) | (Modality::Video, Modality::Image) => 0.85,
        _ => 0.5,
    }
}

/// Normalize features to same dimension
pub fn normalize_features<T: Tensor + TensorOps>(
    features: Vec<T>,
    target_dim: usize,
) -> Result<Vec<T>> {
    // TODO: Implement proper feature normalization/projection
    // For now, passthrough
    Ok(features)
}

