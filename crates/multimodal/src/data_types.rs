//! 📊 Multimodal Data Types
//!
//! Defines data structures for handling different modalities of data
//! including text, images, audio, video, and 3D point clouds.

use anyhow::Result;
use tensor_core::{Tensor, Shape, DType, Device};
use serde::{Serialize, Deserialize};

/// Text data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextData {
    pub content: String,
    pub encoding: String,
    pub language: Option<String>,
    pub tokens: Vec<String>,
    pub embeddings: Option<Vec<f32>>,
}

/// Image data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub format: ImageFormat,
    pub data: Vec<u8>,
    pub metadata: ImageMetadata,
}

/// Image format types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageFormat {
    RGB,
    RGBA,
    Grayscale,
    YUV,
    HSV,
    Lab,
}

/// Image metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageMetadata {
    pub color_space: Option<String>,
    pub bit_depth: Option<u8>,
    pub compression: Option<String>,
    pub camera_info: Option<CameraInfo>,
}

/// Camera information for images
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraInfo {
    pub make: Option<String>,
    pub model: Option<String>,
    pub focal_length: Option<f32>,
    pub aperture: Option<f32>,
    pub exposure_time: Option<f32>,
    pub iso: Option<u32>,
}

/// Audio data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioData {
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: u8,
    pub duration: f32,
    pub format: AudioFormat,
    pub data: Vec<f32>,
    pub metadata: AudioMetadata,
}

/// Audio format types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    PCM,
    MP3,
    WAV,
    FLAC,
    AAC,
    OGG,
}

/// Audio metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioMetadata {
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub genre: Option<String>,
    pub bitrate: Option<u32>,
    pub codec: Option<String>,
}

/// Video data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoData {
    pub width: u32,
    pub height: u32,
    pub frame_rate: f32,
    pub duration: f32,
    pub format: VideoFormat,
    pub frames: Vec<VideoFrame>,
    pub metadata: VideoMetadata,
}

/// Video format types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoFormat {
    MP4,
    AVI,
    MOV,
    MKV,
    WebM,
    H264,
    H265,
}

/// Individual video frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoFrame {
    pub timestamp: f32,
    pub frame_number: u32,
    pub data: Vec<u8>,
    pub keyframe: bool,
}

/// Video metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VideoMetadata {
    pub title: Option<String>,
    pub duration: Option<f32>,
    pub codec: Option<String>,
    pub bitrate: Option<u32>,
    pub resolution: Option<(u32, u32)>,
}

/// 3D point cloud data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloud3D {
    pub points: Vec<Point3D>,
    pub colors: Option<Vec<Color3D>>,
    pub normals: Option<Vec<Normal3D>>,
    pub metadata: PointCloudMetadata,
}

/// 3D point with coordinates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 3D color information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Color3D {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: Option<u8>,
}

/// 3D normal vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normal3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Point cloud metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PointCloudMetadata {
    pub source: Option<String>,
    pub sensor_type: Option<String>,
    pub coordinate_system: Option<String>,
    pub scale: Option<f32>,
    pub origin: Option<Point3D>,
}

/// Mixed modality data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedModalityData {
    pub text: Option<TextData>,
    pub image: Option<ImageData>,
    pub audio: Option<AudioData>,
    pub video: Option<VideoData>,
    pub pointcloud: Option<PointCloud3D>,
    pub alignment: Option<AlignmentInfo>,
}

/// Alignment information for mixed modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentInfo {
    pub temporal_alignment: Option<TemporalAlignment>,
    pub spatial_alignment: Option<SpatialAlignment>,
    pub semantic_alignment: Option<SemanticAlignment>,
}

/// Temporal alignment between modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAlignment {
    pub start_time: f32,
    pub end_time: f32,
    pub sync_offset: f32,
    pub frame_rate: f32,
}

/// Spatial alignment between modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialAlignment {
    pub transformation_matrix: Vec<f32>, // 4x4 matrix
    pub rotation: (f32, f32, f32),
    pub translation: (f32, f32, f32),
    pub scale: f32,
}

/// Semantic alignment between modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAlignment {
    pub concepts: Vec<String>,
    pub relationships: Vec<ConceptRelationship>,
    pub confidence: f32,
}

/// Relationship between concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptRelationship {
    pub source: String,
    pub target: String,
    pub relationship_type: String,
    pub strength: f32,
}

/// Trait for converting data to tensors
pub trait ToTensor<T: Tensor> {
    fn to_tensor(&self, device: &Device) -> Result<T>;
    fn shape(&self) -> Shape;
    fn dtype(&self) -> DType;
}

/// Trait for converting tensors back to data
pub trait FromTensor<T: Tensor> {
    fn from_tensor(tensor: &T) -> Result<Self> where Self: Sized;
}

// Implementations for different data types

impl ToTensor<backend_cpu::CpuTensor> for TextData {
    fn to_tensor(&self, device: &Device) -> Result<backend_cpu::CpuTensor> {
        // Convert text to token embeddings
        let embeddings = self.embeddings.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No embeddings available"))?;
        
        let shape = Shape::new(vec![embeddings.len() / 768, 768]); // Assuming 768-dim embeddings
        Ok(backend_cpu::CpuTensor::from_slice(embeddings, shape, DType::F32, device)?)
    }

    fn shape(&self) -> Shape {
        let len = self.embeddings.as_ref().map_or(0, |e| e.len() / 768);
        Shape::new(vec![len, 768])
    }

    fn dtype(&self) -> DType {
        DType::F32
    }
}

impl ToTensor<backend_cpu::CpuTensor> for ImageData {
    fn to_tensor(&self, device: &Device) -> Result<backend_cpu::CpuTensor> {
        let shape = Shape::new(vec![self.height as usize, self.width as usize, self.channels as usize]);
        let data: Vec<f32> = self.data.iter().map(|&x| x as f32 / 255.0).collect();
        Ok(backend_cpu::CpuTensor::from_slice(&data, shape, DType::F32, device)?)
    }

    fn shape(&self) -> Shape {
        Shape::new(vec![self.height as usize, self.width as usize, self.channels as usize])
    }

    fn dtype(&self) -> DType {
        DType::F32
    }
}

impl ToTensor<backend_cpu::CpuTensor> for AudioData {
    fn to_tensor(&self, device: &Device) -> Result<backend_cpu::CpuTensor> {
        let shape = Shape::new(vec![self.channels as usize, self.data.len() / self.channels as usize]);
        Ok(backend_cpu::CpuTensor::from_slice(&self.data, shape, DType::F32, device)?)
    }

    fn shape(&self) -> Shape {
        Shape::new(vec![self.channels as usize, self.data.len() / self.channels as usize])
    }

    fn dtype(&self) -> DType {
        DType::F32
    }
}

impl ToTensor<backend_cpu::CpuTensor> for PointCloud3D {
    fn to_tensor(&self, device: &Device) -> Result<backend_cpu::CpuTensor> {
        let points: Vec<f32> = self.points.iter()
            .flat_map(|p| vec![p.x, p.y, p.z])
            .collect();
        let shape = Shape::new(vec![self.points.len(), 3]);
        Ok(backend_cpu::CpuTensor::from_slice(&points, shape, DType::F32, device)?)
    }

    fn shape(&self) -> Shape {
        Shape::new(vec![self.points.len(), 3])
    }

    fn dtype(&self) -> DType {
        DType::F32
    }
}
