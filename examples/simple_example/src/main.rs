//! 🌀 HelixML Simple Example
//! 
//! Простой пример использования фреймворка HelixML для создания и обучения нейронной сети.

use helix_ml::*;
use helix_ml::tensor::{TensorRandom, TensorOps, TensorActivation};

fn main() -> Result<()> {
    println!("🌀 HelixML Simple Example");
    println!("=========================");
    
    // Создаем устройство (CPU)
    let device = Device::cpu();
    println!("Device: {:?}", device);
    
    // Создаем простые тензоры для демонстрации
    println!("\nCreating tensors:");
    
    // Создаем тензор 2x3
    let a = CpuTensor::ones(Shape::new(vec![2, 3]), DType::F32, &device)?;
    println!("Tensor A shape: {:?}", a.shape());
    println!("Tensor A data: {:?}", a.data());
    
    // Создаем тензор 2x3 (такой же размер как у A)
    let b = CpuTensor::ones(Shape::new(vec![2, 3]), DType::F32, &device)?;
    println!("Tensor B shape: {:?}", b.shape());
    
    // Простое сложение
    println!("\nTensor operations:");
    let c = a.add(&b)?;
    println!("A + B shape: {:?}", c.shape());
    
    // Создаем тензор с случайными данными
    let random_tensor = CpuTensor::random_uniform(
        Shape::new(vec![2, 2]), 
        0.0, 
        1.0, 
        &device
    )?;
    println!("Random tensor shape: {:?}", random_tensor.shape());
    println!("Random tensor data: {:?}", random_tensor.data());
    
    // Простые математические операции
    println!("\nMathematical operations:");
    let doubled = random_tensor.mul(&CpuTensor::ones(Shape::new(vec![2, 2]), DType::F32, &device)?)?;
    println!("Doubled tensor data: {:?}", doubled.data());
    
    let squared = random_tensor.mul(&random_tensor)?;
    println!("Squared tensor data: {:?}", squared.data());
    
    // Демонстрация активаций
    println!("\nActivation functions:");
    let relu_result = random_tensor.relu()?;
    println!("ReLU result: {:?}", relu_result.data());
    
    let silu_result = random_tensor.silu()?;
    println!("SiLU result: {:?}", silu_result.data());
    
    println!("\n✅ Example completed successfully!");
    
    Ok(())
}