use std::env;
use std::path::PathBuf;

fn main() {
    // Only compile CUDA kernels if CUDA feature is enabled
    #[cfg(feature = "cuda")]
    {
        // Check if CUDA is available
        if env::var("CUDA_ROOT").is_err() && env::var("CUDA_HOME").is_err() {
            println!("cargo:warning=CUDA not found, CUDA backend will not be compiled");
            return;
        }

        let cuda_root = env::var("CUDA_ROOT")
            .or_else(|_| env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());

        let cuda_include = format!("{}/include", cuda_root);
        let cuda_lib = format!("{}/lib64", cuda_root);

        // Compile CUDA kernels
        let kernel_path = PathBuf::from("src/kernels.cu");
        
        if kernel_path.exists() {
            cc::Build::new()
                .cuda(true)
                .file("src/kernels.cu")
                .include(&cuda_include)
                .flag("-gencode")
                .flag("arch=compute_50,code=sm_50")
                .flag("-gencode")
                .flag("arch=compute_60,code=sm_60")
                .flag("-gencode")
                .flag("arch=compute_70,code=sm_70")
                .flag("-gencode")
                .flag("arch=compute_75,code=sm_75")
                .flag("-gencode")
                .flag("arch=compute_80,code=sm_80")
                .flag("-gencode")
                .flag("arch=compute_86,code=sm_86")
                .flag("-O3")
                .flag("-use_fast_math")
                .compile("helix_cuda_kernels");
        }

        // Link CUDA libraries
        println!("cargo:rustc-link-search=native={}", cuda_lib);
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=curand");
        
        // Rebuild if CUDA kernels change
        println!("cargo:rerun-if-changed=src/kernels.cu");
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("cargo:warning=CUDA backend compiled without CUDA support");
    }
}
