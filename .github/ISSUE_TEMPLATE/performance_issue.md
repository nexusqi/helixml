---
name: Performance Issue
about: Report performance problems or request optimization
title: '[PERF] '
labels: ['performance', 'needs-triage']
assignees: 'blackterrordistro'

---

## ⚡ Performance Issue Description

A clear and concise description of the performance problem.

## 📊 Performance Metrics

**Current Performance:**
- Execution time: [e.g. 10 seconds]
- Memory usage: [e.g. 2GB RAM]
- CPU usage: [e.g. 100% on all cores]
- GPU usage: [e.g. 85% utilization]
- Throughput: [e.g. 100 samples/second]

**Expected Performance:**
- Execution time: [e.g. 2 seconds]
- Memory usage: [e.g. 500MB RAM]
- CPU usage: [e.g. 50% on all cores]
- GPU usage: [e.g. 95% utilization]
- Throughput: [e.g. 500 samples/second]

## 🖥️ System Information

**OS**: [e.g. Ubuntu 22.04, macOS 13.0, Windows 11]
**Rust Version**: [e.g. 1.75.0]
**HelixML Version**: [e.g. 0.1.0]
**Hardware**: [e.g. CPU: Intel i7-12700K, GPU: RTX 4090, RAM: 32GB]

## 📝 Code Example

```rust
// Code that exhibits the performance issue
use helix_ml::*;

fn main() -> Result<()> {
    // Your code here
    Ok(())
}
```

## 📈 Benchmark Results

If you have benchmark results, please include them here:

```bash
# Benchmark output
cargo bench --package your-package
```

## 🔍 Profiling Results

If you have profiling results (e.g., from `perf`, `cargo flamegraph`, etc.), please include them.

## 🎯 Use Case

Describe the specific use case where this performance issue occurs.

## 📚 Additional Context

Add any other context about the performance issue here.

## ✅ Checklist

- [ ] I have provided current and expected performance metrics
- [ ] I have included system information
- [ ] I have provided a code example
- [ ] I have included benchmark results if available
- [ ] I have described the specific use case
