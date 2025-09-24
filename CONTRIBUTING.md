# Contributing to HelixML

Thank you for your interest in contributing to HelixML! This document provides guidelines and information for contributors.

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).

## Getting Started

### Prerequisites

- Rust 1.70+ with `rustfmt` and `clippy`
- Git

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/helix-ml.git
   cd helix-ml
   ```

3. Install development dependencies:
   ```bash
   cargo install cargo-watch
   cargo install cargo-expand
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p tensor-core

# Run examples
cargo run -p simple_example
```

### Code Style

- Use `cargo fmt` to format code
- Use `cargo clippy` to check for linting issues
- Follow Rust naming conventions
- Write comprehensive documentation

### Building

```bash
# Build all crates
cargo build --workspace

# Build with optimizations
cargo build --workspace --release
```

## Contribution Guidelines

### Reporting Bugs

1. Check if the bug has already been reported
2. Use the bug report template
3. Provide minimal reproduction code
4. Include environment details (OS, Rust version, etc.)

### Suggesting Features

1. Check if the feature has already been requested
2. Use the feature request template
3. Describe the use case and benefits
4. Consider implementation complexity

### Pull Requests

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Update documentation if needed
5. Ensure all tests pass
6. Submit a pull request

### Commit Messages

Follow conventional commit format:
- `feat: add new feature`
- `fix: resolve bug in module`
- `docs: update documentation`
- `test: add test cases`
- `refactor: improve code structure`

## Project Structure

```
helix-ml/
├── crates/           # Core library crates
│   ├── tensor-core/  # Tensor operations
│   ├── backend-cpu/  # CPU backend
│   ├── autograd/     # Automatic differentiation
│   ├── nn/          # Neural network layers
│   ├── optim/       # Optimizers
│   └── ...
├── examples/        # Example applications
├── docs/           # Documentation
└── lib/            # Main library crate
```

## Architecture Guidelines

### Adding New Crates

1. Create crate in `crates/` directory
2. Add to workspace in root `Cargo.toml`
3. Add dependency to `lib/Cargo.toml`
4. Re-export in `lib/src/lib.rs`

### Tensor Operations

- Follow the trait-based design
- Implement for both CPU and future GPU backends
- Add comprehensive tests
- Document performance characteristics

### Neural Network Layers

- Implement the `Module` trait
- Support both training and inference modes
- Add gradient checkpointing support
- Include shape validation

## Testing

### Unit Tests

- Test all public APIs
- Include edge cases and error conditions
- Use descriptive test names
- Group related tests in modules

### Integration Tests

- Test crate interactions
- Verify example applications
- Test with different configurations

### Performance Tests

- Benchmark critical operations
- Compare with reference implementations
- Monitor for regressions

## Documentation

### Code Documentation

- Document all public APIs
- Include usage examples
- Explain complex algorithms
- Use rustdoc format

### README Updates

- Keep installation instructions current
- Update feature lists
- Add new examples
- Maintain accuracy

## Release Process

1. Update version numbers in `Cargo.toml` files
2. Update `CHANGELOG.md`
3. Create release notes
4. Tag the release
5. Publish to crates.io

## Getting Help

- Check existing issues and discussions
- Join our community chat (if available)
- Ask questions in GitHub Discussions
- Review existing code for patterns

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing to HelixML! 🦀
