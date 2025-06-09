# rs_jaccard - Ultra-Fast Jaccard Distance Calculator

A high-performance Rust implementation for calculating Jaccard distances between k-mer sets, designed as a fast alternative to GAMBIT's distance calculations. This tool provides parallel processing capabilities and multiple output formats for genomic sequence analysis.

## Features

- **Ultra-fast parallel processing** using Rayon for multi-threaded computations
- **Multiple calculation modes**: query vs references, full distance matrices, LSH-based similarity search
- **GAMBIT signature file support** (.gs format) with HDF5 backend
- **Memory-efficient streaming** for large matrices
- **Multiple matrix computation methods**: upper triangle, row-wise, blocked, streaming
- **MinHash LSH** for fast approximate similarity search
- **Flexible input/output formats**: CSV, binary, JSON
- **Progress tracking** with ETA estimates for long-running computations

## Installation

### Prerequisites

- Rust 1.70+ 
- HDF5 library (for GAMBIT signature support)

### Build from source

```bash
# Clone and build
git clone <repository>
cd rs_jaccard
cargo build --release

# The binary will be available at target/release/jaccard
```

### Install HDF5 (if needed)

**macOS:**
```bash
brew install hdf5
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libhdf5-dev
```

**CentOS/RHEL:**
```bash
sudo yum install hdf5-devel
```

## Usage

### Basic Commands

The tool provides several subcommands for different use cases:

#### 1. Query vs References
Calculate distances between a single query and multiple reference sets:

```bash
jaccard query \
  --query query_kmers.txt \
  --reference ref_kmers.txt \
  --bounds ref_bounds.txt \
  --output distances.txt \
  --threads 8
```

#### 2. Query with GAMBIT Signatures
Use GAMBIT signature files directly:

```bash
jaccard query-sig \
  --query-sig query.gs \
  --query-idx 0 \
  --ref-sig references.gs \
  --output distances.txt \
  --threads 8
```

#### 3. Full Distance Matrix
Compute all-vs-all distance matrix:

```bash
jaccard matrix-sig \
  --signatures signatures.gs \
  --output matrix.csv \
  --method rowwise \
  --threads 8
```

#### 4. Matrix with Subset
Compute matrix for only the first N signatures:

```bash
jaccard matrix-sig \
  --signatures signatures.gs \
  --output matrix.csv \
  --first-n 1000 \
  --method blocked \
  --threads 8
```

#### 5. Streaming Matrix (Memory Efficient)
For very large datasets, use streaming output:

```bash
jaccard matrix-sig \
  --signatures signatures.gs \
  --output matrix.csv \
  --method rowwise-stream \
  --threads 8
```

#### 6. LSH Similarity Search
Find similar pairs using Locality-Sensitive Hashing:

```bash
jaccard lsh \
  --signatures signatures.gs \
  --output similar_pairs.csv \
  --threshold 0.8 \
  --num-hashes 128 \
  --threads 8
```

### Matrix Computation Methods

- **`upper`**: Compute only upper triangle (fastest for symmetric matrices)
- **`rowwise`**: Process rows in parallel with progress tracking
- **`blocked`**: Use blocked algorithm for better cache performance
- **`rowwise-stream`**: Stream output to disk (most memory-efficient)

### File Formats

#### Input Formats
- **K-mer coordinates**: Plain text files with one integer per line
- **GAMBIT signatures**: HDF5 format (.gs files)
- **Bounds files**: Text files defining set boundaries

#### Output Formats
- **CSV**: Human-readable with row/column labels
- **Binary**: Compact binary format for large matrices
- **JSON**: Structured format for programmatic access

## Performance Optimizations

- **Compile-time optimizations**: LTO, single codegen unit, panic=abort
- **Parallel processing**: Automatic work distribution across CPU cores
- **Memory efficiency**: Streaming algorithms for large datasets
- **SIMD operations**: Optimized intersection calculations
- **Cache-friendly**: Blocked algorithms for better memory access patterns

## Benchmarking

### Benchmark Script

Compare performance against GAMBIT's built-in distance calculation:

```bash
#!/bin/bash
# benchmark.sh - Compare rs_jaccard vs GAMBIT performance

# Ensure hyperfine is installed
if ! command -v hyperfine &> /dev/null; then
    echo "Installing hyperfine..."
    # macOS
    if command -v brew &> /dev/null; then
        brew install hyperfine
    # Ubuntu/Debian
    elif command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install hyperfine
    else
        echo "Please install hyperfine manually"
        exit 1
    fi
fi

# Run benchmark
hyperfine \
  --prepare 'rm -f test-thanh' \
  'gambit dist --qs ref-signatures.gs --square -o gmabit_dist -c 8' \
  --prepare 'rm -f rust' \
  'jaccard matrix-sig --signatures ref-signatures.gs --output rs_jaccard' \
  --warmup 2 \
  --runs 15 \
  --export-markdown hyperfine-results.md

echo "Benchmark results saved to hyperfine-results.md"
```

| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `gambit dist --qs ref-signatures.gs --square -o gmabit_dist -c 8` | 775.2 ± 38.7 | 759.5 | 914.9 | 32.78 ± 2.09 |
| `jaccard matrix-sig --signatures ref-signatures.gs --output rs_jaccard` | 23.7 ± 0.9 | 22.8 | 26.0 | 1.00 |

### Expected Performance

Typical performance improvements over GAMBIT:
- **2-5x faster** for distance matrix calculations
- **10-50x faster** for LSH similarity search
- **Significantly lower memory usage** with streaming methods
- **Linear scaling** with number of CPU cores

### Memory Usage

For N signatures with M k-mers each:
- **Memory required**: ~4 * N * M bytes for coordinate storage
- **Streaming mode**: Constant memory usage regardless of matrix size
- **Peak memory**: 2-3x less than equivalent Python implementations

## File Format Details

### K-mer Coordinate Files
```
123456
234567
345678
...
```

### Bounds Files
```
0
1000
2500
4000
...
```

### GAMBIT Signature Files
HDF5 format with:
- `/values`: K-mer coordinates
- `/bounds`: Set boundaries  
- `/ids`: Sample identifiers

## Development

### Project Structure
```
rs_jaccard/
├── src/
│   ├── main.rs          # CLI interface and command handling
│   ├── jaccard.rs       # Core Jaccard distance algorithms
│   └── signatures.rs    # GAMBIT signature file handling
├── Cargo.toml           # Project configuration
└── README.md           # This file
```

### Running Tests
```bash
cargo test
```

### Building Optimized Release
```bash
cargo build --release
```

### Profiling
```bash
# Install cargo-flamegraph
cargo install flamegraph

# Profile matrix calculation
cargo flamegraph --bin jaccard -- matrix-sig --signatures test.gs --output /dev/null
```

## Troubleshooting

### Common Issues

1. **HDF5 linking errors**
   ```bash
   export HDF5_DIR=/path/to/hdf5
   cargo build --release
   ```

2. **Out of memory for large matrices**
   - Use `--method rowwise-stream` for streaming output
   - Reduce `--first-n` parameter
   - Consider using LSH for approximate results

3. **Slow performance**
   - Ensure release build: `cargo build --release`
   - Use appropriate number of threads: `--threads $(nproc)`
   - For very large datasets, use blocked method

### Performance Tips

1. **Use appropriate matrix method**:
   - Small matrices (< 1000): `upper` or `rowwise`
   - Medium matrices (1000-10000): `blocked`
   - Large matrices (> 10000): `rowwise-stream`

2. **Optimize thread count**:
   - Usually best to use all CPU cores
   - For I/O bound operations, may benefit from fewer threads

3. **Memory considerations**:
   - Monitor memory usage with large datasets
   - Use streaming methods when memory is limited

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

- Built with [Rayon](https://github.com/rayon-rs/rayon) for parallel processing
- Uses [HDF5](https://www.hdfgroup.org/) for GAMBIT signature support
- Inspired by [GAMBIT](https://github.com/jlumpe/gambit) genomic analysis toolkit 