# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Notice

**This CUB repository has been archived and is now part of the unified nvidia/cccl repository. Future development should happen in the CCCL repository.**

## About CUB

CUB is a CUDA C++ template library providing reusable software components for CUDA programming:
- **Device-wide primitives**: Sort, scan, reduction, histogram (compatible with CUDA dynamic parallelism)
- **Block-wide collectives**: Block-level I/O, sort, scan, reduction, histogram  
- **Warp-wide collectives**: Warp-level scan, reduction
- **Thread/resource utilities**: PTX intrinsics, device reflection, memory allocators

## Build Commands

### Basic Build (Standalone)
CUB typically builds as part of Thrust. For standalone development:

```bash
mkdir build && cd build
cmake ..
cmake --build . -j $(nproc)
```

### Build with Thrust (Recommended)
```bash
# Clone Thrust with CUB as submodule
git clone --recursive https://github.com/NVIDIA/thrust.git
cd thrust/build
cmake -DTHRUST_INCLUDE_CUB_CMAKE=ON ..
cmake --build . -j $(nproc)
```

### Test Commands
```bash
# Run all tests
ctest

# Run specific test configuration
ctest -R "cub.*test"

# Build only tests
cmake --build . --target cub.cpp14.tests
```

### Example Commands  
```bash
# Build examples
cmake --build . --target cub.cpp14.examples

# Run specific example
./bin/cub.cpp14.example.device_radix_sort
```

### Benchmark Commands
```bash
# Enable benchmarks in cmake
cmake -DCUB_ENABLE_BENCHMARKS=ON ..

# Build all benchmarks  
cmake --build . --target cub.all.benches

# Run specific benchmark
./bin/cub.bench.radix_sort.keys.base
```

## Key CMake Options

- `CUB_ENABLE_TESTING=ON/OFF` - Build tests (default: ON)
- `CUB_ENABLE_EXAMPLES=ON/OFF` - Build examples (default: ON) 
- `CUB_ENABLE_BENCHMARKS=ON/OFF` - Build benchmarks (default: OFF)
- `CUB_ENABLE_TUNING=ON/OFF` - Build tuning suite (default: OFF)
- `CUB_ENABLE_TESTS_WITH_RDC=ON/OFF` - Enable relocatable device code for tests

## Architecture Overview

### Core Structure
```
cub/
├── agent/          # Agent-level algorithms (device-wide coordination)
├── block/          # Block-level collective primitives  
├── device/         # Device-wide algorithm entry points
├── grid/           # Grid-level utilities (barriers, queues)
├── iterator/       # Specialized iterator types
├── thread/         # Thread-level utilities  
├── warp/           # Warp-level collective primitives
└── util_*.cuh      # Cross-cutting utilities
```

### Key Components

**Device Layer** (`device/`): Entry points for device-wide algorithms like `DeviceRadixSort`, `DeviceReduce`, `DeviceScan`

**Agent Layer** (`agent/`): Coordination logic for multi-block algorithms, handles work distribution and synchronization

**Block Layer** (`block/`): Thread-block collective primitives like `BlockLoad`, `BlockStore`, `BlockReduce`, `BlockScan`  

**Warp Layer** (`warp/`): Warp-wide collectives using shuffle operations and shared memory

### Algorithm Pattern
Most CUB algorithms follow a common pattern:
1. **Device API**: Public entry point in `device/device_*.cuh`
2. **Dispatch**: Algorithm selection and kernel launch in `device/dispatch/dispatch_*.cuh`  
3. **Agent**: Kernel implementation with work distribution in `agent/agent_*.cuh`
4. **Specialization**: Architecture-specific optimizations in `*/specializations/`

### Template Specialization
CUB uses extensive template specialization for:
- Data types (primitives, custom types)
- Block/warp sizes
- Algorithm variants (different scan operators, etc.)
- Architecture targeting (SM compute capability)

### Testing Framework  
- **Legacy tests**: `test_*.cu` files using custom testing framework
- **Catch2 tests**: `catch2_test_*.cu` files using modern Catch2 framework
- **Parameterized tests**: Use `%PARAM%` comments for generating test variants
- **CDP tests**: CUDA Dynamic Parallelism testing with special RDC handling

### Memory Management
- **Temporary storage**: Algorithms use `TempStorage` pattern for shared/global memory allocation
- **Cached allocators**: `CachingDeviceAllocator` for efficient memory reuse in tests/examples

## Development Patterns

### When Adding New Algorithms
1. Create device-level API in `device/device_*.cuh`
2. Add dispatch logic in `device/dispatch/dispatch_*.cuh`
3. Implement agent in `agent/agent_*.cuh`
4. Add specializations as needed
5. Create comprehensive tests in `test/`
6. Add examples in `examples/`

### Template Parameter Conventions
- `T`: Data type
- `OffsetT`: Index/offset type (usually `int` or `ptrdiff_t`)
- `BLOCK_THREADS`: Threads per block
- `ITEMS_PER_THREAD`: Items processed per thread
- `Algorithm`: Algorithm variant enum

### Compilation Requirements
- NVCC 11.0+ or compatible compiler
- C++14 minimum (configurable up to C++17)
- CUDA compute capability 3.5+
- CMake 3.15+