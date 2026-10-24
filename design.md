# Llama Virtual Memory Manager - Design Document

## 1. Introduction
The Llama Virtual Memory Manager (llama.vm) is designed to handle large language models that exceed available RAM and VRAM capacity by implementing a demand paging system with LRU caching and swap features.

## 2. Architecture
### 2.1 Demand Paging
- Implement a page table to map virtual addresses to physical addresses or disk locations
- Use lazy loading to load pages from disk only when accessed
- Handle page faults by loading required data from disk into RAM

### 2.2 LRU Caching
- Maintain an LRU cache in RAM for frequently accessed pages
- Evict least recently used pages when new pages need to be loaded
- Track access patterns to optimize cache performance

### 2.3 Swap Features
- Implement a swapping mechanism to move infrequently accessed pages from RAM back to disk
- Use a separate swap file or partition dedicated to llama.vm operations
- Ensure efficient handling of swap-in and swap-out operations

## 3. Hardware Support
### 3.1 ROCm (AMD)
- Optimize memory management for AMD GPUs using ROCm platform
- Leverage ROCm-specific features for improved performance

### 3.2 CUDA (NVIDIA) 
- Provide support for NVIDIA GPUs using CUDA platform
- Ensure compatibility with existing CUDA-based workflows

### 3.3 MLX
- MLX support (in progress - see NEXT_STEPS.md)

## 4. Testing and Validation
- Use models of different sizes stored in `~/.cache/huggingface/hub` for testing
- Benchmark performance with various model architectures
- Validate correct handling of page faults, caching, and swapping operations