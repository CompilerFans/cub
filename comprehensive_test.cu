#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <algorithm>

// Include CUB headers
#include "cub/device/device_segmented_merge_sort.cuh"

bool test_pairs_ascending() {
    std::cout << "Testing SortPairs (ascending)..." << std::endl;
    
    const int num_items = 12;
    const int num_segments = 3;
    
    // Test data: three segments [4,2,8,1], [7,3], [6,5,9,0,11,10]
    std::vector<int> h_keys_in = {4, 2, 8, 1, 7, 3, 6, 5, 9, 0, 11, 10};
    std::vector<int> h_values_in = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<int> h_segment_offsets = {0, 4, 6, 12};
    
    std::vector<int> h_keys_out(num_items);
    std::vector<int> h_values_out(num_items);

    // Expected results: [1,2,4,8], [3,7], [0,5,6,9,10,11]
    std::vector<int> expected_keys = {1, 2, 4, 8, 3, 7, 0, 5, 6, 9, 10, 11};

    // Device memory
    int *d_keys_in, *d_keys_out, *d_values_in, *d_values_out, *d_segment_offsets;
    
    cudaMalloc(&d_keys_in, num_items * sizeof(int));
    cudaMalloc(&d_keys_out, num_items * sizeof(int));
    cudaMalloc(&d_values_in, num_items * sizeof(int));
    cudaMalloc(&d_values_out, num_items * sizeof(int));
    cudaMalloc(&d_segment_offsets, (num_segments + 1) * sizeof(int));
    
    cudaMemcpy(d_keys_in, h_keys_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_in, h_values_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_segment_offsets, h_segment_offsets.data(), (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cudaError_t result = cub::DeviceSegmentedMergeSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items, num_segments,
        d_segment_offsets, d_segment_offsets + 1);
    
    if (result != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(result) << std::endl;
        return false;
    }
    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    result = cub::DeviceSegmentedMergeSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items, num_segments,
        d_segment_offsets, d_segment_offsets + 1);
    
    if (result != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(result) << std::endl;
        return false;
    }
    
    cudaMemcpy(h_keys_out.data(), d_keys_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values_out.data(), d_values_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < num_items; i++) {
        if (h_keys_out[i] != expected_keys[i]) {
            correct = false;
            std::cout << "  Mismatch at position " << i << ": got " << h_keys_out[i] << ", expected " << expected_keys[i] << std::endl;
        }
    }
    
    if (correct) {
        std::cout << "  PASSED" << std::endl;
    } else {
        std::cout << "  FAILED" << std::endl;
        std::cout << "  Input keys:  ";
        for (int i = 0; i < num_items; i++) std::cout << h_keys_in[i] << " ";
        std::cout << std::endl;
        std::cout << "  Output keys: ";
        for (int i = 0; i < num_items; i++) std::cout << h_keys_out[i] << " ";
        std::cout << std::endl;
        std::cout << "  Expected:    ";
        for (int i = 0; i < num_items; i++) std::cout << expected_keys[i] << " ";
        std::cout << std::endl;
    }
    
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_segment_offsets);
    cudaFree(d_temp_storage);
    
    return correct;
}

bool test_pairs_descending() {
    std::cout << "Testing SortPairsDescending (descending)..." << std::endl;
    
    const int num_items = 8;
    const int num_segments = 2;
    
    std::vector<int> h_keys_in = {4, 2, 8, 1, 7, 3, 6, 5};
    std::vector<int> h_values_in = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int> h_segment_offsets = {0, 4, 8};
    
    std::vector<int> h_keys_out(num_items);
    std::vector<int> h_values_out(num_items);

    // Expected results (descending): [8,4,2,1], [7,6,5,3]
    std::vector<int> expected_keys = {8, 4, 2, 1, 7, 6, 5, 3};

    int *d_keys_in, *d_keys_out, *d_values_in, *d_values_out, *d_segment_offsets;
    
    cudaMalloc(&d_keys_in, num_items * sizeof(int));
    cudaMalloc(&d_keys_out, num_items * sizeof(int));
    cudaMalloc(&d_values_in, num_items * sizeof(int));
    cudaMalloc(&d_values_out, num_items * sizeof(int));
    cudaMalloc(&d_segment_offsets, (num_segments + 1) * sizeof(int));
    
    cudaMemcpy(d_keys_in, h_keys_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_in, h_values_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_segment_offsets, h_segment_offsets.data(), (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cudaError_t result = cub::DeviceSegmentedMergeSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items, num_segments,
        d_segment_offsets, d_segment_offsets + 1);
    
    if (result != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(result) << std::endl;
        return false;
    }
    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    result = cub::DeviceSegmentedMergeSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items, num_segments,
        d_segment_offsets, d_segment_offsets + 1);
    
    if (result != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(result) << std::endl;
        return false;
    }
    
    cudaMemcpy(h_keys_out.data(), d_keys_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    
    bool correct = true;
    for (int i = 0; i < num_items; i++) {
        if (h_keys_out[i] != expected_keys[i]) {
            correct = false;
            std::cout << "  Mismatch at position " << i << ": got " << h_keys_out[i] << ", expected " << expected_keys[i] << std::endl;
        }
    }
    
    if (correct) {
        std::cout << "  PASSED" << std::endl;
    } else {
        std::cout << "  FAILED" << std::endl;
    }
    
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_segment_offsets);
    cudaFree(d_temp_storage);
    
    return correct;
}

bool test_keys_only() {
    std::cout << "Testing SortKeys (keys only)..." << std::endl;
    
    const int num_items = 6;
    const int num_segments = 2;
    
    std::vector<int> h_keys_in = {9, 1, 5, 8, 3, 7};
    std::vector<int> h_segment_offsets = {0, 3, 6};
    
    std::vector<int> h_keys_out(num_items);

    // Expected results: [1,5,9], [3,7,8]
    std::vector<int> expected_keys = {1, 5, 9, 3, 7, 8};

    int *d_keys_in, *d_keys_out, *d_segment_offsets;
    
    cudaMalloc(&d_keys_in, num_items * sizeof(int));
    cudaMalloc(&d_keys_out, num_items * sizeof(int));
    cudaMalloc(&d_segment_offsets, (num_segments + 1) * sizeof(int));
    
    cudaMemcpy(d_keys_in, h_keys_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_segment_offsets, h_segment_offsets.data(), (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cudaError_t result = cub::DeviceSegmentedMergeSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        num_items, num_segments,
        d_segment_offsets, d_segment_offsets + 1);
    
    if (result != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(result) << std::endl;
        return false;
    }
    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    result = cub::DeviceSegmentedMergeSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        num_items, num_segments,
        d_segment_offsets, d_segment_offsets + 1);
    
    if (result != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(result) << std::endl;
        return false;
    }
    
    cudaMemcpy(h_keys_out.data(), d_keys_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    
    bool correct = true;
    for (int i = 0; i < num_items; i++) {
        if (h_keys_out[i] != expected_keys[i]) {
            correct = false;
            std::cout << "  Mismatch at position " << i << ": got " << h_keys_out[i] << ", expected " << expected_keys[i] << std::endl;
        }
    }
    
    if (correct) {
        std::cout << "  PASSED" << std::endl;
    } else {
        std::cout << "  FAILED" << std::endl;
    }
    
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_segment_offsets);
    cudaFree(d_temp_storage);
    
    return correct;
}

int main() {
    std::cout << "=== DeviceSegmentedMergeSort Comprehensive Test ===" << std::endl;
    
    bool all_passed = true;
    
    all_passed &= test_pairs_ascending();
    all_passed &= test_pairs_descending();
    all_passed &= test_keys_only();
    
    std::cout << std::endl;
    std::cout << "=== Overall Result: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << " ===" << std::endl;
    
    return all_passed ? 0 : 1;
}