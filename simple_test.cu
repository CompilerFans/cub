#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Include CUB headers
#include "cub/device/device_segmented_merge_sort.cuh"

int main()
{
    std::cout << "Starting DeviceSegmentedMergeSort simple test..." << std::endl;

    // Simple test data
    const int num_items = 8;
    const int num_segments = 2;
    
    // Host data
    std::vector<int> h_keys_in = {4, 2, 8, 1, 7, 3, 6, 5};  // Two segments: [4,2,8,1] and [7,3,6,5]
    std::vector<int> h_values_in = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int> h_segment_offsets = {0, 4, 8};  // First segment: 0-3, Second segment: 4-7
    
    std::vector<int> h_keys_out(num_items);
    std::vector<int> h_values_out(num_items);

    // Device pointers
    int *d_keys_in, *d_keys_out, *d_values_in, *d_values_out, *d_segment_offsets;
    
    // Allocate device memory
    cudaMalloc(&d_keys_in, num_items * sizeof(int));
    cudaMalloc(&d_keys_out, num_items * sizeof(int));
    cudaMalloc(&d_values_in, num_items * sizeof(int));
    cudaMalloc(&d_values_out, num_items * sizeof(int));
    cudaMalloc(&d_segment_offsets, (num_segments + 1) * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_keys_in, h_keys_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_in, h_values_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_segment_offsets, h_segment_offsets.data(), (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    // Determine temp storage size
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cudaError_t result = cub::DeviceSegmentedMergeSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items, num_segments,
        d_segment_offsets, d_segment_offsets + 1);
    
    std::cout << "Temp storage size: " << temp_storage_bytes << " bytes" << std::endl;
    
    if (result != cudaSuccess) {
        std::cout << "Error determining temp storage: " << cudaGetErrorString(result) << std::endl;
        return 1;
    }
    
    // Allocate temp storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Run the sort
    result = cub::DeviceSegmentedMergeSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items, num_segments,
        d_segment_offsets, d_segment_offsets + 1);
    
    if (result != cudaSuccess) {
        std::cout << "Error running sort: " << cudaGetErrorString(result) << std::endl;
        return 1;
    }
    
    // Copy results back
    cudaMemcpy(h_keys_out.data(), d_keys_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values_out.data(), d_values_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "Input keys:    ";
    for (int i = 0; i < num_items; i++) std::cout << h_keys_in[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Output keys:   ";
    for (int i = 0; i < num_items; i++) std::cout << h_keys_out[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Input values:  ";
    for (int i = 0; i < num_items; i++) std::cout << h_values_in[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Output values: ";
    for (int i = 0; i < num_items; i++) std::cout << h_values_out[i] << " ";
    std::cout << std::endl;
    
    // Verify correctness (first segment should be [1,2,4,8], second segment should be [3,5,6,7])
    std::vector<int> expected_keys = {1, 2, 4, 8, 3, 5, 6, 7};
    bool correct = true;
    for (int i = 0; i < num_items; i++) {
        if (h_keys_out[i] != expected_keys[i]) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Test result: " << (correct ? "PASSED" : "FAILED") << std::endl;
    
    // Cleanup
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_segment_offsets);
    cudaFree(d_temp_storage);
    
    return correct ? 0 : 1;
}