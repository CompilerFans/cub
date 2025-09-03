/*
 * 单文件编译示例：DeviceSegmentedMergeSort
 * 编译命令：nvcc -I./cub -I../../../libcudacxx/include -std=c++14 -arch=sm_75 standalone_example.cu -o standalone_example
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// 只需要包含这一个头文件
#include "cub/device/device_segmented_merge_sort.cuh"

void print_array(const std::vector<int>& arr, const std::string& name) {
    std::cout << name << ": ";
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== DeviceSegmentedMergeSort 单文件编译示例 ===" << std::endl;
    
    // 测试数据：3个段的数据
    const int num_items = 10;
    const int num_segments = 3;
    
    // 输入数据: 段1[5,1,9,2], 段2[8,3], 段3[7,6,4,0] 
    std::vector<int> keys_in = {5, 1, 9, 2, 8, 3, 7, 6, 4, 0};
    std::vector<int> values_in = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> segment_offsets = {0, 4, 6, 10};
    
    std::vector<int> keys_out(num_items);
    std::vector<int> values_out(num_items);
    
    print_array(keys_in, "输入键");
    print_array(values_in, "输入值");
    
    // 分配GPU内存
    int *d_keys_in, *d_keys_out, *d_values_in, *d_values_out, *d_offsets;
    
    cudaMalloc(&d_keys_in, num_items * sizeof(int));
    cudaMalloc(&d_keys_out, num_items * sizeof(int)); 
    cudaMalloc(&d_values_in, num_items * sizeof(int));
    cudaMalloc(&d_values_out, num_items * sizeof(int));
    cudaMalloc(&d_offsets, (num_segments + 1) * sizeof(int));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_keys_in, keys_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_in, values_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, segment_offsets.data(), (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    std::cout << "\n--- 测试1: 升序排序 (SortPairs) ---" << std::endl;
    
    // 获取临时存储大小
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceSegmentedMergeSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items, num_segments,
        d_offsets, d_offsets + 1);
    
    std::cout << "临时存储需求: " << temp_storage_bytes << " 字节" << std::endl;
    
    // 分配临时存储
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // 执行排序
    cudaError_t result = cub::DeviceSegmentedMergeSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items, num_segments,
        d_offsets, d_offsets + 1);
    
    if (result == cudaSuccess) {
        // 拷贝结果回CPU
        cudaMemcpy(keys_out.data(), d_keys_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(values_out.data(), d_values_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);
        
        print_array(keys_out, "输出键(升序)");
        print_array(values_out, "输出值(升序)");
        std::cout << "排序成功！预期: [1,2,5,9] [3,8] [0,4,6,7]" << std::endl;
    } else {
        std::cout << "排序失败: " << cudaGetErrorString(result) << std::endl;
    }
    
    std::cout << "\n--- 测试2: 降序排序 (SortPairsDescending) ---" << std::endl;
    
    // 重置输入数据（因为之前的排序改变了数据）
    cudaMemcpy(d_keys_in, keys_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_in, values_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    
    result = cub::DeviceSegmentedMergeSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items, num_segments,
        d_offsets, d_offsets + 1);
    
    if (result == cudaSuccess) {
        cudaMemcpy(keys_out.data(), d_keys_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(values_out.data(), d_values_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);
        
        print_array(keys_out, "输出键(降序)");
        print_array(values_out, "输出值(降序)");
        std::cout << "排序成功！预期: [9,5,2,1] [8,3] [7,6,4,0]" << std::endl;
    } else {
        std::cout << "排序失败: " << cudaGetErrorString(result) << std::endl;
    }
    
    std::cout << "\n--- 测试3: 仅键排序 (SortKeys) ---" << std::endl;
    
    // 重置输入数据
    cudaMemcpy(d_keys_in, keys_in.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    
    result = cub::DeviceSegmentedMergeSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        num_items, num_segments,
        d_offsets, d_offsets + 1);
    
    if (result == cudaSuccess) {
        cudaMemcpy(keys_out.data(), d_keys_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);
        
        print_array(keys_out, "输出键(仅键)");
        std::cout << "仅键排序成功！" << std::endl;
    } else {
        std::cout << "仅键排序失败: " << cudaGetErrorString(result) << std::endl;
    }
    
    // 清理内存
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_offsets);
    cudaFree(d_temp_storage);
    
    std::cout << "\n=== 单文件编译示例完成 ===" << std::endl;
    return 0;
}

/*
编译说明:

1. 基本编译:
   nvcc -I./cub -I../../../libcudacxx/include -std=c++14 -arch=sm_75 standalone_example.cu -o standalone_example

2. 如果需要调试信息:
   nvcc -I./cub -I../../../libcudacxx/include -std=c++14 -arch=sm_75 -g -G standalone_example.cu -o standalone_example

3. 优化编译:
   nvcc -I./cub -I../../../libcudacxx/include -std=c++14 -arch=sm_75 -O3 standalone_example.cu -o standalone_example

4. 支持的GPU架构:
   - sm_50, sm_52, sm_53 (Maxwell)
   - sm_60, sm_61, sm_62 (Pascal) 
   - sm_70, sm_72, sm_75 (Volta/Turing)
   - sm_80, sm_86, sm_90 (Ampere/Hopper)

使用示例:
   nvcc -I./cub -I../../../libcudacxx/include -std=c++14 -arch=sm_75 standalone_example.cu -o standalone_example
   ./standalone_example

依赖项:
- CUDA Toolkit (11.0+推荐)
- CUB headers (包含在此项目中)
- libcudacxx headers (包含在此项目中)
*/