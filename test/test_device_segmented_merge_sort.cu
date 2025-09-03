/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_segmented_merge_sort.cuh>
#include <cub/util_allocator.cuh>
#include <test_util.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <random>

// %PARAM% TEST_KEY_T kt 0:1:2:3
// %PARAM% TEST_VALUE_T vt 0:1:2:3

using namespace cub;

//---------------------------------------------------------------------
// Globals
//---------------------------------------------------------------------
CachingDeviceAllocator g_allocator(true);
bool g_verbose = false;

constexpr static int MAX_ITERATIONS = 3;

/**
 * Key type parameterization
 */
#if TEST_KEY_T == 0
  using KeyT = std::int8_t;
#elif TEST_KEY_T == 1
  using KeyT = std::int16_t;
#elif TEST_KEY_T == 2
  using KeyT = std::int32_t;
#elif TEST_KEY_T == 3
  using KeyT = std::int64_t;
#endif

/**
 * Value type parameterization
 */
#if TEST_VALUE_T == 0
  using ValueT = std::int8_t;
#elif TEST_VALUE_T == 1
  using ValueT = std::int16_t;
#elif TEST_VALUE_T == 2
  using ValueT = std::int32_t;
#elif TEST_VALUE_T == 3
  using ValueT = std::int64_t;
#endif

/**
 * Test configuration structure
 */
struct TestConfig
{
  int num_segments;
  int min_segment_size;
  int max_segment_size;
  bool is_descending;
  bool keys_only;
  
  TestConfig(int num_segments, 
             int min_segment_size, 
             int max_segment_size, 
             bool is_descending = false, 
             bool keys_only = false)
    : num_segments(num_segments)
    , min_segment_size(min_segment_size)
    , max_segment_size(max_segment_size)
    , is_descending(is_descending)
    , keys_only(keys_only)
  {}
};

/**
 * Generate test data with segments of varying sizes
 */
template <typename KeyType, typename ValueType>
void GenerateSegmentedData(thrust::host_vector<KeyType> &h_keys,
                           thrust::host_vector<ValueType> &h_values,
                           thrust::host_vector<int> &h_segment_offsets,
                           const TestConfig &config)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> size_dist(config.min_segment_size, config.max_segment_size);
  
  h_segment_offsets.clear();
  h_keys.clear();
  h_values.clear();
  
  h_segment_offsets.push_back(0);
  
  for (int seg = 0; seg < config.num_segments; seg++)
  {
    int segment_size = size_dist(gen);
    int start_offset = h_segment_offsets.back();
    int end_offset = start_offset + segment_size;
    
    h_segment_offsets.push_back(end_offset);
    
    // Generate random keys for this segment
    for (int i = start_offset; i < end_offset; i++)
    {
      h_keys.push_back(static_cast<KeyType>(gen() % 1000));
      if (!config.keys_only)
      {
        h_values.push_back(static_cast<ValueType>(i));
      }
    }
  }
}

/**
 * Reference sort using std::sort for validation
 */
template <typename KeyType, typename ValueType>
void ReferenceSort(thrust::host_vector<KeyType> &h_keys_ref,
                   thrust::host_vector<ValueType> &h_values_ref,
                   const thrust::host_vector<int> &h_segment_offsets,
                   const TestConfig &config)
{
  for (int seg = 0; seg < config.num_segments; seg++)
  {
    int start = h_segment_offsets[seg];
    int end = h_segment_offsets[seg + 1];
    
    if (start >= end) continue;
    
    if (config.keys_only)
    {
      if (config.is_descending)
      {
        std::sort(h_keys_ref.begin() + start, h_keys_ref.begin() + end, std::greater<KeyType>());
      }
      else
      {
        std::sort(h_keys_ref.begin() + start, h_keys_ref.begin() + end, std::less<KeyType>());
      }
    }
    else
    {
      // Create pairs for sorting with values
      std::vector<std::pair<KeyType, ValueType>> pairs;
      for (int i = start; i < end; i++)
      {
        pairs.emplace_back(h_keys_ref[i], h_values_ref[i]);
      }
      
      if (config.is_descending)
      {
        std::sort(pairs.begin(), pairs.end(), 
                  [](const auto &a, const auto &b) { return a.first > b.first; });
      }
      else
      {
        std::sort(pairs.begin(), pairs.end(), 
                  [](const auto &a, const auto &b) { return a.first < b.first; });
      }
      
      // Copy back sorted data
      for (int i = 0; i < static_cast<int>(pairs.size()); i++)
      {
        h_keys_ref[start + i] = pairs[i].first;
        h_values_ref[start + i] = pairs[i].second;
      }
    }
  }
}

/**
 * Test segmented merge sort keys only
 */
template <typename KeyType>
void TestSegmentedMergeSortKeys(const TestConfig &config)
{
  thrust::host_vector<KeyType> h_keys, h_keys_ref;
  thrust::host_vector<int> h_values_dummy;  // Not used for keys-only
  thrust::host_vector<int> h_segment_offsets;
  
  // Generate test data
  GenerateSegmentedData(h_keys, h_values_dummy, h_segment_offsets, config);
  h_keys_ref = h_keys;
  
  // Reference sort
  ReferenceSort(h_keys_ref, h_values_dummy, h_segment_offsets, config);
  
  // Prepare device data
  thrust::device_vector<KeyType> d_keys_in = h_keys;
  thrust::device_vector<KeyType> d_keys_out(h_keys.size());
  thrust::device_vector<int> d_segment_offsets = h_segment_offsets;
  
  // Allocate temporary storage
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  
  if (config.is_descending)
  {
    CubDebugExit(DeviceSegmentedMergeSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      static_cast<int>(h_keys.size()),
      config.num_segments,
      thrust::raw_pointer_cast(d_segment_offsets.data()),
      thrust::raw_pointer_cast(d_segment_offsets.data()) + 1));
  }
  else
  {
    CubDebugExit(DeviceSegmentedMergeSort::SortKeys(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      static_cast<int>(h_keys.size()),
      config.num_segments,
      thrust::raw_pointer_cast(d_segment_offsets.data()),
      thrust::raw_pointer_cast(d_segment_offsets.data()) + 1));
  }
  
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
  
  // Run the sort
  if (config.is_descending)
  {
    CubDebugExit(DeviceSegmentedMergeSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      static_cast<int>(h_keys.size()),
      config.num_segments,
      thrust::raw_pointer_cast(d_segment_offsets.data()),
      thrust::raw_pointer_cast(d_segment_offsets.data()) + 1));
  }
  else
  {
    CubDebugExit(DeviceSegmentedMergeSort::SortKeys(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      static_cast<int>(h_keys.size()),
      config.num_segments,
      thrust::raw_pointer_cast(d_segment_offsets.data()),
      thrust::raw_pointer_cast(d_segment_offsets.data()) + 1));
  }
  
  // Copy results back to host
  thrust::host_vector<KeyType> h_keys_out = d_keys_out;
  
  // Verify results
  AssertEquals(h_keys_ref.size(), h_keys_out.size());
  for (size_t i = 0; i < h_keys_ref.size(); i++)
  {
    AssertEquals(h_keys_ref[i], h_keys_out[i]);
  }
  
  // Cleanup
  if (d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

/**
 * Test segmented merge sort key-value pairs
 */
template <typename KeyType, typename ValueType>
void TestSegmentedMergeSortPairs(const TestConfig &config)
{
  thrust::host_vector<KeyType> h_keys, h_keys_ref;
  thrust::host_vector<ValueType> h_values, h_values_ref;
  thrust::host_vector<int> h_segment_offsets;
  
  // Generate test data
  TestConfig config_pairs = config;
  config_pairs.keys_only = false;
  GenerateSegmentedData(h_keys, h_values, h_segment_offsets, config_pairs);
  h_keys_ref = h_keys;
  h_values_ref = h_values;
  
  // Reference sort
  ReferenceSort(h_keys_ref, h_values_ref, h_segment_offsets, config_pairs);
  
  // Prepare device data
  thrust::device_vector<KeyType> d_keys_in = h_keys;
  thrust::device_vector<KeyType> d_keys_out(h_keys.size());
  thrust::device_vector<ValueType> d_values_in = h_values;
  thrust::device_vector<ValueType> d_values_out(h_values.size());
  thrust::device_vector<int> d_segment_offsets = h_segment_offsets;
  
  // Allocate temporary storage
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  
  if (config.is_descending)
  {
    CubDebugExit(DeviceSegmentedMergeSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      static_cast<int>(h_keys.size()),
      config.num_segments,
      thrust::raw_pointer_cast(d_segment_offsets.data()),
      thrust::raw_pointer_cast(d_segment_offsets.data()) + 1));
  }
  else
  {
    CubDebugExit(DeviceSegmentedMergeSort::SortPairs(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      static_cast<int>(h_keys.size()),
      config.num_segments,
      thrust::raw_pointer_cast(d_segment_offsets.data()),
      thrust::raw_pointer_cast(d_segment_offsets.data()) + 1));
  }
  
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
  
  // Run the sort
  if (config.is_descending)
  {
    CubDebugExit(DeviceSegmentedMergeSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      static_cast<int>(h_keys.size()),
      config.num_segments,
      thrust::raw_pointer_cast(d_segment_offsets.data()),
      thrust::raw_pointer_cast(d_segment_offsets.data()) + 1));
  }
  else
  {
    CubDebugExit(DeviceSegmentedMergeSort::SortPairs(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      static_cast<int>(h_keys.size()),
      config.num_segments,
      thrust::raw_pointer_cast(d_segment_offsets.data()),
      thrust::raw_pointer_cast(d_segment_offsets.data()) + 1));
  }
  
  // Copy results back to host
  thrust::host_vector<KeyType> h_keys_out = d_keys_out;
  thrust::host_vector<ValueType> h_values_out = d_values_out;
  
  // Verify results
  AssertEquals(h_keys_ref.size(), h_keys_out.size());
  AssertEquals(h_values_ref.size(), h_values_out.size());
  
  for (size_t i = 0; i < h_keys_ref.size(); i++)
  {
    AssertEquals(h_keys_ref[i], h_keys_out[i]);
    AssertEquals(h_values_ref[i], h_values_out[i]);
  }
  
  // Cleanup
  if (d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

/**
 * Main test function
 */
void TestSegmentedMergeSort()
{
  // Test configurations
  std::vector<TestConfig> test_configs = {
    // Small segments
    TestConfig(100, 1, 32, false, true),    // Keys only, ascending
    TestConfig(100, 1, 32, true, true),     // Keys only, descending
    TestConfig(50, 1, 32, false, false),    // Pairs, ascending
    TestConfig(50, 1, 32, true, false),     // Pairs, descending
    
    // Medium segments
    TestConfig(20, 32, 512, false, true),   // Keys only, ascending
    TestConfig(20, 32, 512, true, true),    // Keys only, descending
    TestConfig(10, 32, 512, false, false),  // Pairs, ascending
    TestConfig(10, 32, 512, true, false),   // Pairs, descending
    
    // Large segments
    TestConfig(5, 512, 2048, false, true),  // Keys only, ascending
    TestConfig(5, 512, 2048, true, true),   // Keys only, descending
    TestConfig(3, 512, 2048, false, false), // Pairs, ascending
    TestConfig(3, 512, 2048, true, false),  // Pairs, descending
    
    // Edge cases
    TestConfig(1000, 0, 1, false, true),    // Many empty/single-element segments
    TestConfig(1, 10000, 10000, false, true), // Single large segment
  };
  
  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    for (const auto &config : test_configs)
    {
      if (config.keys_only)
      {
        TestSegmentedMergeSortKeys<KeyT>(config);
      }
      else
      {
        TestSegmentedMergeSortPairs<KeyT, ValueT>(config);
      }
    }
  }
}

/**
 * Main function
 */
int main(int argc, char** argv)
{
  // Initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");

  // Print device info
  int device_count;
  CubDebugExit(cudaGetDeviceCount(&device_count));
  printf("Device count: %d\n", device_count);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // Run tests
  TestSegmentedMergeSort();

  return 0;
}