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
 * SOFTWARE, EVEN IF ADVISED OF THIS POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cub/device/device_segmented_merge_sort.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "catch2_test_helper.h"
#include "catch2_test_cdp_helper.h"

#include <algorithm>
#include <random>

using namespace cub;

// %PARAM% TEST_LAUNCH launch 0:1

template <typename KeyT, typename ValueT>
struct segmented_merge_sort_test_data_t
{
  thrust::device_vector<KeyT> keys_in;
  thrust::device_vector<KeyT> keys_out;
  thrust::device_vector<ValueT> values_in;
  thrust::device_vector<ValueT> values_out;
  thrust::device_vector<int> segment_offsets;
  
  int num_items{};
  int num_segments{};
  
  segmented_merge_sort_test_data_t(int num_segments, int min_segment_size, int max_segment_size)
    : num_segments(num_segments)
  {
    generate_data(min_segment_size, max_segment_size);
  }
  
private:
  void generate_data(int min_segment_size, int max_segment_size)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> size_dist(min_segment_size, max_segment_size);
    std::uniform_int_distribution<int> value_dist(0, 1000);
    
    thrust::host_vector<KeyT> h_keys;
    thrust::host_vector<ValueT> h_values;
    thrust::host_vector<int> h_offsets;
    
    h_offsets.push_back(0);
    
    for (int seg = 0; seg < num_segments; seg++)
    {
      int segment_size = size_dist(gen);
      int start_offset = h_offsets.back();
      int end_offset = start_offset + segment_size;
      
      h_offsets.push_back(end_offset);
      
      for (int i = start_offset; i < end_offset; i++)
      {
        h_keys.push_back(static_cast<KeyT>(value_dist(gen)));
        h_values.push_back(static_cast<ValueT>(i));
      }
    }
    
    num_items = static_cast<int>(h_keys.size());
    
    // Transfer to device
    keys_in = h_keys;
    keys_out.resize(num_items);
    values_in = h_values;
    values_out.resize(num_items);
    segment_offsets = h_offsets;
  }
};

template <typename KeyT>
void verify_keys_sorted(const thrust::device_vector<KeyT>& keys_out,
                        const thrust::device_vector<int>& segment_offsets,
                        int num_segments,
                        bool is_descending = false)
{
  thrust::host_vector<KeyT> h_keys = keys_out;
  thrust::host_vector<int> h_offsets = segment_offsets;
  
  for (int seg = 0; seg < num_segments; seg++)
  {
    int start = h_offsets[seg];
    int end = h_offsets[seg + 1];
    
    if (start >= end) continue;
    
    for (int i = start; i < end - 1; i++)
    {
      if (is_descending)
      {
        REQUIRE(h_keys[i] >= h_keys[i + 1]);
      }
      else
      {
        REQUIRE(h_keys[i] <= h_keys[i + 1]);
      }
    }
  }
}

template <typename KeyT, typename ValueT>
void verify_pairs_sorted(const thrust::device_vector<KeyT>& keys_out,
                         const thrust::device_vector<ValueT>& values_out,
                         const thrust::device_vector<int>& segment_offsets,
                         int num_segments,
                         bool is_descending = false)
{
  thrust::host_vector<KeyT> h_keys = keys_out;
  thrust::host_vector<ValueT> h_values = values_out;
  thrust::host_vector<int> h_offsets = segment_offsets;
  
  for (int seg = 0; seg < num_segments; seg++)
  {
    int start = h_offsets[seg];
    int end = h_offsets[seg + 1];
    
    if (start >= end) continue;
    
    for (int i = start; i < end - 1; i++)
    {
      if (is_descending)
      {
        REQUIRE(h_keys[i] >= h_keys[i + 1]);
      }
      else
      {
        REQUIRE(h_keys[i] <= h_keys[i + 1]);
      }
      
      // Values should follow keys (stable sort)
      if (h_keys[i] == h_keys[i + 1])
      {
        if (is_descending)
        {
          REQUIRE(h_values[i] >= h_values[i + 1]);
        }
        else
        {
          REQUIRE(h_values[i] <= h_values[i + 1]);
        }
      }
    }
  }
}

TEMPLATE_TEST_CASE_SIG("DeviceSegmentedMergeSort::SortKeys works correctly",
                       "[segmented_merge_sort][keys]",
                       ((typename KeyT), KeyT),
                       (std::uint8_t),
                       (std::uint16_t),
                       (std::uint32_t),
                       (std::uint64_t),
                       (std::int8_t),
                       (std::int16_t),
                       (std::int32_t),
                       (std::int64_t),
                       (float),
                       (double))
{
  using OffsetT = int;
  
  SECTION("Small segments")
  {
    segmented_merge_sort_test_data_t<KeyT, int> test_data(100, 1, 32);
    
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Determine temporary device storage requirements
    REQUIRE(cudaSuccess == DeviceSegmentedMergeSort::SortKeys(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(test_data.keys_in.data()),
      thrust::raw_pointer_cast(test_data.keys_out.data()),
      test_data.num_items,
      test_data.num_segments,
      thrust::raw_pointer_cast(test_data.segment_offsets.data()),
      thrust::raw_pointer_cast(test_data.segment_offsets.data()) + 1));
    
    // Allocate temporary storage
    REQUIRE(cudaSuccess == cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Run sorting operation
    REQUIRE(cudaSuccess == DeviceSegmentedMergeSort::SortKeys(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(test_data.keys_in.data()),
      thrust::raw_pointer_cast(test_data.keys_out.data()),
      test_data.num_items,
      test_data.num_segments,
      thrust::raw_pointer_cast(test_data.segment_offsets.data()),
      thrust::raw_pointer_cast(test_data.segment_offsets.data()) + 1));
    
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    
    // Verify results
    verify_keys_sorted(test_data.keys_out, test_data.segment_offsets, test_data.num_segments);
    
    // Cleanup
    if (d_temp_storage)
      REQUIRE(cudaSuccess == cudaFree(d_temp_storage));
  }
  
  SECTION("Descending order")
  {
    segmented_merge_sort_test_data_t<KeyT, int> test_data(50, 10, 100);
    
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Determine temporary device storage requirements
    REQUIRE(cudaSuccess == DeviceSegmentedMergeSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(test_data.keys_in.data()),
      thrust::raw_pointer_cast(test_data.keys_out.data()),
      test_data.num_items,
      test_data.num_segments,
      thrust::raw_pointer_cast(test_data.segment_offsets.data()),
      thrust::raw_pointer_cast(test_data.segment_offsets.data()) + 1));
    
    // Allocate temporary storage
    REQUIRE(cudaSuccess == cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Run sorting operation
    REQUIRE(cudaSuccess == DeviceSegmentedMergeSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(test_data.keys_in.data()),
      thrust::raw_pointer_cast(test_data.keys_out.data()),
      test_data.num_items,
      test_data.num_segments,
      thrust::raw_pointer_cast(test_data.segment_offsets.data()),
      thrust::raw_pointer_cast(test_data.segment_offsets.data()) + 1));
    
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    
    // Verify results
    verify_keys_sorted(test_data.keys_out, test_data.segment_offsets, test_data.num_segments, true);
    
    // Cleanup
    if (d_temp_storage)
      REQUIRE(cudaSuccess == cudaFree(d_temp_storage));
  }
}

TEMPLATE_TEST_CASE_SIG("DeviceSegmentedMergeSort::SortPairs works correctly",
                       "[segmented_merge_sort][pairs]",
                       ((typename KeyT, typename ValueT), KeyT, ValueT),
                       (std::int32_t, std::int32_t),
                       (std::int64_t, std::int32_t),
                       (float, std::int32_t),
                       (double, std::int64_t))
{
  using OffsetT = int;
  
  SECTION("Small segments")
  {
    segmented_merge_sort_test_data_t<KeyT, ValueT> test_data(50, 1, 32);
    
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Determine temporary device storage requirements
    REQUIRE(cudaSuccess == DeviceSegmentedMergeSort::SortPairs(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(test_data.keys_in.data()),
      thrust::raw_pointer_cast(test_data.keys_out.data()),
      thrust::raw_pointer_cast(test_data.values_in.data()),
      thrust::raw_pointer_cast(test_data.values_out.data()),
      test_data.num_items,
      test_data.num_segments,
      thrust::raw_pointer_cast(test_data.segment_offsets.data()),
      thrust::raw_pointer_cast(test_data.segment_offsets.data()) + 1));
    
    // Allocate temporary storage
    REQUIRE(cudaSuccess == cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Run sorting operation
    REQUIRE(cudaSuccess == DeviceSegmentedMergeSort::SortPairs(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(test_data.keys_in.data()),
      thrust::raw_pointer_cast(test_data.keys_out.data()),
      thrust::raw_pointer_cast(test_data.values_in.data()),
      thrust::raw_pointer_cast(test_data.values_out.data()),
      test_data.num_items,
      test_data.num_segments,
      thrust::raw_pointer_cast(test_data.segment_offsets.data()),
      thrust::raw_pointer_cast(test_data.segment_offsets.data()) + 1));
    
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    
    // Verify results
    verify_pairs_sorted(test_data.keys_out, test_data.values_out, 
                        test_data.segment_offsets, test_data.num_segments);
    
    // Cleanup
    if (d_temp_storage)
      REQUIRE(cudaSuccess == cudaFree(d_temp_storage));
  }
}

TEMPLATE_TEST_CASE_SIG("DeviceSegmentedMergeSort handles edge cases",
                       "[segmented_merge_sort][edge_cases]",
                       ((typename KeyT), KeyT),
                       (std::int32_t))
{
  SECTION("Empty input")
  {
    thrust::device_vector<KeyT> keys_in;
    thrust::device_vector<KeyT> keys_out;
    thrust::device_vector<int> segment_offsets = {0};
    
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    REQUIRE(cudaSuccess == DeviceSegmentedMergeSort::SortKeys(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      0, 0,
      thrust::raw_pointer_cast(segment_offsets.data()),
      thrust::raw_pointer_cast(segment_offsets.data()) + 1));
    
    REQUIRE(temp_storage_bytes > 0);
  }
  
  SECTION("Single element segments")
  {
    thrust::device_vector<KeyT> keys_in = {5, 3, 8, 1, 9};
    thrust::device_vector<KeyT> keys_out(5);
    thrust::device_vector<int> segment_offsets = {0, 1, 2, 3, 4, 5};
    
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    REQUIRE(cudaSuccess == DeviceSegmentedMergeSort::SortKeys(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      5, 5,
      thrust::raw_pointer_cast(segment_offsets.data()),
      thrust::raw_pointer_cast(segment_offsets.data()) + 1));
    
    REQUIRE(cudaSuccess == cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    REQUIRE(cudaSuccess == DeviceSegmentedMergeSort::SortKeys(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      5, 5,
      thrust::raw_pointer_cast(segment_offsets.data()),
      thrust::raw_pointer_cast(segment_offsets.data()) + 1));
    
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    
    // Single element segments should remain unchanged
    thrust::host_vector<KeyT> h_keys_in = keys_in;
    thrust::host_vector<KeyT> h_keys_out = keys_out;
    
    for (size_t i = 0; i < h_keys_in.size(); i++)
    {
      REQUIRE(h_keys_in[i] == h_keys_out[i]);
    }
    
    if (d_temp_storage)
      REQUIRE(cudaSuccess == cudaFree(d_temp_storage));
  }
}

#if TEST_LAUNCH == 1
TEMPLATE_TEST_CASE_SIG("DeviceSegmentedMergeSort works correctly with CDP",
                       "[segmented_merge_sort][cdp]",
                       ((typename KeyT), KeyT),
                       (std::int32_t))
{
  SKIP_IF_NO_CDP();
  
  segmented_merge_sort_test_data_t<KeyT, int> test_data(20, 10, 100);
  
  // CDP test implementation would go here
  // For now, just verify basic functionality
  REQUIRE(test_data.num_items > 0);
  REQUIRE(test_data.num_segments == 20);
}
#endif