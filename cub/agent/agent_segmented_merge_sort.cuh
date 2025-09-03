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

#pragma once

#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/config.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_sort.cuh>

CUB_NAMESPACE_BEGIN

/**
 * @brief Agent for segmented merge sort operations
 */
template <bool IS_DESCENDING,
          typename PolicyT,
          typename KeyT,
          typename ValueT,
          typename OffsetT>
struct AgentSegmentedMergeSort
{
  //---------------------------------------------------------------------
  // Constants and type definitions
  //---------------------------------------------------------------------

  static constexpr int BLOCK_THREADS    = PolicyT::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = PolicyT::ITEMS_PER_THREAD;
  static constexpr bool KEYS_ONLY       = std::is_same<ValueT, NullType>::value;
  static constexpr int TILE_SIZE        = BLOCK_THREADS * ITEMS_PER_THREAD;

  using BlockMergeSortT = BlockMergeSort<KeyT, BLOCK_THREADS, ITEMS_PER_THREAD, ValueT>;
  using BlockLoadKeysT  = BlockLoad<KeyT, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE>;
  using BlockStoreKeysT = BlockStore<KeyT, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_TRANSPOSE>;
  
  using BlockLoadValuesT  = BlockLoad<ValueT, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE>;
  using BlockStoreValuesT = BlockStore<ValueT, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_TRANSPOSE>;

  /**
   * @brief Standard comparison operator
   */
  struct CompareOp
  {
    __device__ __forceinline__ bool operator()(const KeyT &a, const KeyT &b) const
    {
      return IS_DESCENDING ? (a > b) : (a < b);
    }
  };

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  OffsetT segment_begin;
  OffsetT segment_end;
  OffsetT num_items;

  //---------------------------------------------------------------------
  // Shared memory storage
  //---------------------------------------------------------------------

  union TempStorage
  {
    typename BlockMergeSortT::TempStorage    sort;
    typename BlockLoadKeysT::TempStorage     load_keys;
    typename BlockStoreKeysT::TempStorage    store_keys;
    typename BlockLoadValuesT::TempStorage   load_values;
    typename BlockStoreValuesT::TempStorage  store_values;
  };

  TempStorage &temp_storage;

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  __device__ __forceinline__
  AgentSegmentedMergeSort(OffsetT segment_begin,
                          OffsetT segment_end,
                          TempStorage &temp_storage)
      : segment_begin(segment_begin)
      , segment_end(segment_end)  
      , num_items(segment_end - segment_begin)
      , temp_storage(temp_storage)
  {}

  //---------------------------------------------------------------------
  // Utility functions
  //---------------------------------------------------------------------

  /**
   * @brief Get appropriate out-of-bounds sentinel value that won't interfere with sorting
   */
  __device__ __forceinline__ KeyT GetSentinelValue() const
  {
    if (IS_DESCENDING)
    {
      // For descending sort, use minimum value so it ends up at the end
      return Traits<KeyT>::Lowest();
    }
    else
    {
      // For ascending sort, use maximum value so it ends up at the end
      return Traits<KeyT>::Max();
    }
  }

  //---------------------------------------------------------------------
  // Simple load/store operations
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // Main processing method
  //---------------------------------------------------------------------
  
  __device__ __forceinline__ void ProcessSegment(
    const KeyT *d_keys_in,
    KeyT *d_keys_out,
    const ValueT *d_values_in,
    ValueT *d_values_out)
  {
    // Early exit for empty segments
    if (num_items <= 0) return;

    KeyT thread_keys[ITEMS_PER_THREAD];
    ValueT thread_values[ITEMS_PER_THREAD];
    CompareOp compare_op;
    KeyT oob_default = GetSentinelValue();
    
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      OffsetT global_idx = segment_begin + threadIdx.x * ITEMS_PER_THREAD + i;
      if (global_idx < segment_end)
      {
        thread_keys[i] = d_keys_in[global_idx];
        if (!KEYS_ONLY)
          thread_values[i] = d_values_in[global_idx];
      }
      else
      {
        thread_keys[i] = oob_default;
      }
    }

    // Sort using BlockMergeSort with stable sorting
    BlockMergeSortT block_sort(temp_storage.sort);
    
    if (KEYS_ONLY)
    {
      block_sort.StableSort(thread_keys, compare_op, static_cast<int>(num_items), oob_default);
    }
    else
    {
      block_sort.StableSort(thread_keys, thread_values, compare_op, static_cast<int>(num_items), oob_default);
    }

    __syncthreads();

    // Store results
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      OffsetT global_idx = segment_begin + threadIdx.x * ITEMS_PER_THREAD + i;
      if (global_idx < segment_end)
      {
        d_keys_out[global_idx] = thread_keys[i];
        if (!KEYS_ONLY)
          d_values_out[global_idx] = thread_values[i];
      }
    }
  }
};

CUB_NAMESPACE_END