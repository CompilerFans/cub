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
#include <cub/config.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>
#include <cub/thread/thread_operators.cuh>

CUB_NAMESPACE_BEGIN

/**
 * @brief Fixed agent for segmented merge sort using BlockMergeSort correctly
 */
template <bool IS_DESCENDING,
          typename PolicyT,
          typename KeyT,
          typename ValueT,
          typename OffsetT>
struct AgentSegmentedMergeSortFixed
{
  //---------------------------------------------------------------------
  // Constants and type definitions
  //---------------------------------------------------------------------

  static constexpr int BLOCK_THREADS    = PolicyT::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = PolicyT::ITEMS_PER_THREAD;
  static constexpr bool KEYS_ONLY       = std::is_same<ValueT, NullType>::value;

  using BlockMergeSortT = BlockMergeSort<KeyT, BLOCK_THREADS, ITEMS_PER_THREAD, ValueT>;

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

  using TempStorage = typename BlockMergeSortT::TempStorage;
  TempStorage &temp_storage;

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  __device__ __forceinline__
  AgentSegmentedMergeSortFixed(OffsetT segment_begin,
                              OffsetT segment_end,
                              TempStorage &temp_storage)
      : segment_begin(segment_begin)
      , segment_end(segment_end)  
      , num_items(segment_end - segment_begin)
      , temp_storage(temp_storage)
  {}

  //---------------------------------------------------------------------
  // Main processing method
  //---------------------------------------------------------------------
  
  __device__ __forceinline__ void ProcessSegment(
    const KeyT *d_keys_in,
    KeyT *d_keys_out,
    const ValueT *d_values_in,
    ValueT *d_values_out)
  {
    KeyT thread_keys[ITEMS_PER_THREAD];
    ValueT thread_values[ITEMS_PER_THREAD];

    // Initialize with default values
    CompareOp compare_op;
    // Get appropriate out-of-bounds default value
    KeyT oob_default;
    if (IS_DESCENDING)
    {
      // For descending sort, use minimum value
      oob_default = Traits<KeyT>::Lowest();
    }
    else
    {
      // For ascending sort, use maximum value
      oob_default = Traits<KeyT>::Max();
    }

    // Load keys per thread
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
        // Values don't need initialization for OOB items
      }
    }

    // Sort using BlockMergeSort
    BlockMergeSortT block_sort(temp_storage);
    
    if (KEYS_ONLY)
    {
      if (num_items == BLOCK_THREADS * ITEMS_PER_THREAD)
      {
        block_sort.Sort(thread_keys, compare_op);
      }
      else
      {
        block_sort.Sort(thread_keys, compare_op, static_cast<int>(num_items), oob_default);
      }
    }
    else
    {
      if (num_items == BLOCK_THREADS * ITEMS_PER_THREAD)
      {
        block_sort.Sort(thread_keys, thread_values, compare_op);
      }
      else
      {
        block_sort.Sort(thread_keys, thread_values, compare_op, static_cast<int>(num_items), oob_default);
      }
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