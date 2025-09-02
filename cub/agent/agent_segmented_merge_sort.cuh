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

#include <cub/block/block_load.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_store.cuh>
#include <cub/config.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_merge_sort.cuh>

CUB_NAMESPACE_BEGIN

/**
 * @brief Agent for segmented merge sort
 *
 * @tparam IS_DESCENDING
 *   Whether or not the sorted-order is high-to-low
 *
 * @tparam PolicyT
 *   Parameterized tuning policy type
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam CompareOpT
 *   Binary comparison function object type
 */
template <bool IS_DESCENDING,
          typename PolicyT,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename CompareOpT>
struct AgentSegmentedMergeSort
{
  //---------------------------------------------------------------------
  // Constants
  //---------------------------------------------------------------------

  static constexpr int BLOCK_THREADS    = PolicyT::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = PolicyT::ITEMS_PER_THREAD;
  static constexpr int TILE_SIZE        = BLOCK_THREADS * ITEMS_PER_THREAD;
  static constexpr bool KEYS_ONLY       = std::is_same<ValueT, NullType>::value;

  //---------------------------------------------------------------------
  // Type definitions
  //---------------------------------------------------------------------

  using BlockMergeSortT = BlockMergeSort<KeyT, 
                                        BLOCK_THREADS, 
                                        ITEMS_PER_THREAD, 
                                        ValueT>;

  using BlockLoadKeysT = BlockLoad<KeyT, 
                                  BLOCK_THREADS, 
                                  ITEMS_PER_THREAD, 
                                  BLOCK_LOAD_TRANSPOSE>;

  using BlockLoadValuesT = BlockLoad<ValueT, 
                                    BLOCK_THREADS, 
                                    ITEMS_PER_THREAD, 
                                    BLOCK_LOAD_TRANSPOSE>;

  using BlockStoreKeysT = BlockStore<KeyT, 
                                    BLOCK_THREADS, 
                                    ITEMS_PER_THREAD, 
                                    BLOCK_STORE_TRANSPOSE>;

  using BlockStoreValuesT = BlockStore<ValueT, 
                                      BLOCK_THREADS, 
                                      ITEMS_PER_THREAD, 
                                      BLOCK_STORE_TRANSPOSE>;

  using WarpMergeSortT = WarpMergeSort<KeyT, 
                                      ITEMS_PER_THREAD, 
                                      CUB_PTX_WARP_THREADS, 
                                      ValueT>;

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  OffsetT num_items;
  CompareOpT compare_op;

  //---------------------------------------------------------------------
  // Shared memory storage
  //---------------------------------------------------------------------

  union _TempStorage
  {
    typename BlockMergeSortT::TempStorage block_merge_sort;
    
    struct LoadStore
    {
      union
      {
        typename BlockLoadKeysT::TempStorage load_keys;
        typename BlockStoreKeysT::TempStorage store_keys;
      };
      
      union
      {
        typename BlockLoadValuesT::TempStorage load_values;
        typename BlockStoreValuesT::TempStorage store_values;
      };
    } load_store;

    typename WarpMergeSortT::TempStorage warp_merge_sort[BLOCK_THREADS / CUB_PTX_WARP_THREADS];
  };

  using TempStorage = Uninitialized<_TempStorage>;
  _TempStorage &temp_storage;

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  __device__ __forceinline__
  AgentSegmentedMergeSort(OffsetT num_items,
                          TempStorage &temp_storage,
                          CompareOpT compare_op)
      : num_items(num_items)
      , compare_op(compare_op)
      , temp_storage(temp_storage.Alias())
  {}

  //---------------------------------------------------------------------
  // Block-wide merge sort for a single segment
  //---------------------------------------------------------------------

  __device__ __forceinline__ void
  BlockSortFullTile(const KeyT *d_keys_in,
                    KeyT *d_keys_out,
                    const ValueT *d_values_in,
                    ValueT *d_values_out)
  {
    KeyT thread_keys[ITEMS_PER_THREAD];
    ValueT thread_values[ITEMS_PER_THREAD];

    // Load keys and values
    BlockLoadKeysT(temp_storage.load_store.load_keys).Load(d_keys_in, thread_keys);
    __syncthreads();

    if (!KEYS_ONLY)
    {
      BlockLoadValuesT(temp_storage.load_store.load_values).Load(d_values_in, thread_values);
      __syncthreads();
    }

    // Sort keys and values
    if (IS_DESCENDING)
    {
      BlockMergeSortT(temp_storage.block_merge_sort).SortDescending(thread_keys, thread_values, compare_op);
    }
    else
    {
      BlockMergeSortT(temp_storage.block_merge_sort).Sort(thread_keys, thread_values, compare_op);
    }
    __syncthreads();

    // Store sorted keys and values
    BlockStoreKeysT(temp_storage.load_store.store_keys).Store(d_keys_out, thread_keys);
    __syncthreads();

    if (!KEYS_ONLY)
    {
      BlockStoreValuesT(temp_storage.load_store.store_values).Store(d_values_out, thread_values);
      __syncthreads();
    }
  }

  //---------------------------------------------------------------------
  // Block-wide merge sort for a partial tile
  //---------------------------------------------------------------------

  __device__ __forceinline__ void
  BlockSortPartialTile(const KeyT *d_keys_in,
                       KeyT *d_keys_out,
                       const ValueT *d_values_in,
                       ValueT *d_values_out,
                       OffsetT valid_items)
  {
    KeyT thread_keys[ITEMS_PER_THREAD];
    ValueT thread_values[ITEMS_PER_THREAD];

    // Load keys and values with bounds checking
    KeyT oob_default = IS_DESCENDING ? 
                       std::numeric_limits<KeyT>::lowest() : 
                       std::numeric_limits<KeyT>::max();

    BlockLoadKeysT(temp_storage.load_store.load_keys).Load(
        d_keys_in, thread_keys, valid_items, oob_default);
    __syncthreads();

    if (!KEYS_ONLY)
    {
      BlockLoadValuesT(temp_storage.load_store.load_values).Load(
          d_values_in, thread_values, valid_items);
      __syncthreads();
    }

    // Sort keys and values
    if (IS_DESCENDING)
    {
      BlockMergeSortT(temp_storage.block_merge_sort).SortDescending(thread_keys, thread_values, compare_op);
    }
    else
    {
      BlockMergeSortT(temp_storage.block_merge_sort).Sort(thread_keys, thread_values, compare_op);
    }
    __syncthreads();

    // Store sorted keys and values with bounds checking
    BlockStoreKeysT(temp_storage.load_store.store_keys).Store(
        d_keys_out, thread_keys, valid_items);
    __syncthreads();

    if (!KEYS_ONLY)
    {
      BlockStoreValuesT(temp_storage.load_store.store_values).Store(
          d_values_out, thread_values, valid_items);
      __syncthreads();
    }
  }

  //---------------------------------------------------------------------
  // Warp-wide merge sort for small segments
  //---------------------------------------------------------------------

  __device__ __forceinline__ void
  WarpSort(const KeyT *d_keys_in,
           KeyT *d_keys_out,
           const ValueT *d_values_in,
           ValueT *d_values_out,
           OffsetT valid_items)
  {
    const int warp_id = threadIdx.x / CUB_PTX_WARP_THREADS;
    const int lane_id = threadIdx.x % CUB_PTX_WARP_THREADS;

    // Check if this warp participates
    if (warp_id * CUB_PTX_WARP_THREADS >= valid_items)
    {
      return;
    }

    KeyT thread_keys[ITEMS_PER_THREAD];
    ValueT thread_values[ITEMS_PER_THREAD];

    // Calculate how many items this warp processes
    OffsetT warp_offset = warp_id * CUB_PTX_WARP_THREADS * ITEMS_PER_THREAD;
    OffsetT warp_items = min(static_cast<OffsetT>(CUB_PTX_WARP_THREADS * ITEMS_PER_THREAD), 
                             valid_items - warp_offset);

    // Load keys for this warp
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      OffsetT idx = warp_offset + lane_id + i * CUB_PTX_WARP_THREADS;
      if (idx < valid_items)
      {
        thread_keys[i] = d_keys_in[idx];
        if (!KEYS_ONLY)
        {
          thread_values[i] = d_values_in[idx];
        }
      }
      else
      {
        thread_keys[i] = IS_DESCENDING ? 
                         std::numeric_limits<KeyT>::lowest() : 
                         std::numeric_limits<KeyT>::max();
      }
    }

    // Sort within warp  
    if (IS_DESCENDING)
    {
      WarpMergeSortT(temp_storage.warp_merge_sort[warp_id]).SortDescending(thread_keys, thread_values, compare_op);
    }
    else
    {
      WarpMergeSortT(temp_storage.warp_merge_sort[warp_id]).Sort(thread_keys, thread_values, compare_op);
    }

    // Store sorted data
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      OffsetT idx = warp_offset + lane_id + i * CUB_PTX_WARP_THREADS;
      if (idx < valid_items)
      {
        d_keys_out[idx] = thread_keys[i];
        if (!KEYS_ONLY)
        {
          d_values_out[idx] = thread_values[i];
        }
      }
    }
  }

  //---------------------------------------------------------------------
  // Process a single segment
  //---------------------------------------------------------------------

  __device__ __forceinline__ void
  ProcessSegment(const KeyT *d_keys_in,
                 KeyT *d_keys_out,
                 const ValueT *d_values_in,
                 ValueT *d_values_out)
  {
    // Handle different segment sizes
    if (num_items <= CUB_PTX_WARP_THREADS * ITEMS_PER_THREAD)
    {
      // Small segment: use warp-level sort
      WarpSort(d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    }
    else if (num_items == TILE_SIZE)
    {
      // Full tile: use optimized full-tile sort
      BlockSortFullTile(d_keys_in, d_keys_out, d_values_in, d_values_out);
    }
    else
    {
      // Partial tile: use guarded sort
      BlockSortPartialTile(d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    }
  }

  //---------------------------------------------------------------------
  // Process multiple passes for large segments (future enhancement)
  //---------------------------------------------------------------------

  __device__ __forceinline__ void
  ProcessLargeSegment(const KeyT *d_keys_in,
                      KeyT *d_keys_out,
                      const ValueT *d_values_in,
                      ValueT *d_values_out)
  {
    // For now, we only handle segments up to TILE_SIZE
    // Future enhancement: implement multi-pass merge sort for larger segments
    ProcessSegment(d_keys_in, d_keys_out, d_values_in, d_values_out);
  }
};

CUB_NAMESPACE_END