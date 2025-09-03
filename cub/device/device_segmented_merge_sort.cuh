/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * @file cub::DeviceSegmentedMergeSort provides device-wide, parallel 
 *       operations for computing a batched merge sort across multiple, 
 *       non-overlapping sequences of data items residing within 
 *       device-accessible memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include <cub/config.cuh>
#include <cub/device/dispatch/dispatch_segmented_merge_sort_simple.cuh>
#include <cub/util_deprecated.cuh>

CUB_NAMESPACE_BEGIN

/**
 * @brief DeviceSegmentedMergeSort provides device-wide, parallel operations 
 *        for computing a batched merge sort across multiple, non-overlapping 
 *        sequences of data items residing within device-accessible memory. 
 * @ingroup SegmentedModule
 *
 * @par Overview
 * The merge sort algorithm is a comparison-based sorting algorithm that works
 * by dividing the input into smaller subsequences, sorting them recursively,
 * and then merging them back together. It has O(n log n) time complexity and
 * is stable, meaning equal elements maintain their relative order.
 *
 * @par Segments are not required to be contiguous. Any element of input(s) or 
 * output(s) outside the specified segments will not be accessed nor modified.  
 *
 * @par Usage Considerations
 * @cdp_class{DeviceSegmentedMergeSort}
 *
 */
struct DeviceSegmentedMergeSort
{
  /******************************************************************//**
   * @name Key-value pairs
   *********************************************************************/
  //@{

  // 注意：为了避免constexpr问题，我们移除了需要CompareOpT参数的SortPairs重载函数
  // 用户应该使用不带比较函数的版本（内置升序/降序排序）

  /**
   * @brief Sorts segments of key-value pairs into ascending order using 
   *        the default less-than comparison operator.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            size_t &temp_storage_bytes,
            const KeyT *d_keys_in,
            KeyT *d_keys_out,
            const ValueT *d_values_in,
            ValueT *d_values_out,
            int num_items,
            int num_segments,
            BeginOffsetIteratorT d_begin_offsets,
            EndOffsetIteratorT d_end_offsets,
            cudaStream_t stream = 0)
  {
    // 直接调用内部实现，避免std::less constexpr问题
    using DispatchT = DispatchSegmentedMergeSortSimple<false, KeyT, ValueT, 
                                                      BeginOffsetIteratorT, EndOffsetIteratorT, 
                                                      int>;
    
    DoubleBuffer<KeyT> d_keys(const_cast<KeyT*>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT*>(d_values_in), d_values_out);
    
    return DispatchT::Dispatch(d_temp_storage, temp_storage_bytes,
                              d_keys, d_values, num_items, num_segments,
                              d_begin_offsets, d_end_offsets, 
                              stream);
  }

  /**
   * @brief Sorts segments of key-value pairs into descending order.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      const KeyT *d_keys_in,
                      KeyT *d_keys_out,
                      const ValueT *d_values_in,
                      ValueT *d_values_out,
                      int num_items,
                      int num_segments,
                      BeginOffsetIteratorT d_begin_offsets,
                      EndOffsetIteratorT d_end_offsets,
                      cudaStream_t stream = 0)
  {
    using DispatchT = DispatchSegmentedMergeSortSimple<true, KeyT, ValueT, 
                                                      BeginOffsetIteratorT, EndOffsetIteratorT, 
                                                      int>;
    
    DoubleBuffer<KeyT> d_keys(const_cast<KeyT*>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT*>(d_values_in), d_values_out);
    
    return DispatchT::Dispatch(d_temp_storage, temp_storage_bytes,
                              d_keys, d_values, num_items, num_segments,
                              d_begin_offsets, d_end_offsets, 
                              stream);
  }


  //@}  // end member group
  /******************************************************************//**
   * @name Keys-only
   *********************************************************************/
  //@{

  /**
   * @brief Sorts segments of keys into ascending order.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           size_t &temp_storage_bytes,
           const KeyT *d_keys_in,
           KeyT *d_keys_out,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           cudaStream_t stream = 0)
  {
    using DispatchT = DispatchSegmentedMergeSortSimple<false, KeyT, NullType, 
                                                      BeginOffsetIteratorT, EndOffsetIteratorT, 
                                                      int>;
    
    DoubleBuffer<KeyT> d_keys(const_cast<KeyT*>(d_keys_in), d_keys_out);
    DoubleBuffer<NullType> d_values;
    
    return DispatchT::Dispatch(d_temp_storage, temp_storage_bytes,
                              d_keys, d_values, num_items, num_segments,
                              d_begin_offsets, d_end_offsets, 
                              stream);
  }


  /**
   * @brief Sorts segments of keys into descending order.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     const KeyT *d_keys_in,
                     KeyT *d_keys_out,
                     int num_items,
                     int num_segments,
                     BeginOffsetIteratorT d_begin_offsets,
                     EndOffsetIteratorT d_end_offsets,
                     cudaStream_t stream = 0)
  {
    using DispatchT = DispatchSegmentedMergeSortSimple<true, KeyT, NullType, 
                                                      BeginOffsetIteratorT, EndOffsetIteratorT, 
                                                      int>;
    
    DoubleBuffer<KeyT> d_keys(const_cast<KeyT*>(d_keys_in), d_keys_out);
    DoubleBuffer<NullType> d_values;
    
    return DispatchT::Dispatch(d_temp_storage, temp_storage_bytes,
                              d_keys, d_values, num_items, num_segments,
                              d_begin_offsets, d_end_offsets, 
                              stream);
  }


  //@}  // end member group
};

CUB_NAMESPACE_END