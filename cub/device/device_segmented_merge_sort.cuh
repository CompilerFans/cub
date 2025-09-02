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
#include <cub/device/dispatch/dispatch_segmented_merge_sort.cuh>
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

  /**
   * @brief Sorts segments of key-value pairs into ascending order. 
   *        (`~2N` auxiliary storage required)
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation
   * - When input a contiguous sequence of segments, a single sequence
   *   `segment_offsets` (of length `num_segments + 1`) can be aliased
   *   for both the `d_begin_offsets` and `d_end_offsets` parameters (where
   *   the latter is specified as `segment_offsets + 1`).
   * - Let `in` be one of `{d_keys_in, d_values_in}` and `out` be any of
   *   `{d_keys_out, d_values_out}`. The range `[out, out + num_items)` shall 
   *   not overlap `[in, in + num_items)`, 
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - @devicestorageNP For sorting using only `O(P)` temporary storage, see 
   *   the sorting interface using DoubleBuffer wrappers below.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys_in[i]`, `d_values_in[i]`, 
   *   `d_keys_out[i]`, `d_values_out[i]` will not be accessed nor modified.   
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments 
   * (with one zero-length segment) of `int` keys with associated vector of 
   * `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>  
   * // or equivalently <cub/device/device_segmented_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedMergeSort::SortPairs(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedMergeSort::SortPairs(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]
   * @endcode
   *
   * @tparam KeyT                  
   *   **[inferred]** Key type
   *
   * @tparam ValueT                
   *   **[inferred]** Value type
   *
   * @tparam BeginOffsetIteratorT  
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT    
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   ending offsets \iterator
   *
   * @tparam CompareOpT
   *   **[inferred]** Type of binary comparison function object having the 
   *   signature `bool operator()(const KeyT &a, const KeyT &b)`
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in 
   *   Device-accessible pointer to the input data of key data to sort
   *
   * @param[out] d_keys_out 
   *   Device-accessible pointer to the sorted output sequence of key data
   *
   * @param[in] d_values_in 
   *   Device-accessible pointer to the corresponding input sequence of 
   *   associated value items
   *
   * @param[out] d_values_out 
   *   Device-accessible pointer to the correspondingly-reordered output 
   *   sequence of associated value items
   *
   * @param[in] num_items 
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments 
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets 
   *   Random-access input iterator to the sequence of beginning offsets of 
   *   length `num_segments`, such that `d_begin_offsets[i]` is the first 
   *   element of the *i*<sup>th</sup> data segment in `d_keys_*` and 
   *   `d_values_*`
   *
   * @param[in] d_end_offsets 
   *   Random-access input iterator to the sequence of ending offsets of length 
   *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of 
   *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`. If 
   *   `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is 
   *   considered empty.
   *
   * @param[in] compare_op 
   *   Binary comparison function object which returns true if the first argument
   *   is ordered before the second
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename CompareOpT>
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
            CompareOpT compare_op,
            cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(d_values_in),
                                  d_values_out);

    return DispatchSegmentedMergeSort<false,
                                      KeyT,
                                      ValueT,
                                      BeginOffsetIteratorT,
                                      EndOffsetIteratorT,
                                      OffsetT,
                                      CompareOpT>::Dispatch(d_temp_storage,
                                                           temp_storage_bytes,
                                                           d_keys,
                                                           d_values,
                                                           num_items,
                                                           num_segments,
                                                           d_begin_offsets,
                                                           d_end_offsets,
                                                           compare_op,
                                                           stream);
  }

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
    return SortPairs(d_temp_storage,
                     temp_storage_bytes,
                     d_keys_in,
                     d_keys_out,
                     d_values_in,
                     d_values_out,
                     num_items,
                     num_segments,
                     d_begin_offsets,
                     d_end_offsets,
                     ::cuda::std::less<KeyT>{},
                     stream);
  }

  /**
   * @brief Sorts segments of key-value pairs into descending order.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename CompareOpT>
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
                      CompareOpT compare_op,
                      cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(d_values_in),
                                  d_values_out);

    return DispatchSegmentedMergeSort<true,
                                      KeyT,
                                      ValueT,
                                      BeginOffsetIteratorT,
                                      EndOffsetIteratorT,
                                      OffsetT,
                                      CompareOpT>::Dispatch(d_temp_storage,
                                                           temp_storage_bytes,
                                                           d_keys,
                                                           d_values,
                                                           num_items,
                                                           num_segments,
                                                           d_begin_offsets,
                                                           d_end_offsets,
                                                           compare_op,
                                                           stream);
  }

  /**
   * @brief Sorts segments of key-value pairs into descending order using 
   *        the default greater-than comparison operator.
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
    return SortPairsDescending(d_temp_storage,
                               temp_storage_bytes,
                               d_keys_in,
                               d_keys_out,
                               d_values_in,
                               d_values_out,
                               num_items,
                               num_segments,
                               d_begin_offsets,
                               d_end_offsets,
                               ::cuda::std::greater<KeyT>{},
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
            typename EndOffsetIteratorT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           size_t &temp_storage_bytes,
           const KeyT *d_keys_in,
           KeyT *d_keys_out,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           CompareOpT compare_op,
           cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<NullType> d_values;

    return DispatchSegmentedMergeSort<false,
                                      KeyT,
                                      NullType,
                                      BeginOffsetIteratorT,
                                      EndOffsetIteratorT,
                                      OffsetT,
                                      CompareOpT>::Dispatch(d_temp_storage,
                                                           temp_storage_bytes,
                                                           d_keys,
                                                           d_values,
                                                           num_items,
                                                           num_segments,
                                                           d_begin_offsets,
                                                           d_end_offsets,
                                                           compare_op,
                                                           stream);
  }

  /**
   * @brief Sorts segments of keys into ascending order using 
   *        the default less-than comparison operator.
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
    return SortKeys(d_temp_storage,
                    temp_storage_bytes,
                    d_keys_in,
                    d_keys_out,
                    num_items,
                    num_segments,
                    d_begin_offsets,
                    d_end_offsets,
                    ::cuda::std::less<KeyT>{},
                    stream);
  }

  /**
   * @brief Sorts segments of keys into descending order.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     const KeyT *d_keys_in,
                     KeyT *d_keys_out,
                     int num_items,
                     int num_segments,
                     BeginOffsetIteratorT d_begin_offsets,
                     EndOffsetIteratorT d_end_offsets,
                     CompareOpT compare_op,
                     cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<NullType> d_values;

    return DispatchSegmentedMergeSort<true,
                                      KeyT,
                                      NullType,
                                      BeginOffsetIteratorT,
                                      EndOffsetIteratorT,
                                      OffsetT,
                                      CompareOpT>::Dispatch(d_temp_storage,
                                                           temp_storage_bytes,
                                                           d_keys,
                                                           d_values,
                                                           num_items,
                                                           num_segments,
                                                           d_begin_offsets,
                                                           d_end_offsets,
                                                           compare_op,
                                                           stream);
  }

  /**
   * @brief Sorts segments of keys into descending order using 
   *        the default greater-than comparison operator.
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
    return SortKeysDescending(d_temp_storage,
                              temp_storage_bytes,
                              d_keys_in,
                              d_keys_out,
                              num_items,
                              num_segments,
                              d_begin_offsets,
                              d_end_offsets,
                              ::cuda::std::greater<KeyT>{},
                              stream);
  }

  //@}  // end member group
};

CUB_NAMESPACE_END