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

#include <cub/agent/agent_segmented_merge_sort_final.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_namespace.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN

/**
 * @brief Simple policy for merge sort
 */
struct SimpleMergeSortPolicy
{
  static constexpr int BLOCK_THREADS = 128;
  static constexpr int ITEMS_PER_THREAD = 9;
};

/**
 * @brief Kernel for sorting segments using BlockMergeSort
 */
template <bool IS_DESCENDING,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
__launch_bounds__(SimpleMergeSortPolicy::BLOCK_THREADS)
__global__ void DeviceSegmentedMergeSortKernel(
    const KeyT *d_keys_in,
    KeyT *d_keys_out,
    const ValueT *d_values_in,
    ValueT *d_values_out,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    int num_segments)
{
  using AgentT = AgentSegmentedMergeSortFinal<IS_DESCENDING,
                                             SimpleMergeSortPolicy,
                                             KeyT,
                                             ValueT,
                                             OffsetT>;

  const unsigned int segment_id = blockIdx.x;
  
  if (segment_id >= num_segments)
    return;

  // Get segment boundaries
  OffsetT segment_begin = d_begin_offsets[segment_id];
  OffsetT segment_end = d_end_offsets[segment_id];
  
  if (segment_begin >= segment_end)
    return;

  __shared__ typename AgentT::TempStorage temp_storage;

  AgentT agent(segment_begin, segment_end, temp_storage);
  agent.ProcessSegment(d_keys_in, d_keys_out, d_values_in, d_values_out);
}

/**
 * @brief Simple dispatch class for segmented merge sort
 */
template <bool IS_DESCENDING,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
struct DispatchSegmentedMergeSortSimple
{
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           DoubleBuffer<KeyT> &d_keys,
           DoubleBuffer<ValueT> &d_values,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           cudaStream_t stream)
  {
    (void)num_items; // Suppress unused parameter warning
    cudaError error = cudaSuccess;

    do
    {
      // No temp storage needed for simple case
      temp_storage_bytes = 1; // Non-zero to indicate success
      
      // Return if the caller is simply requesting the size of temp storage
      if (d_temp_storage == nullptr)
        return cudaSuccess;

      // Launch kernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        num_segments,
        SimpleMergeSortPolicy::BLOCK_THREADS,
        0,
        stream
      ).doit(DeviceSegmentedMergeSortKernel<IS_DESCENDING,
                                            KeyT,
                                            ValueT,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT,
                                            OffsetT>,
             d_keys.Current(),
             d_keys.Alternate(),
             d_values.Current(),
             d_values.Alternate(),
             d_begin_offsets,
             d_end_offsets,
             num_segments);

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
        break;

      // Sync the stream if specified to flush runtime errors
      if (CubDebug(error = SyncStream(stream)))
        break;

    } while (0);

    return error;
  }
};

CUB_NAMESPACE_END