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

#include <cub/agent/agent_segmented_merge_sort.cuh>
#include <cub/detail/device_double_buffer.cuh>
#include <cub/detail/temporary_storage.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_namespace.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <type_traits>

CUB_NAMESPACE_BEGIN

/**
 * @brief Kernel for sorting segments using merge sort
 */
template <bool IS_DESCENDING,
          typename ChainedPolicyT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT,
          typename CompareOpT>
__launch_bounds__(ChainedPolicyT::ActivePolicy::BLOCK_THREADS)
__global__ void DeviceSegmentedMergeSortKernel(
    const KeyT *d_keys_in,
    KeyT *d_keys_out,
    const ValueT *d_values_in,
    ValueT *d_values_out,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    int num_segments,
    CompareOpT compare_op)
{
  using ActivePolicyT = typename ChainedPolicyT::ActivePolicy;
  using AgentSegmentedMergeSortT =
    AgentSegmentedMergeSort<IS_DESCENDING,
                            ActivePolicyT,
                            KeyT,
                            ValueT,
                            OffsetT,
                            CompareOpT>;

  const unsigned int segment_id = blockIdx.x;
  
  if (segment_id >= num_segments)
  {
    return;
  }

  OffsetT segment_begin = d_begin_offsets[segment_id];
  OffsetT segment_end   = d_end_offsets[segment_id];
  OffsetT num_items     = segment_end - segment_begin;

  if (num_items <= 0)
  {
    return;
  }

  __shared__ typename AgentSegmentedMergeSortT::TempStorage temp_storage;

  AgentSegmentedMergeSortT agent(num_items, temp_storage, compare_op);
  agent.ProcessSegment(d_keys_in + segment_begin,
                       d_keys_out + segment_begin,
                       d_values_in + segment_begin,
                       d_values_out + segment_begin);
}

/**
 * @brief Dispatch class for segmented merge sort
 */
template <bool IS_DESCENDING,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT,
          typename CompareOpT>
struct DispatchSegmentedMergeSort
{
  //---------------------------------------------------------------------
  // Tuning policies
  //---------------------------------------------------------------------

  /// SM35
  struct Policy350
  {
    static constexpr int BLOCK_THREADS = 128;
    static constexpr int ITEMS_PER_THREAD = 8;
  };

  /// SM70
  struct Policy700 : Policy350
  {
    static constexpr int BLOCK_THREADS = 256;
    static constexpr int ITEMS_PER_THREAD = 12;
  };

  /// SM80
  struct Policy800 : Policy700
  {
    static constexpr int BLOCK_THREADS = 256;
    static constexpr int ITEMS_PER_THREAD = 16;
  };

  /// SM90
  struct Policy900 : Policy800
  {
    static constexpr int BLOCK_THREADS = 256;
    static constexpr int ITEMS_PER_THREAD = 20;
  };

  /// Tuning policies of current PTX compiler pass
  using PtxPolicy = Policy350;

  // "Opaque" policies (whose parameterizations aren't reflected in the interface)
  struct KernelConfig
  {
    int block_threads;
    int items_per_thread;
  };

  //---------------------------------------------------------------------
  // Tuning policies of current PTX compiler pass
  //---------------------------------------------------------------------

#if (CUB_PTX_ARCH >= 900)
  using ActivePolicyT = Policy900;

#elif (CUB_PTX_ARCH >= 800)
  using ActivePolicyT = Policy800;

#elif (CUB_PTX_ARCH >= 700)
  using ActivePolicyT = Policy700;

#else
  using ActivePolicyT = Policy350;

#endif

  /// MaxPolicy
  using MaxPolicy = Policy900;

  /// Single-sweep kernel entry point
  using KernelT = void (*)(const KeyT *,
                          KeyT *,
                          const ValueT *,
                          ValueT *,
                          BeginOffsetIteratorT,
                          EndOffsetIteratorT,
                          int,
                          CompareOpT);

  //---------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine for computing a device-wide segmented merge sort
   */
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           DoubleBuffer<KeyT> &d_keys,
           DoubleBuffer<ValueT> &d_values,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           CompareOpT compare_op,
           cudaStream_t stream)
  {
    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      if (CubDebug(error = PtxVersion(ptx_version)))
        break;

      // Get kernel kernel dispatch configurations
      KernelConfig config;
      InitConfigs(ptx_version, config);

      // Number of items per thread
      int tile_size = config.block_threads * config.items_per_thread;

      // Zero-size check
      if (num_items == 0 || num_segments == 0)
      {
        if (d_temp_storage == nullptr)
        {
          temp_storage_bytes = 0;
        }
        break;
      }

      // For now, we don't need temporary storage for merge sort
      // In future implementations, we might need it for multi-pass sorting
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 1;  // Minimal allocation
        break;
      }

      // Log kernel configuration
      #if defined(CUB_DETAIL_DEBUG_ENABLE_LOG)
      _CubLog("Invoking DeviceSegmentedMergeSortKernel<<<%d, %d>>>(), "
              "%d items per thread, %d SM occupancy\n",
              num_segments,
              config.block_threads,
              config.items_per_thread,
              1);
      #endif

      // Invoke kernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
          num_segments,
          config.block_threads,
          0,
          stream
      ).doit(DeviceSegmentedMergeSortKernel<IS_DESCENDING,
                                            MaxPolicy,
                                            KeyT,
                                            ValueT,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT,
                                            OffsetT,
                                            CompareOpT>,
             d_keys.Current(),
             d_keys.Alternate(),
             d_values.Current(),
             d_values.Alternate(),
             d_begin_offsets,
             d_end_offsets,
             num_segments,
             compare_op);

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
        break;

      // Sync the stream if specified to flush runtime errors
      error = detail::DebugSyncStream(stream);
      if (CubDebug(error))
        break;

      // Update selectors
      d_keys.selector ^= 1;
      d_values.selector ^= 1;
    }
    while (0);

    return error;
  }

  /**
   * Initialize kernel dispatch configurations with the policies corresponding 
   * to the PTX assembly we will use
   */
  CUB_RUNTIME_FUNCTION __forceinline__ static void
  InitConfigs(int ptx_version, KernelConfig &config)
  {
    if (ptx_version >= 900)
    {
      config.block_threads = Policy900::BLOCK_THREADS;
      config.items_per_thread = Policy900::ITEMS_PER_THREAD;
    }
    else if (ptx_version >= 800)
    {
      config.block_threads = Policy800::BLOCK_THREADS;
      config.items_per_thread = Policy800::ITEMS_PER_THREAD;
    }
    else if (ptx_version >= 700)
    {
      config.block_threads = Policy700::BLOCK_THREADS;
      config.items_per_thread = Policy700::ITEMS_PER_THREAD;
    }
    else
    {
      config.block_threads = Policy350::BLOCK_THREADS;
      config.items_per_thread = Policy350::ITEMS_PER_THREAD;
    }
  }
};

CUB_NAMESPACE_END