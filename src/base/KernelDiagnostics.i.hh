//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelDiagnostics.i.hh
//---------------------------------------------------------------------------//

#include <cuda_runtime_api.h>
#include "Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Add diagnostics for a kernel function.
 */
template<class T>
void KernelDiagnostics::store_occupancy(T&                 function,
                                        const std::string& name,
                                        unsigned int       block_size)
{
    // Get maximum threads per MP
    int            device = 0;
    cudaDeviceProp props;
    CELER_CUDA_CALL(cudaGetDevice(&device));
    CELER_CUDA_CALL(cudaGetDeviceProperties(&props, device));
    int max_device_threads = props.maxThreadsPerMultiProcessor;

    // Get attributes about kernel
    int                max_func_blocks = 0;
    int                shmem           = 0;
    cudaFuncAttributes attr;
    CELER_CUDA_CALL(cudaFuncGetAttributes(&attr, function));
    CELER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_func_blocks, function, block_size, shmem));
    double occupancy = static_cast<double>(max_func_blocks * block_size)
                       / static_cast<double>(max_device_threads);

    // Store the registers and occupancy
    occupancy_[name] = {attr.numRegs, occupancy};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
