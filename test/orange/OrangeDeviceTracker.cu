//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeDeviceTracker.cu
//---------------------------------------------------------------------------//
#include "OrangeDeviceTracker.hh"

#include "corecel/Types.hh"
#include "corecel/device_runtime_api.h"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/ThreadId.hh"
#include "orange/OrangeTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
__global__ void initialize_kernel(OrangeParams::DeviceRef* params,
                                  DeviceRef<OrangeStateData>* states,
                                  Span<int> indices,
                                  Span<Real3> pos,
                                  Span<Real3> dir)
{
    auto tid = KernelParamCalculator::thread_id();
    if (tid.get() >= indices.size())
    {
        return;
    }

    auto pid = TrackSlotId{static_cast<size_type>(indices[tid.get()])};
    OrangeTrackView track(*params, *states, pid);
    track = {pos[pid.get()], dir[pid.get()]};

    printf(
        "Index = %5d; Pos = %12.6f, %12.6f, %12.6f; Dir = %12.6f, %12.6f "
        "%12.6f\n",
        indices[tid.get()],
        pos[tid.get()][0],
        pos[tid.get()][1],
        pos[tid.get()][2],
        dir[tid.get()][0],
        dir[tid.get()][1],
        dir[tid.get()][2]);
}

//---------------------------------------------------------------------------//
// PRIVATE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Initialize the geometry.
 */
void OrangeDeviceTracker::initialize(IndexVector& indices,
                                     SpaceVector& pos,
                                     SpaceVector& dir)
{
    auto params = make_observer(params_vec_);
    auto states = make_observer(states_vec_);
    CELER_LAUNCH_KERNEL(initialize,
                        indices.size(),
                        0,
                        params.get(),
                        states.get(),
                        indices.device_ref(),
                        pos.device_ref(),
                        dir.device_ref());
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas