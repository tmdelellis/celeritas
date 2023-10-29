//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeDeviceTracker.cu
//---------------------------------------------------------------------------//
#include "OrangeDeviceTracker.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
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
__global__ void initialize_kernel(OrangeParams::DeviceRef const* params,
                                  DeviceRef<OrangeStateData> const* states,
                                  Span<unsigned int const> indices,
                                  Span<Real3 const> pos,
                                  Span<Real3 const> dir)
{
    auto tid = KernelParamCalculator::thread_id();
    if (tid.get() >= indices.size())
    {
        return;
    }

    auto pid = TrackSlotId{static_cast<size_type>(indices[tid.get()])};
    OrangeTrackView track(*params, *states, pid);
    track = {pos[tid.get()], dir[tid.get()]};
}

//---------------------------------------------------------------------------//
__global__ void d2b_kernel(OrangeParams::DeviceRef const* params,
                           DeviceRef<OrangeStateData> const* states,
                           Span<unsigned int const> indices,
                           Span<double> distances)
{
    auto tid = KernelParamCalculator::thread_id();
    if (tid.get() >= indices.size())
    {
        return;
    }

    auto pid
        = celeritas::TrackSlotId{static_cast<size_type>(indices[tid.get()])};
    OrangeTrackView track(*params, *states, pid);
    distances[tid.get()] = track.find_next_step().distance;
}

//---------------------------------------------------------------------------//
__global__ void move_to_point_kernel(OrangeParams::DeviceRef const* params,
                                     DeviceRef<OrangeStateData> const* states,
                                     Span<unsigned int const> indices,
                                     Span<bool const> mask,
                                     Span<double const> distances)
{
    auto tid = KernelParamCalculator::thread_id();

    if (tid.get() < indices.size() && mask[tid.get()])
    {
        auto pid = TrackSlotId{static_cast<size_type>(indices[tid.get()])};
        OrangeTrackView track(*params, *states, pid);

        auto next_step = track.find_next_step().distance;
        track.move_internal(distances[tid.get()]);
    }
}

//---------------------------------------------------------------------------//
__global__ void move_across_surface_kernel(
    OrangeParams::DeviceRef const* params,
    DeviceRef<OrangeStateData> const* states,
    Span<unsigned int const> all_matids,
    Span<unsigned int const> indices,
    Span<bool const> mask,
    Span<OrangeDeviceTracker::BoundaryState> boundary_states,
    Span<unsigned int> cells,
    Span<unsigned int> matids)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();

    if (tid.get() < indices.size() && mask[tid.get()])
    {
        auto pid = TrackSlotId{static_cast<size_type>(indices[tid.get()])};
        celeritas::OrangeTrackView track(*params, *states, pid);
        track.find_next_step();
        track.move_to_boundary();
        track.cross_boundary();

        if (!track.is_outside())
        {
            boundary_states[tid.get()]
                = OrangeDeviceTracker::BoundaryState::INSIDE;
            cells[tid.get()] = track.volume_id().get();
            matids[tid.get()] = all_matids[track.volume_id().get()];
        }
        else
        {
            boundary_states[tid.get()]
                = OrangeDeviceTracker::BoundaryState::OUTSIDE;
            cells[tid.get()] = OrangeDeviceTracker::invalid_id();
            matids[tid.get()] = OrangeDeviceTracker::invalid_id();
        }
    }
}

//---------------------------------------------------------------------------//
// PRIVATE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Initialize the geometry.
 */
void OrangeDeviceTracker::initialize(IndexVector const& indices,
                                     SpaceVector const& pos,
                                     SpaceVector const& dir) const
{
    CELER_EXPECT(indices.size() == pos.size());
    CELER_EXPECT(indices.size() == dir.size());

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
/*!
 * Distance to boundary.
 */
void OrangeDeviceTracker::distance_to_boundary(IndexVector const& indices,
                                               DoubleVector& distances) const
{
    CELER_EXPECT(indices.size() == distances.size());

    auto params = make_observer(params_vec_);
    auto states = make_observer(states_vec_);
    CELER_LAUNCH_KERNEL(d2b,
                        indices.size(),
                        0,
                        params.get(),
                        states.get(),
                        indices.device_ref(),
                        distances.device_ref());
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
/*!
 * Move to point.
 */
void OrangeDeviceTracker::move_to_point(IndexVector const& indices,
                                        BoolVector const& mask,
                                        DoubleVector const& distances) const
{
    CELER_EXPECT(indices.size() == distances.size());
    CELER_EXPECT(indices.size() == mask.size());

    auto params = make_observer(params_vec_);
    auto states = make_observer(states_vec_);
    CELER_LAUNCH_KERNEL(move_to_point,
                        indices.size(),
                        0,
                        params.get(),
                        states.get(),
                        indices.device_ref(),
                        mask.device_ref(),
                        distances.device_ref());
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
/*!
 * Move across surface.
 */
void OrangeDeviceTracker::move_across_surface(IndexVector const& indices,
                                              BoolVector const& mask,
                                              BoundaryVector& boundary_states,
                                              IndexVector& cells,
                                              IndexVector& matids) const
{
    CELER_EXPECT(indices.size() == boundary_states.size());
    CELER_EXPECT(indices.size() == mask.size());
    CELER_EXPECT(indices.size() == matids.size());

    auto params = make_observer(params_vec_);
    auto states = make_observer(states_vec_);
    CELER_LAUNCH_KERNEL(move_across_surface,
                        indices.size(),
                        0,
                        params.get(),
                        states.get(),
                        matids_.device_ref(),
                        indices.device_ref(),
                        mask.device_ref(),
                        boundary_states.device_ref(),
                        cells.device_ref(),
                        matids.device_ref());
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas