//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeDeviceTracker.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/DeviceVector.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeParams.hh"
#include "orange/Types.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * ORANGE device tracker.
 */
class OrangeDeviceTracker
{
  public:
    //!@{
    //! \name Type aliases

    //!@}

  public:
    // Construct with defaults
    OrangeDeviceTracker(std::string const& json_file);

    // Allocate the state data
    void allocate(unsigned int num_particles);

    // Track through geometry
    void track(Real3 low, Real3 high);

  private:
    //// TYPES ////

    using IndexVector = DeviceVector<int>;
    using SpaceVector = DeviceVector<Real3>;

    //// IMPLEMENTATION ////

    // Initialize
    void initialize(IndexVector& indices, SpaceVector& pos, SpaceVector& dir);

    //// DATA ////

    // Params data and params pointer (stored through a device vector)
    OrangeParams params_;
    DeviceVector<OrangeParams::DeviceRef> params_vec_;

    // State device store and pointer (stored through a device vector)
    CollectionStateStore<OrangeStateData, MemSpace::device> states_;
    DeviceVector<DeviceRef<OrangeStateData>> states_vec_;

    // Matids
    DeviceVector<unsigned int> matids_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
