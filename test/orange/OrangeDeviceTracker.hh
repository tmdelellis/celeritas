//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeDeviceTracker.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
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

    enum class BoundaryState
    {
        INSIDE = 0,
        OUTSIDE = 1
    };

  public:
    // Construct with defaults
    OrangeDeviceTracker(std::string const& json_file);

    // Allocate the state data
    void allocate(unsigned int num_particles);

    // Track through geometry
    void track(Real3 low, Real3 high) const;

    // Invalid id
    CELER_FUNCTION static constexpr unsigned int invalid_id()
    {
        return static_cast<unsigned int>(-1);
    }

  private:
    //// TYPES ////

    using IndexVector = DeviceVector<unsigned int>;
    using SpaceVector = DeviceVector<Real3>;
    using DoubleVector = DeviceVector<double>;
    using BoolVector = DeviceVector<bool>;
    using BoundaryVector = DeviceVector<BoundaryState>;

    //// IMPLEMENTATION ////

    // Initialize
    void initialize(IndexVector const& indices,
                    SpaceVector const& pos,
                    SpaceVector const& dir) const;

    // Distance to boundary
    void distance_to_boundary(IndexVector const& indices,
                              DoubleVector& distances) const;

    // Move to point
    void move_to_point(IndexVector const& indices,
                       BoolVector const& mask,
                       DoubleVector const& distances) const;

    // Move to and across next surface
    void move_across_surface(IndexVector const& indices,
                             BoolVector const& mask,
                             BoundaryVector& boundary_states,
                             IndexVector& cells,
                             IndexVector& matids) const;

    //// DATA ////

    // Params data and params pointer (stored through a device vector)
    OrangeParams params_;
    DeviceVector<OrangeParams::DeviceRef> params_vec_;

    // State device store and pointer (stored through a device vector)
    CollectionStateStore<OrangeStateData, MemSpace::device> states_;
    DeviceVector<DeviceRef<OrangeStateData>> states_vec_;

    // Matids
    IndexVector matids_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
