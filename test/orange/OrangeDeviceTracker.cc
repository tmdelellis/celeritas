//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeDeviceTracker.cc
//---------------------------------------------------------------------------//
#include "OrangeDeviceTracker.hh"

#include <fstream>
#include <iostream>
#include <random>
#include <nlohmann/json.hpp>

#include "celeritas/random/distribution/IsotropicDistribution.hh"
#include "celeritas/random/distribution/UniformBoxDistribution.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct from JSON file.
 */
OrangeDeviceTracker::OrangeDeviceTracker(std::string const& json_file)
    : params_(json_file), params_vec_(1)
{
    // Copy the pointer to the device through a device vector
    params_vec_.copy_to_device({&params_.device_ref(), 1});

    auto matids_from_json = [](std::string const& json_filename) {
        using nlohmann::json;

        std::vector<unsigned int> all_matids;

        std::ifstream infile(json_filename);
        nlohmann::json::parse(infile)["materials"]["cell_to_mat"].get_to(
            all_matids);
        return all_matids;
    };

    // Store the matids from the json file
    auto matids = matids_from_json(json_file);
    matids_ = DeviceVector<unsigned int>(matids.size());
    matids_.copy_to_device(make_span(matids));
}

//---------------------------------------------------------------------------//
/*!
 * Allocate states.
 */
void OrangeDeviceTracker::allocate(unsigned int num_particles)
{
    states_ = CollectionStateStore<OrangeStateData, MemSpace::device>(
        params_.host_ref(), num_particles);
    states_vec_ = DeviceVector<DeviceRef<OrangeStateData>>(1);
    states_vec_.copy_to_device({&states_.ref(), 1});
}

//---------------------------------------------------------------------------//
/*!
 * Track through geometry.
 */
void OrangeDeviceTracker::track(Real3 low, Real3 high) const
{
    auto N = states_.size();
    std::mt19937 rng(23423121);

    UniformBoxDistribution<> sample_pos(low, high);
    IsotropicDistribution<> sample_dir;

    std::vector<Real3> pos(N);
    std::vector<Real3> dir(N);
    std::vector<unsigned int> indices(N);

    std::iota(indices.begin(), indices.end(), 0);

    for (auto n : range(N))
    {
        pos[n] = sample_pos(rng);
        dir[n] = sample_dir(rng);
    }

    SpaceVector pos_d(N);
    SpaceVector dir_d(N);
    IndexVector indices_d(N);

    pos_d.copy_to_device(make_span(pos));
    dir_d.copy_to_device(make_span(dir));
    indices_d.copy_to_device(make_span(indices));

    this->initialize(indices_d, pos_d, dir_d);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
