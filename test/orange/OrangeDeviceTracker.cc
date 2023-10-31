//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeDeviceTracker.cc
//---------------------------------------------------------------------------//
#include "OrangeDeviceTracker.hh"

#include <fstream>
#include <iomanip>
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
unsigned int
OrangeDeviceTracker::track(Real3 low, Real3 high, bool output) const
{
    using std::cout;
    using std::endl;

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

    // Initialize the geometry
    IndexVector indices_d(N);
    indices_d.copy_to_device(make_span(indices));
    {
        SpaceVector pos_d(N);
        SpaceVector dir_d(N);

        pos_d.copy_to_device(make_span(pos));
        dir_d.copy_to_device(make_span(dir));

        this->initialize(indices_d, pos_d, dir_d);
    }

    // Tracking fields
    std::vector<double> distances;
    std::vector<unsigned int> mask;
    std::vector<BoundaryState> bnd_states;
    std::vector<unsigned int> cells;
    std::vector<unsigned int> matids;

    // Track particles
    unsigned int step = 0;
    while (!indices_d.empty())
    {
        auto size = indices_d.size();

        if (output)
        {
            cout << "Iteration = " << step << " ; indices = " << indices.size()
                 << " ; states = " << states_.size() << endl;

            auto pos_d = this->device_vector(pos, size);
            auto dir_d = this->device_vector(dir, size);

            this->pos_dir(indices_d, pos_d, dir_d);

            pos_d.copy_to_host(make_span(pos));
            dir_d.copy_to_host(make_span(dir));

            for (auto n : range(size))
            {
                cout << " * " << std::left << std::setw(6) << indices[n]
                     << std::showpos << std::fixed << std::setprecision(5)
                     << std::setw(12) << pos[n][0] << std::setw(12)
                     << pos[n][1] << std::setw(12) << pos[n][2]
                     << std::setw(12) << dir[n][0] << std::setw(12)
                     << dir[n][1] << std::setw(12) << dir[n][2] << endl;
            }
            cout << endl;
        }

        // Calculate distance to boundary
        auto distances_d = this->device_vector(distances, size);
        this->distance_to_boundary(indices_d, distances_d);

        if (output)
        {
            distances_d.copy_to_host(make_span(distances));
            for (auto n : range(size))
            {
                cout << " ^ " << std::left << std::setw(6) << indices[n]
                     << std::fixed << std::setprecision(5) << std::showpos
                     << distances[n] << endl;
            }
            cout << endl;
        }

        // Move across surface
        std::fill(mask.begin(), mask.end(), 1);
        auto bnd_states_d = this->device_vector(bnd_states, size);
        auto mask_d = this->device_vector(mask, size);
        auto cells_d = this->device_vector(cells, size);
        auto matids_d = this->device_vector(matids, size);
        this->move_across_surface(
            indices_d, mask_d, bnd_states_d, cells_d, matids_d);

        // Check for leaving
        {
            bnd_states_d.copy_to_host(make_span(bnd_states));
            std::vector<unsigned int> tmp_indices;
            for (auto n : range(size))
            {
                if (bnd_states[n] == BoundaryState::INSIDE)
                {
                    tmp_indices.push_back(indices[n]);
                }
            }
            indices = tmp_indices;
        }

        ++step;

        indices_d = this->device_vector(indices, indices.size());
    }

    return step;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
