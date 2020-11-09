//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KNDemoIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include <string>
#include "KNDemoKernel.hh"
#include "base/UniformGrid.hh"
#include "base/Types.hh"
#include <nlohmann/json.hpp>

namespace demo_interactor
{
//---------------------------------------------------------------------------//
// Classes
//---------------------------------------------------------------------------//
//! Input for a single run
struct KNDemoRunArgs
{
    using size_type  = celeritas::size_type;
    using GridParams = celeritas::UniformGrid::Params;

    double        energy;
    unsigned long seed;
    size_type     num_tracks;
    size_type     max_steps;
    GridParams    tally_grid;
};

//! Output from a single run
struct KNDemoResult
{
    using size_type = celeritas::size_type;

    std::vector<double>    time;  //!< Real time per step
    std::vector<size_type> alive; //!< Num living tracks per step
    std::vector<double>    edep;  //!< Energy deposition along the grid
    double                 total_time = 0; //!< All time
};

//! Kernel diagnostics
struct KNDemoKernelDiag
{
    std::vector<std::string>  kernel;    //!< Kernel name
    std::vector<unsigned int> registers; //!< Kernel register use
    std::vector<double>       occupancy; //!< Kernel occupancy
};

//---------------------------------------------------------------------------//
// JSON I/O functions
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, const CudaGridParams& value);
void from_json(const nlohmann::json& j, CudaGridParams& value);

void to_json(nlohmann::json& j, const KNDemoRunArgs& value);
void from_json(const nlohmann::json& j, KNDemoRunArgs& value);

void to_json(nlohmann::json& j, const KNDemoResult& value);
void from_json(const nlohmann::json& j, KNDemoResult& value);

void to_json(nlohmann::json& j, const KNDemoKernelDiag& value);
void from_json(const nlohmann::json& j, KNDemoKernelDiag& value);

//---------------------------------------------------------------------------//
} // namespace demo_interactor

namespace celeritas
{
void to_json(nlohmann::json& j, const UniformGrid::Params& value);
void from_json(const nlohmann::json& j, UniformGrid::Params& value);
} // namespace celeritas
