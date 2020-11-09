//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelDiagnostics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>
#include <string>
#include <utility>
#include "celeritas_config.h"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Accumulate diagnostics about kernels.
 *
 * This class allows storage of Occupancy, Register and other diagnostics for
 * Cuda kernels.
 */
class KernelDiagnostics
{
  public:
    //@{
    //! Type aliases
    using KernelOccupancy    = std::pair<unsigned int, double>;
    using KernelOccupancyMap = std::unordered_map<std::string, KernelOccupancy>;
    //@}

  public:
    // Calculate occupancy for a kernel
    template<class T>
    static void store_occupancy(T&                 function,
                                const std::string& name,
                                unsigned int       block_size);

    //! Get kernel data.
    static const KernelOccupancyMap& occupancy_map() { return occupancy_; }

  private:
    // >>> STATIC DIAGNOSTIC STORAGE

    // Stored occupancy data
    static KernelOccupancyMap occupancy_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#ifdef CELERITAS_USE_CUDA
#    include "KernelDiagnostics.i.hh"
#endif