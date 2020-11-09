//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelDiagnostics.cuda.cc
//---------------------------------------------------------------------------//
#include "KernelDiagnostics.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// STATIC MEMBER INITIALIZATION
//---------------------------------------------------------------------------//

KernelDiagnostics::KernelOccupancyMap KernelDiagnostics::occupancy_;

//---------------------------------------------------------------------------//
} // namespace celeritas