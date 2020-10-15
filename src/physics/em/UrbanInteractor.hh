//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UrbanInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/SecondaryAllocatorView.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "UrbanInteractorPointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform multiple scattering using the Urban model
 *
 * This is a model for the continuous-discrete multiple scattering of electrons
 * and positrons.
 *
 * \note This performs the same sampling routine as in Geant4's G4UrbanMscModel
 * class, as documented in section 8.1 of the Geant4 Physics Reference (release
 * 10.6).
 */
class UrbanInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION UrbanInteractor(const UrbanInteractorPointers& shared,
                                          const ParticleTrackView& particle,
                                          const Real3& inc_direction,
                                          SecondaryAllocatorView& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

    // >>> COMMON PROPERTIES

    //! Minimum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_incident_energy()
    {
        return units::MevEnergy{0};
    }

    //! Maximum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_incident_energy()
    {
        return units::MevEnergy{100.0}; // XXX
    }

  private:
    // Shared constant physics properties
    const UrbanInteractorPointers& shared_;
    // Incident gamma energy
    const units::MevEnergy inc_energy_;
    // Incident direction
    const Real3& inc_direction_;
    // Allocate space for one or more secondary particles
    SecondaryAllocatorView& allocate_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "UrbanInteractor.i.hh"
