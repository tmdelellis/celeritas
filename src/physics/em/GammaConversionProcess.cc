//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GammaConversionProcess.cc
//---------------------------------------------------------------------------//
#include "GammaConversionProcess.hh"

#include <utility>
#include "BetheHeitlerModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from particles and imported Geant data.
 */
GammaConversionProcess::GammaConversionProcess(SPConstParticles particles,
                                               SPConstImported  process_data)
    : particles_(std::move(particles))
    , imported_(process_data,
                particles_,
                ImportProcessClass::conversion,
                {pdg::gamma()})
{
    CELER_EXPECT(particles_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto GammaConversionProcess::build_models(ModelIdGenerator next_id) const
    -> VecModel
{
    return {std::make_shared<BetheHeitlerModel>(next_id(), *particles_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto GammaConversionProcess::step_limits(Applicability applic) const
    -> StepLimitBuilders
{
    return imported_.step_limits(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string GammaConversionProcess::label() const
{
    return "Photon annihiliation";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
