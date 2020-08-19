//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//

#include "PrimaryGeneratorAction.hh"

#include <G4ParticleTable.hh>
#include <G4SystemOfUnits.hh>
#include "base/Assert.hh"

namespace geant_exporter
{
//---------------------------------------------------------------------------//
/*!
 * Constructor creates a working particle gun for the Geant4 minimal run
 */
PrimaryGeneratorAction::PrimaryGeneratorAction()
    : G4VUserPrimaryGeneratorAction(), particle_gun_(nullptr)
{
    // Creating the particle gun
    G4int number_of_particles = 1;
    particle_gun_ = std::make_unique<G4ParticleGun>(number_of_particles);

    // Selecting the particle
    G4ParticleDefinition* particle;
    particle = G4ParticleTable::GetParticleTable()->FindParticle("e-");
    CHECK(particle);

    // Setting up the particle gun
    G4ThreeVector pos(0, 0, 0);
    particle_gun_->SetParticleDefinition(particle);
    particle_gun_->SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
    particle_gun_->SetParticleEnergy(10 * GeV);
    particle_gun_->SetParticlePosition(pos);
}

//---------------------------------------------------------------------------//
PrimaryGeneratorAction::~PrimaryGeneratorAction() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct source particles at the beginning of each event.
 */
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    particle_gun_->GeneratePrimaryVertex(event);
}

//---------------------------------------------------------------------------//
} // namespace geant_exporter
