/*
 * KBiotSavartChargeDensitySolver.cc
 *
 *  Created on: 18 Apr 2025
 *      Author: pslocum
 */

#include "KBiotSavartChargeDensitySolver.hh"

#include "KEMCoreMessage.hh"

namespace KEMField
{

KBiotSavartChargeDensitySolver::KBiotSavartChargeDensitySolver()
{
}

KBiotSavartChargeDensitySolver::~KBiotSavartChargeDensitySolver() = default;

void KBiotSavartChargeDensitySolver::InitializeCore(KSurfaceContainer& container)
{
    if (container.empty()) {
        kem_cout(eError) << "ERROR: Biot Savart solver container is empty (did you forget to set up the geometry?)" << eom;
    }
}


} /* namespace KEMField */
