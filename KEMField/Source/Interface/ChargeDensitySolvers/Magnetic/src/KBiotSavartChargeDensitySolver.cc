/*
 * KBiotSavartChargeDensitySolver.cc
 *
 *  Created on: 12 Aug 2015
 *      Author: wolfgang
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
        kem_cout(eError) << "ERROR: Krylov solver got no electrode elements (did you forget to setup a geometry mesh?)" << eom;
    }
}


} /* namespace KEMField */
