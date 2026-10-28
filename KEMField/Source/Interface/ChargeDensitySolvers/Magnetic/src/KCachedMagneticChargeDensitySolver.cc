/*
 * KCachedMagneticChargeDensitySolver.cc
 *
 *  Created on: 18 Apr 2025
 *      Author: pslocum
 */

#include "KCachedMagneticChargeDensitySolver.hh"

#include "KEMFileInterface.hh"
#include "KEMSimpleException.hh"

namespace KEMField
{

KCachedMagneticChargeDensitySolver::KCachedMagneticChargeDensitySolver() = default;
KCachedMagneticChargeDensitySolver::~KCachedMagneticChargeDensitySolver() = default;
void KCachedMagneticChargeDensitySolver::InitializeCore(KSurfaceContainer& container)
{
    bool tSolution = false;

    if ((fName.empty()) && (fHash.empty())) {
        throw KEMSimpleException("must provide a name or a hash for cached solution");
    }

    if (!fName.empty()) {
        kem_cout_debug("cached magnetic charge density solver looking for solution with name <" << fName << "> ..." << eom);
        KEMFileInterface::GetInstance()->FindByName(container, fName, tSolution);
    }
    else if (!fHash.empty()) {
        kem_cout_debug("cached magnetic charge density solver looking for solution with hash <" << fHash << "> ..." << eom);
        KEMFileInterface::GetInstance()->FindByHash(container, fHash, tSolution);
    }

    if (tSolution == false) {
        throw KEMSimpleException("could not find cached solution in directory <" +
                                 KEMFileInterface::GetInstance()->ActiveDirectory() +
                                 ">\n"
                                 "with name <" +
                                 fName + "> and hash <" + fHash + ">");
    }
}

}  // namespace KEMField
