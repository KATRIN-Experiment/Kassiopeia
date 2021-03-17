/*
 * KCachedChargeDensitySolver.cc
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */

#include "KCachedChargeDensitySolver.hh"

#include "KEMFileInterface.hh"
#include "KEMSimpleException.hh"

namespace KEMField
{

KCachedChargeDensitySolver::KCachedChargeDensitySolver() = default;
KCachedChargeDensitySolver::~KCachedChargeDensitySolver() = default;
void KCachedChargeDensitySolver::InitializeCore(KSurfaceContainer& container)
{
    bool tSolution = false;

    if ((fName.empty()) && (fHash.empty())) {
        throw KEMSimpleException("must provide a name or a hash for cached bem solution");
    }

    if (!fName.empty()) {
        kem_cout_debug("cached charge density solver looking for solution with name <" << fName << "> ..." << eom);
        KEMFileInterface::GetInstance()->FindByName(container, fName, tSolution);
    }
    else if (!fHash.empty()) {
        kem_cout_debug("cached charge density solver looking for solution with hash <" << fHash << "> ..." << eom);
        KEMFileInterface::GetInstance()->FindByHash(container, fHash, tSolution);
    }

    if (tSolution == false) {
        throw KEMSimpleException("could not find cached bem solution in directory <" +
                                 KEMFileInterface::GetInstance()->ActiveDirectory() +
                                 ">\n"
                                 "with name <" +
                                 fName + "> and hash <" + fHash + ">");
    }
}

}  // namespace KEMField
