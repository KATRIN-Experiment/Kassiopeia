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

KCachedChargeDensitySolver::KCachedChargeDensitySolver() : fName(), fHash() {}
KCachedChargeDensitySolver::~KCachedChargeDensitySolver() {}
void KCachedChargeDensitySolver::InitializeCore(KSurfaceContainer& container)
{
    bool tSolution = false;

    if ((fName.size() == 0) && (fHash.size() == 0)) {
        throw KEMSimpleException("must provide a name or a hash for cached bem solution");
    }

    if (fName.size() != 0) {
        KEMFileInterface::GetInstance()->FindByName(container, fName, tSolution);
    }
    else if (fHash.size() != 0) {
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
