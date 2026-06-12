/*
 * KCachedMagneticChargeDensitySolverBuilder.hh
 *
 *  Created on: 18 Apr 2025
 *      Author: pslocum
 */

#ifndef KCACHEDMAGNETICCHARGEDENSITYSOLVERBUILDER_HH_
#define KCACHEDMAGNETICCHARGEDENSITYSOLVERBUILDER_HH_

#include "KCachedMagneticChargeDensitySolver.hh"
#include "KComplexElement.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KCachedMagneticChargeDensitySolver> KCachedMagneticChargeDensitySolverBuilder;

template<> inline bool KCachedMagneticChargeDensitySolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        fObject->SetName(name);
        return true;
    }
    if (aContainer->GetName() == "hash") {
        std::string hash;
        aContainer->CopyTo(hash);
        fObject->SetHash(hash);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif /* KCACHEDMAGNETICCHARGEDENSITYSOLVERBUILDER_HH_ */
