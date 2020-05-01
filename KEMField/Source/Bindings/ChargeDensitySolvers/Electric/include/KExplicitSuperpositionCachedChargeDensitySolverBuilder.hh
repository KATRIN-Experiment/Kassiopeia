/*
 * KExplicitSuperpositionCachedChargeDensitySolverBuilder.hh
 *
 *  Created on: 27 Jun 2016
 *      Author: wolfgang
 */

#ifndef KEXPLICITSUPERPOSITIONCACHEDCHARGEDENSITYSOLVERBUILDER_HH_
#define KEXPLICITSUPERPOSITIONCACHEDCHARGEDENSITYSOLVERBUILDER_HH_

#include "KComplexElement.hh"
#include "KExplicitSuperpositionCachedChargeDensitySolver.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KExplicitSuperpositionCachedChargeDensitySolver>
    KExplicitSuperpositionCachedChargeDensitySolverBuilder;

template<> inline bool KExplicitSuperpositionCachedChargeDensitySolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KEMField::KExplicitSuperpositionCachedChargeDensitySolver::SetName);
        return true;
    }
    return false;
}

template<> inline bool KExplicitSuperpositionCachedChargeDensitySolverBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "component") {
        anElement->ReleaseTo(fObject, &KEMField::KExplicitSuperpositionCachedChargeDensitySolver::AddSolutionComponent);
        return true;
    }
    return false;
}

} /* namespace katrin */

#endif /* KEXPLICITSUPERPOSITIONCACHEDCHARGEDENSITYSOLVERBUILDER_HH_ */
