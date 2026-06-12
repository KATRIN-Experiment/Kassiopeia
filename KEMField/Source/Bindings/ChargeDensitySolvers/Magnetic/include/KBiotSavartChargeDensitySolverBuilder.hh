/*
 * KBiotSavartChargeDensitySolverBuilder.hh
 *
 *  Created on: 2 Apr 2025
 *      Author: pslocum
 */

#ifndef KBIOTSAVARTCHARGEDENSITYSOLVERBUILDER_HH_
#define KBIOTSAVARTCHARGEDENSITYSOLVERBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMBindingsMessage.hh"
#include "KEMStringUtils.hh"
#include "KBiotSavartChargeDensitySolver.hh"


namespace katrin
{

typedef KComplexElement<KEMField::KBiotSavartChargeDensitySolver> KBiotSavartChargeDensitySolverBuilder;

template<> inline bool KBiotSavartChargeDensitySolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->Empty())
    {
        KEMField::kem_cout(eError) << "ERROR: Biot Savart solver container is empty (did you forget to set up the geometry?)" << eom;
    }
    return true;
}

template<> inline bool KBiotSavartChargeDensitySolverBuilder::AddElement(KContainer* anElement)
{
    if (anElement->Empty())
    {
        KEMField::kem_cout(eError) << "ERROR: Element is empty.)" << eom;
    }
    return true;
}

template<> inline bool KBiotSavartChargeDensitySolverBuilder::End()
{
    return true;
}


} /* namespace katrin */

#endif /* KBIOTSAVARTCHARGEDENSITYSOLVERBUILDER_HH_ */
