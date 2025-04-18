/*
 * KCachedMagneticChargeDensitySolverBuilder.cc
 *
 *  Created on: 18 Apr 2025
 *      Author: pslocum
 */

#ifndef SRC_KCACHEDMAGNETICCHARGEDENSITYSOLVERBUILDER_CC_
#define SRC_KCACHEDMAGNETICCHARGEDENSITYSOLVERBUILDER_CC_

#include "KCachedMagneticChargeDensitySolverBuilder.hh"

//#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KCachedMagneticChargeDensitySolverBuilder::~KComplexElement() = default;

STATICINT sKCachedMagneticChargeDensitySolverStructure = KCachedMagneticChargeDensitySolverBuilder::Attribute<std::string>("name") +
                                                 KCachedMagneticChargeDensitySolverBuilder::Attribute<std::string>("hash");

}  // namespace katrin


#endif /* SRC_KCACHEDCHARGEDENSITYSOLVERBUILDER_CC_ */
