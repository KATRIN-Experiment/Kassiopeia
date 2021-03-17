/*
 * KCachedChargeDensitySolverBuilder.cc
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */

#ifndef SRC_KCACHEDCHARGEDENSITYSOLVERBUILDER_CC_
#define SRC_KCACHEDCHARGEDENSITYSOLVERBUILDER_CC_

#include "KCachedChargeDensitySolverBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KCachedChargeDensitySolverBuilder::~KComplexElement() = default;

STATICINT sKCachedChargeDensitySolverStructure = KCachedChargeDensitySolverBuilder::Attribute<std::string>("name") +
                                                 KCachedChargeDensitySolverBuilder::Attribute<std::string>("hash");

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KCachedChargeDensitySolver>("cached_bem_solver") +
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KCachedChargeDensitySolver>("cached_charge_density_solver");
}  // namespace katrin


#endif /* SRC_KCACHEDCHARGEDENSITYSOLVERBUILDER_CC_ */
