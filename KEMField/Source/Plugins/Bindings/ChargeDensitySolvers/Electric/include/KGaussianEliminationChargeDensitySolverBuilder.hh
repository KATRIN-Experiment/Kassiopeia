/*
 * KGaussianEliminationChargeDensitySolverBuilder.hh
 *
 *  Created on: 23 Jun 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KGAUSSIANELIMINATIONCHARGEDENSITYSOLVERBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KGAUSSIANELIMINATIONCHARGEDENSITYSOLVERBUILDER_HH_

#include "KComplexElement.hh"
#include "KGaussianEliminationChargeDensitySolver.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"
#include "KElectrostaticBoundaryIntegratorAttributeProcessor.hh"
#include "KEMBindingsMessage.hh"

namespace katrin {

typedef KComplexElement< KEMField::KGaussianEliminationChargeDensitySolver >
			KGaussianEliminationChargeDensitySolverBuilder;

template< >
bool KGaussianEliminationChargeDensitySolverBuilder::AddAttribute(KContainer* aContainer)
{
	if(aContainer->GetName() == "integrator")
		return AddElectrostaticIntegratorPolicy(fObject,aContainer);

	return false;
}

} // katrin



#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KGAUSSIANELIMINATIONCHARGEDENSITYSOLVERBUILDER_HH_ */
