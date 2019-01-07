/*
 * KGaussianEliminationChargeDensitySolverBuilder.hh
 *
 *  Created on: 23 Jun 2015
 *      Author: wolfgang
 */

#ifndef KGAUSSIANELIMINATIONCHARGEDENSITYSOLVERBUILDER_HH_
#define KGAUSSIANELIMINATIONCHARGEDENSITYSOLVERBUILDER_HH_

#include "KElectrostaticBoundaryIntegratorAttributeProcessor.hh"
#include "KEMBindingsMessage.hh"
#include "KComplexElement.hh"
#include "KGaussianEliminationChargeDensitySolver.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"

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



#endif /* KGAUSSIANELIMINATIONCHARGEDENSITYSOLVERBUILDER_HH_ */
