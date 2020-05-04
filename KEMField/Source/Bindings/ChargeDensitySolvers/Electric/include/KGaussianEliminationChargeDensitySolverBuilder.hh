/*
 * KGaussianEliminationChargeDensitySolverBuilder.hh
 *
 *  Created on: 23 Jun 2015
 *      Author: wolfgang
 */

#ifndef KGAUSSIANELIMINATIONCHARGEDENSITYSOLVERBUILDER_HH_
#define KGAUSSIANELIMINATIONCHARGEDENSITYSOLVERBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMBindingsMessage.hh"
#include "KElectrostaticBoundaryIntegratorAttributeProcessor.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"
#include "KGaussianEliminationChargeDensitySolver.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KGaussianEliminationChargeDensitySolver>
    KGaussianEliminationChargeDensitySolverBuilder;

template<> bool KGaussianEliminationChargeDensitySolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "integrator")
        return AddElectrostaticIntegratorPolicy(fObject, aContainer);

    return false;
}

}  // namespace katrin


#endif /* KGAUSSIANELIMINATIONCHARGEDENSITYSOLVERBUILDER_HH_ */
