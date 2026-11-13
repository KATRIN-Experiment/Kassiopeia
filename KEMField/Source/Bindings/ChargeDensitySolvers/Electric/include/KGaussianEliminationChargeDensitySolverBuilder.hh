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

//! uses the gaussian elimination technique to compute charge densities.
//! @example
//! <gaussian_elimination_charge_density_solver
//!     integrator="analytic"
//! />
typedef KComplexElement<KEMField::KGaussianEliminationChargeDensitySolver>
    KGaussianEliminationChargeDensitySolverBuilder;

template<> inline bool KGaussianEliminationChargeDensitySolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "integrator")
        return AddElectrostaticIntegratorPolicy(fObject, aContainer);

    return false;
}

}  // namespace katrin


#endif /* KGAUSSIANELIMINATIONCHARGEDENSITYSOLVERBUILDER_HH_ */
