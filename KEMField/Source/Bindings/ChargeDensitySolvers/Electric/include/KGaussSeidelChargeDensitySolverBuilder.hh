/*
 * KGaussSeidelChargeDensitySolverBuilder.hh
 *
 *  Created on: 23 Jun 2015
 *      Author: wolfgang
 */

#ifndef KGAUSSSEIDELCHARGEDENSITYSOLVERBUILDER_HH_
#define KGAUSSSEIDELCHARGEDENSITYSOLVERBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMBindingsMessage.hh"
#include "KElectrostaticBoundaryIntegratorAttributeProcessor.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"
#include "KGaussSeidelChargeDensitySolver.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KGaussSeidelChargeDensitySolver>
    KGaussSeidelChargeDensitySolverBuilder;

template<> inline bool KGaussSeidelChargeDensitySolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "integrator")
        return AddElectrostaticIntegratorPolicy(fObject, aContainer);

    if (aContainer->GetName() == "use_opencl") {
        aContainer->CopyTo(fObject, &KEMField::KGaussSeidelChargeDensitySolver::UseOpenCL);
        return true;
    }
    return false;
}

}  // namespace katrin


#endif /* KGAUSSSEIDELCHARGEDENSITYSOLVERBUILDER_HH_ */
