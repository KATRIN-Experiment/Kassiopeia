/*
 * KRobinHoodChargeDensitySolverBuilder.hh
 *
 *  Created on: 29 Jul 2015
 *      Author: wolfgang
 */

#ifndef KROBINHOODCHARGEDENSITYSOLVERBUILDER_HH_
#define KROBINHOODCHARGEDENSITYSOLVERBUILDER_HH_

#include "KComplexElement.hh"
#include "KElectrostaticBoundaryIntegratorAttributeProcessor.hh"
#include "KRobinHoodChargeDensitySolver.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KRobinHoodChargeDensitySolver> KRobinHoodChargeDensitySolverBuilder;

template<> inline bool KRobinHoodChargeDensitySolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "integrator")
        return AddElectrostaticIntegratorPolicy(fObject, aContainer);

    if (aContainer->GetName() == "tolerance") {
        aContainer->CopyTo(fObject, &KEMField::KRobinHoodChargeDensitySolver::SetTolerance);
        return true;
    }
    if (aContainer->GetName() == "check_sub_interval") {
        aContainer->CopyTo(fObject, &KEMField::KRobinHoodChargeDensitySolver::SetCheckSubInterval);
        return true;
    }
    if (aContainer->GetName() == "display_interval") {
        aContainer->CopyTo(fObject, &KEMField::KRobinHoodChargeDensitySolver::SetDisplayInterval);
        return true;
    }
    if (aContainer->GetName() == "write_interval") {
        aContainer->CopyTo(fObject, &KEMField::KRobinHoodChargeDensitySolver::SetWriteInterval);
        return true;
    }
    if (aContainer->GetName() == "plot_interval") {
        aContainer->CopyTo(fObject, &KEMField::KRobinHoodChargeDensitySolver::SetPlotInterval);
        return true;
    }
    if (aContainer->GetName() == "cache_matrix_elements") {
        aContainer->CopyTo(fObject, &KEMField::KRobinHoodChargeDensitySolver::CacheMatrixElements);
        return true;
    }
    if (aContainer->GetName() == "use_opencl") {
        aContainer->CopyTo(fObject, &KEMField::KRobinHoodChargeDensitySolver::UseOpenCL);
        return true;
    }
    if (aContainer->GetName() == "use_vtk") {
        aContainer->CopyTo(fObject, &KEMField::KRobinHoodChargeDensitySolver::UseVTK);
        return true;
    }
    return false;
}


} /* namespace katrin */

#endif /* KROBINHOODCHARGEDENSITYSOLVERBUILDER_HH_ */
