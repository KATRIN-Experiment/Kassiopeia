/*
 * KKrylovChargeDensitySolverBuilder.hh
 *
 *  Created on: 17 Aug 2015
 *      Author: wolfgang
 */

#ifndef KKRYLOVCHARGEDENSITYSOLVERBUILDER_HH_
#define KKRYLOVCHARGEDENSITYSOLVERBUILDER_HH_

#include "KBoundaryMatrixGenerator.hh"
#include "KComplexElement.hh"
#include "KEMBindingsMessage.hh"
#include "KEMStringUtils.hh"
#include "KKrylovChargeDensitySolver.hh"
#include "KKrylovSolverConfigurationReader.hh"
#include "KSmartPointerRelease.hh"


namespace katrin
{

typedef KComplexElement<KEMField::KKrylovChargeDensitySolver> KKrylovChargeDensitySolverBuilder;

template<> inline bool KKrylovChargeDensitySolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (SetKrylovSolverConfiguration(*aContainer, *fObject))
        return true;
    return false;
}

template<> inline bool KKrylovChargeDensitySolverBuilder::AddElement(KContainer* anElement)
{
    if (SetKrylovSolverMatrixAndPrecon(*anElement, *fObject))
        return true;
    return false;
}

template<> inline bool KKrylovChargeDensitySolverBuilder::End()
{
    if (!fObject->GetMatrixGenerator()) {
        BINDINGMSG(eError) << " No matrix specified in krylov_charge_"
                              "density_solver or krylov_bem_solver."
                           << eom;
    }
    else
        return true;
    return false;
}

} /* namespace katrin */

#endif /* KKRYLOVCHARGEDENSITYSOLVERBUILDER_HH_ */
