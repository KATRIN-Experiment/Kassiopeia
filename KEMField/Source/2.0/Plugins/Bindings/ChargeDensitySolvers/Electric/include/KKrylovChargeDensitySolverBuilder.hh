/*
 * KKrylovChargeDensitySolverBuilder.hh
 *
 *  Created on: 17 Aug 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KKRYLOVCHARGEDENSITYSOLVERBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KKRYLOVCHARGEDENSITYSOLVERBUILDER_HH_

#include "KKrylovChargeDensitySolver.hh"
#include "KComplexElement.hh"

#include "KBoundaryMatrixGenerator.hh"

#include "KEMBindingsMessage.hh"

#include "KSmartPointerRelease.hh"
#include "KEMStringUtils.hh"

#include "KKrylovSolverConfigurationReader.hh"

namespace katrin {

typedef KComplexElement< KEMField::KKrylovChargeDensitySolver >
KKrylovChargeDensitySolverBuilder;

template< >
inline bool KKrylovChargeDensitySolverBuilder::
	AddAttribute(KContainer* aContainer)
{
	if(SetKrylovSolverConfiguration(*aContainer,*fObject))
		return true;
	return false;
}

template< >
inline bool KKrylovChargeDensitySolverBuilder::
		AddElement(KContainer * anElement)
{
	if(SetKrylovSolverMatrixAndPrecon(*anElement,*fObject))
		return true;
	return false;
}

template< >
inline bool KKrylovChargeDensitySolverBuilder::End() {
	if(!fObject->GetMatrixGenerator())
	{
		BINDINGMSG( eError ) << " No matrix specified in krylov_charge_"
				"density_solver or krylov_bem_solver." << eom;
	}
	else return true;
	return false;
}

} /* namespace katrin */

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KKRYLOVCHARGEDENSITYSOLVERBUILDER_HH_ */
