/*
 * KKrylovPreconditionerGeneratorBuilder.hh
 *
 *  Created on: 19 Aug 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_PreconditionerGeneratorS_ELECTRIC_INCLUDE_KKRYLOVPRECONDITIONERGENERATORBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_PreconditionerGeneratorS_ELECTRIC_INCLUDE_KKRYLOVPRECONDITIONERGENERATORBUILDER_HH_

#include "KComplexElement.hh"
#include "KKrylovPreconditionerGenerator.hh"
#include "KEMBindingsMessage.hh"
#include "KKrylovSolverConfigurationReader.hh"

namespace katrin {

typedef KComplexElement<KEMField::KKrylovPreconditionerGenerator>
KKrylovPreconditionerGeneratorBuilder;

template< >
inline bool KKrylovPreconditionerGeneratorBuilder::
	AddAttribute(KContainer* aContainer)
{
	if(SetKrylovSolverConfiguration(*aContainer,*fObject))
		return true;
	return false;
}

template< >
inline bool KKrylovPreconditionerGeneratorBuilder::
	AddElement(KContainer* anElement)
{
	if(SetKrylovSolverMatrixAndPrecon(*anElement,*fObject))
		return true;
	return false;
}

template< >
inline bool KKrylovPreconditionerGeneratorBuilder::End() {
	if(!fObject->GetMatrixGenerator())
	{
		BINDINGMSG( eError ) << " No matrix specified in krylov_preconditioner."
				<< eom;
	}
	else return true;
	return false;
}

} /* namespace katrin */

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_PreconditionerGeneratorS_ELECTRIC_INCLUDE_KKRYLOVPRECONDITIONERGENERATORBUILDER_HH_ */
