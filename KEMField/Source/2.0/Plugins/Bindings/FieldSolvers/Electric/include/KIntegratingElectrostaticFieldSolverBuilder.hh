/*
 * KIntegratingElectrostaticFieldSolverBuilder.hh
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDSOLVERS_ELECTRIC_INCLUDE_KINTEGRATINGELECTROSTATICFIELDSOLVERBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDSOLVERS_ELECTRIC_INCLUDE_KINTEGRATINGELECTROSTATICFIELDSOLVERBUILDER_HH_

#include "KComplexElement.hh"
#include "KIntegratingElectrostaticFieldSolver.hh"
#include "KElectrostaticBoundaryIntegratorAttributeProcessor.hh"

namespace katrin {

typedef KComplexElement< KEMField::KIntegratingElectrostaticFieldSolver > KIntegratingElectrostaticFieldSolverBuilder;

template< >
inline bool KIntegratingElectrostaticFieldSolverBuilder::AddAttribute( KContainer* aContainer )
{
	if( aContainer->GetName() == "integrator" )
		return AddElectrostaticIntegratorPolicy(fObject,aContainer);

	if( aContainer->GetName() == "use_opencl" )
	{
		aContainer->CopyTo( fObject, &KEMField::KIntegratingElectrostaticFieldSolver::UseOpenCL );
		return true;
	}
	return false;
}

} //katrin




#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDSOLVERS_ELECTRIC_INCLUDE_KINTEGRATINGELECTROSTATICFIELDSOLVERBUILDER_HH_ */
