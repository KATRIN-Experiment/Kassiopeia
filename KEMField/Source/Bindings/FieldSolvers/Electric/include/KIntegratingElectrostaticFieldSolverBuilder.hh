/*
 * KIntegratingElectrostaticFieldSolverBuilder.hh
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */

#ifndef KINTEGRATINGELECTROSTATICFIELDSOLVERBUILDER_HH_
#define KINTEGRATINGELECTROSTATICFIELDSOLVERBUILDER_HH_

#include "KElectrostaticBoundaryIntegratorAttributeProcessor.hh"
#include "KComplexElement.hh"
#include "KIntegratingElectrostaticFieldSolver.hh"

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




#endif /* KINTEGRATINGELECTROSTATICFIELDSOLVERBUILDER_HH_ */
