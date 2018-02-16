/*
 * KIntegratingElectrostaticFieldSolverBuilder.cc
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */

#include "KIntegratingElectrostaticFieldSolverBuilder.hh"
#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin {

template< >
KIntegratingElectrostaticFieldSolverBuilder::~KComplexElement()
{
}

STATICINT sKIntegratingElectrostaticFieldSolverStructure =
		KIntegratingElectrostaticFieldSolverBuilder::Attribute<std::string >( "integrator") +
		KIntegratingElectrostaticFieldSolverBuilder::Attribute< bool >( "use_opencl" );

STATICINT sKElectrostaticBoundaryField =
KElectrostaticBoundaryFieldBuilder::ComplexElement< KIntegratingElectrostaticFieldSolver >( "integrating_field_solver" );
} // katrin
