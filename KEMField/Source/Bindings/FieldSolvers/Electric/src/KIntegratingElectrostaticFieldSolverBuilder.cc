/*
 * KIntegratingElectrostaticFieldSolverBuilder.cc
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */

#include "KIntegratingElectrostaticFieldSolverBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KIntegratingElectrostaticFieldSolverBuilder::~KComplexElement() = default;

STATICINT sKIntegratingElectrostaticFieldSolverStructure =
    //! @values numeric, analytic, rwg, reference, default
    KIntegratingElectrostaticFieldSolverBuilder::Attribute<std::string>("integrator") +
    //! if true and kemfield has been compiled with opencl support, opencl will be used in the computation.
    KIntegratingElectrostaticFieldSolverBuilder::Attribute<bool>("use_opencl");

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KIntegratingElectrostaticFieldSolver>(
        "integrating_field_solver");
}  // namespace katrin
