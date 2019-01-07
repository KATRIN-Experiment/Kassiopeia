/*
 * KIntegratingMagnetostaticFieldSolverBuilder.cc
 *
 *  Created on: 28 Mar 2016
 *      Author: wolfgang
 */

#include "KIntegratingMagnetostaticFieldSolverBuilder.hh"

#include "KStaticElectromagnetFieldBuilder.hh"

using namespace KEMField;
namespace katrin {

template< >
KIntegratingMagnetostaticFieldSolverBuilder::~KComplexElement()
{
}

STATICINT sKStaticElectromagnetFieldStructure =
        KStaticElectromagnetFieldBuilder::ComplexElement< KIntegratingMagnetostaticFieldSolver >("integrating_field_solver");

} /* namespace katrin */
