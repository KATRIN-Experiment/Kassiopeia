/*
 * KIntegratingMagnetostaticFieldSolverBuilder.hh
 *
 *  Created on: 28 Mar 2016
 *      Author: wolfgang
 */

#ifndef KINTEGRATINGMAGNETOSTATICFIELDSOLVERBUILDER_HH_
#define KINTEGRATINGMAGNETOSTATICFIELDSOLVERBUILDER_HH_

#include "KComplexElement.hh"
#include "KIntegratingMagnetostaticFieldSolver.hh"
namespace katrin
{

//! performs a sum over all components to obtain the final field.
//! @example <integrating_field_solver/>
typedef KComplexElement<KEMField::KIntegratingMagnetostaticFieldSolver> KIntegratingMagnetostaticFieldSolverBuilder;

} /* namespace katrin */

#endif /* KINTEGRATINGMAGNETOSTATICFIELDSOLVERBUILDER_HH_ */
