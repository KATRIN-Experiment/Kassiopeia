/*
 * KMagfieldCoilsFieldSolverBuilder.cc
 *
 *  Created on: 31 Jan 2023
 *      Author: Jan Behrens
 */

#include "KMagfieldCoilsFieldSolverBuilder.hh"

#include "KStaticElectromagnetFieldBuilder.hh"

using namespace KEMField;
namespace katrin
{

template<> KMagfieldCoilsFieldSolverBuilder::~KComplexElement() = default;

STATICINT sKMagfieldCoilsFieldSolverStructure =
    KMagfieldCoilsFieldSolverBuilder::Attribute<string>("directory") +  // dir name
    KMagfieldCoilsFieldSolverBuilder::Attribute<string>("name") +  // object name
    KMagfieldCoilsFieldSolverBuilder::Attribute<string>("file") +  // coil file name
    KMagfieldCoilsFieldSolverBuilder::Attribute<bool>("replace_file") +  // write coil file
    KMagfieldCoilsFieldSolverBuilder::Attribute<bool>("force_elliptic") +
    KMagfieldCoilsFieldSolverBuilder::Attribute<unsigned>("n_elliptic") +
    KMagfieldCoilsFieldSolverBuilder::Attribute<unsigned>("n_max") +
    KMagfieldCoilsFieldSolverBuilder::Attribute<double>("eps_tol");

STATICINT sKStaticElectromagnetFieldStructure =
    KStaticElectromagnetFieldBuilder::ComplexElement<KMagfieldCoilsFieldSolver>("magfield_coil_field_solver");

} /* namespace katrin */
