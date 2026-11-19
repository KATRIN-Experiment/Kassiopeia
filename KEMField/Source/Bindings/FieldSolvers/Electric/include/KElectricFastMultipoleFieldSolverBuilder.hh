/*
 * KElectricFastMultipoleFieldSolverBuilder.hh
 *
 *  Created on: 24 Jul 2015
 *      Author: wolfgang
 */

#ifndef KELECTRICFASTMULTIPOLEFIELDSOLVERBUILDER_HH_
#define KELECTRICFASTMULTIPOLEFIELDSOLVERBUILDER_HH_

#include "KComplexElement.hh"
#include "KElectricFastMultipoleFieldSolver.hh"

namespace katrin
{

//! the fast multipole field solver adaptively divides the
//! world cube into smaller cube and computes multipole expansions
//! around the cube center describing the far field for the source elements
//! inside the cube.
//! It then transforms these (among other things using the fast Fourier
//! transformation) into local expansions describing the influence of
//! sources far enough away on the electric field and potential inside the cube.
//! These expansions are stored inside a tree structure describing the different
//! cubes. This is the initialization procedure.
//!
//! After initialization the field and potential is evaluated at a point by
//! computing the corresponding expansion and adding the directly determined
//! contribution of nearby source elements.
//! @example
//! <fast_multipole_field_solver
//!     top_level_divisions="4"
//!     tree_level_divisions="2"
//!     expansion_degree="5"
//!     neighbor_order="2"
//!     maximum_tree_depth="2"
//!     region_expansion_factor="1.1"
//!     use_region_size_estimation="true"
//!     world_cube_center_x="0.1"
//!     world_cube_center_y="0.1"
//!     world_cube_center_z="0.1"
//!     world_cube_length="1.5"
//!     use_caching="true"
//!     verbosity="3"
//!     insertion_ratio="1.33333333333"
//!     use_opencl="true"
//! />
typedef KComplexElement<KEMField::KElectricFastMultipoleFieldSolver> KElectricFastMultipoleFieldSolverBuilder;

template<> inline bool KElectricFastMultipoleFieldSolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "top_level_divisions") {
        fObject->GetParameters()->top_level_divisions = aContainer->AsReference<unsigned int>();
        return true;
    }
    if (aContainer->GetName() == "tree_level_divisions") {
        fObject->GetParameters()->divisions = aContainer->AsReference<unsigned int>();
        return true;
    }
    if (aContainer->GetName() == "expansion_degree") {
        fObject->GetParameters()->degree = aContainer->AsReference<unsigned int>();
        return true;
    }
    if (aContainer->GetName() == "neighbor_order") {
        fObject->GetParameters()->zeromask = aContainer->AsReference<unsigned int>();
        return true;
    }
    if (aContainer->GetName() == "maximum_tree_depth") {
        fObject->GetParameters()->maximum_tree_depth = aContainer->AsReference<unsigned int>();
        return true;
    }
    if (aContainer->GetName() == "region_expansion_factor") {
        fObject->GetParameters()->region_expansion_factor = aContainer->AsReference<double>();
        return true;
    }
    if (aContainer->GetName() == "use_region_size_estimation") {
        fObject->GetParameters()->use_region_estimation = aContainer->AsReference<bool>();
        return true;
    }
    if (aContainer->GetName() == "world_cube_center_x") {
        fObject->GetParameters()->world_center_x = aContainer->AsReference<double>();
        return true;
    }
    if (aContainer->GetName() == "world_cube_center_y") {
        fObject->GetParameters()->world_center_y = aContainer->AsReference<double>();
        return true;
    }
    if (aContainer->GetName() == "world_cube_center_z") {
        fObject->GetParameters()->world_center_z = aContainer->AsReference<double>();
        return true;
    }
    if (aContainer->GetName() == "world_cube_length") {
        fObject->GetParameters()->world_length = aContainer->AsReference<double>();
        return true;
    }
    if (aContainer->GetName() == "use_caching") {
        fObject->GetParameters()->use_caching = aContainer->AsReference<bool>();
        return true;
    }
    if (aContainer->GetName() == "verbosity") {
        fObject->GetParameters()->verbosity = aContainer->AsReference<unsigned int>();
        return true;
    }
    if (aContainer->GetName() == "insertion_ratio") {
        fObject->GetParameters()->insertion_ratio = aContainer->AsReference<double>();
        return true;
    }
    if (aContainer->GetName() == "use_opencl") {
        aContainer->CopyTo(fObject, &KEMField::KElectricFastMultipoleFieldSolver::UseOpenCL);
        return true;
    }
    if (aContainer->GetName() == "split_mode") {
        aContainer->CopyTo(fObject, &KEMField::KElectricFastMultipoleFieldSolver::SetSplitMode);
        return true;
    }
    return false;
}


} /* namespace katrin */
#endif /* KELECTRICFASTMULTIPOLEFIELDSOLVERBUILDER_HH_ */
