/*
 * KFastMultipoleMatrixGeneratorBuilder.hh
 *
 *  Created on: 18 Aug 2015
 *      Author: wolfgang
 */

#ifndef KFASTMULTIPOLEMATRIXGENERATORBUILDER_HH_
#define KFASTMULTIPOLEMATRIXGENERATORBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMStreamableThreeVector.hh"
#include "KElectrostaticBoundaryIntegratorAttributeProcessor.hh"
#include "KFastMultipoleMatrixGenerator.hh"

namespace katrin
{

//! The fast multipole matrix uses the KATRIN FFTM method to compute the multiplication
//! of the charge density problem matrix with a vector.
//! It works by constructing an adaptive tree of cubes and subcubes, computing
//! multipole expansions as approximations of the electric field by local charges and
//! converting them into local expansion to approximate the field of far away
//! charges. Then the result vector is created by evaluating at the source element's
//! centroids.
//!
//! Most important for accuracy of this approximate multiplication method
//! are the neighbor_order and expansion_degree.
//!
//! The level divisions and maximum tree depth allow to optimize the computation
//! time.
//!
//! The world cube is the top level cube that has to contain all source elements
//! and is a starting point for the algorithm.
//! @example
//! <fast_multipole_matrix
//!     strategy="balanced"
//!     top_level_divisions="4"
//!     tree_level_divisions="2"
//!     expansion_degree="4"
//!     neighbor_order="2"
//!     maximum_tree_depth="2"
//!     region_expansion_factor="1.1"
//!     use_region_size_estimation="true"
//!     world_cube_center="0 0 0"
//!     world_cube_length="10"
//!     use_caching="false"
//!     verbosity="3"
//!     insertion_ratio="1.33333333333"
//!     bias_degree="2"
//!     allowed_number="1"
//!     allowed_fraction="0.5"
//!
//! />
typedef KComplexElement<KEMField::KFastMultipoleMatrixGenerator> KFastMultipoleMatrixGeneratorBuilder;

template<> inline bool KFastMultipoleMatrixGeneratorBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "integrator")
        return AddElectrostaticIntegrator(fObject, aContainer);

    if (aContainer->GetName() == "strategy") {
        int val = KEMField::KFMSubdivisionStrategy::Aggressive;
        std::string strategy_name = aContainer->AsString();
        if (strategy_name == "aggressive") {
            val = KEMField::KFMSubdivisionStrategy::Aggressive;
        }
        if (strategy_name == "balanced") {
            val = KEMField::KFMSubdivisionStrategy::Balanced;
        }
        if (strategy_name == "guided") {
            val = KEMField::KFMSubdivisionStrategy::Guided;
        }
        fObject->SetStrategy(val);
        return true;
    }
    if (aContainer->GetName() == "top_level_divisions") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetTopLevelDivisions);
        return true;
    }
    if (aContainer->GetName() == "tree_level_divisions") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetDivisions);
        return true;
    }
    if (aContainer->GetName() == "expansion_degree") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetDegree);
        return true;
    }
    if (aContainer->GetName() == "neighbor_order") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetZeromask);
        return true;
    }
    if (aContainer->GetName() == "maximum_tree_depth") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetMaximumTreeDepth);
        return true;
    }
    if (aContainer->GetName() == "region_expansion_factor") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetRegionExpansionFactor);
        return true;
    }
    if (aContainer->GetName() == "use_region_size_estimation") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetUseRegionEstimation);
        return true;
    }
    if (aContainer->GetName() == "world_cube_center") {
        KEMField::KEMStreamableThreeVector center;
        aContainer->CopyTo(center);
        fObject->SetWorldCenter(center.GetThreeVector());
        return true;
    }
    if (aContainer->GetName() == "world_cube_length") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetWorldLength);
        return true;
    }
    if (aContainer->GetName() == "use_caching") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetUseCaching);
        return true;
    }
    if (aContainer->GetName() == "verbosity") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetVerbosity);
        return true;
    }
    if (aContainer->GetName() == "insertion_ratio") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetInsertionRatio);
        return true;
    }
    if (aContainer->GetName() == "bias_degree") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetBiasDegree);
        return true;
    }
    if (aContainer->GetName() == "allowed_number") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetAllowedNumber);
        return true;
    }
    if (aContainer->GetName() == "allowed_fraction") {
        aContainer->CopyTo(fObject, &KEMField::KFastMultipoleMatrixGenerator::SetAllowedFraction);
        return true;
    }
    return false;
}

} /* namespace katrin */

#endif /* KFASTMULTIPOLEMATRIXGENERATORBUILDER_HH_ */
