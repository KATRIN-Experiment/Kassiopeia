/*
 * KFastMultipoleMatrixGeneratorBuilder.hh
 *
 *  Created on: 18 Aug 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KFASTMULTIPOLEMATRIXGENERATORBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KFASTMULTIPOLEMATRIXGENERATORBUILDER_HH_

#include "KComplexElement.hh"
#include "KFastMultipoleMatrixGenerator.hh"
#include "KElectrostaticBoundaryIntegratorAttributeProcessor.hh"

#include "KEMStreamableThreeVector.hh"

namespace katrin {

typedef KComplexElement<KEMField::KFastMultipoleMatrixGenerator>
KFastMultipoleMatrixGeneratorBuilder;

template< >
inline bool KFastMultipoleMatrixGeneratorBuilder::
	AddAttribute(KContainer* aContainer)
{
    if( aContainer->GetName() == "integrator")
        return AddElectrostaticIntegrator(fObject,aContainer);

    if( aContainer->GetName() == "strategy")
    {
        int val = KEMField::KFMSubdivisionStrategy::Aggressive;
        std::string strategy_name = aContainer->AsReference< string >();
        if(strategy_name == string("aggressive"))
        {
            val = KEMField::KFMSubdivisionStrategy::Aggressive;
        }
        if(strategy_name == string("balanced"))
        {
            val = KEMField::KFMSubdivisionStrategy::Balanced;
        }
        if(strategy_name == string("guided"))
        {
            val = KEMField::KFMSubdivisionStrategy::Guided;
        }
        fObject->SetStrategy(val);
        return true;
    }
	if( aContainer->GetName() == "top_level_divisions"){
		aContainer->CopyTo( fObject, &KEMField::KFastMultipoleMatrixGenerator::
				SetTopLevelDivisions);
		return true;
	}
	if( aContainer->GetName() == "tree_level_divisions" ){
		aContainer->CopyTo( fObject,&KEMField::KFastMultipoleMatrixGenerator::
				SetDivisions);
		return true;
	}
	if( aContainer->GetName() == "expansion_degree" ){
		aContainer->CopyTo( fObject,&KEMField::KFastMultipoleMatrixGenerator::
				SetDegree);
		return true;
	}
	if( aContainer->GetName() == "neighbor_order" ){
		aContainer->CopyTo( fObject,&KEMField::KFastMultipoleMatrixGenerator::
				SetZeromask);
		return true;
	}
	if( aContainer->GetName() == "maximum_tree_depth" ){
		aContainer->CopyTo(fObject,&KEMField::KFastMultipoleMatrixGenerator::
				SetMaximumTreeDepth);
		return true;
	}
	if( aContainer->GetName() == "region_expansion_factor" ){
		aContainer->CopyTo(	fObject,&KEMField::KFastMultipoleMatrixGenerator::
				SetRegionExpansionFactor);
		return true;
	}
	if( aContainer->GetName() == "use_region_size_estimation" )	{
		aContainer->CopyTo( fObject,&KEMField::KFastMultipoleMatrixGenerator::
				SetUseRegionEstimation);
		return true;
	}
	if( aContainer->GetName() == "world_cube_center" ){
		KEMField::KEMStreamableThreeVector center;
		aContainer->CopyTo(center);
		fObject->SetWorldCenter(center.GetThreeVector());
		return true;
	}
	if( aContainer->GetName() == "world_cube_length" ){
		aContainer->CopyTo(	fObject,&KEMField::KFastMultipoleMatrixGenerator::
				SetWorldLength);
		return true;
	}
	if( aContainer->GetName() == "use_caching" ){
		aContainer->CopyTo(	fObject,&KEMField::KFastMultipoleMatrixGenerator::
				SetUseCaching);
		return true;
	}
	if( aContainer->GetName() == "verbosity" ){
		aContainer->CopyTo( fObject,&KEMField::KFastMultipoleMatrixGenerator::
				SetVerbosity);
		return true;
	}
	if( aContainer->GetName() == "insertion_ratio" ){
		aContainer->CopyTo(	fObject,&KEMField::KFastMultipoleMatrixGenerator::
				SetInsertionRatio);
		return true;
	}
	if( aContainer->GetName() == "bias_degree" ){
	        aContainer->CopyTo( fObject,&KEMField::KFastMultipoleMatrixGenerator::
	                SetBiasDegree);
	        return true;
	}
	if( aContainer->GetName() == "allowed_number" ){
	        aContainer->CopyTo( fObject,&KEMField::KFastMultipoleMatrixGenerator::
	                SetAllowedNumber);
	        return true;
	}
	if( aContainer->GetName() == "allowed_fraction" ){
	        aContainer->CopyTo( fObject,&KEMField::KFastMultipoleMatrixGenerator::
	                SetAllowedFraction);
	        return true;
	}
	return false;
}

} /* namespace katrin */

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KFASTMULTIPOLEMATRIXGENERATORBUILDER_HH_ */
