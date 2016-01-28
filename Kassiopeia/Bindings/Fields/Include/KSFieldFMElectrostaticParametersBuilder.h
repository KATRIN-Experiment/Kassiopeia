/*
 * KSFieldFastMultipoleBEMSolverBuilder.h
 *
 *  Created on: 20.04.2015
 *      Author: gosda
 */

#ifndef KSFIELDPASTMULTIPOLEBEMSOLVERBUILDER_H_
#define KSFIELDPASTMULTIPOLEBEMSOLVERBUILDER_H_

#include "KComplexElement.hh"
#include "KFMElectrostaticParameters.hh"

using namespace KEMField;
namespace katrin
{

typedef KComplexElement< KFMElectrostaticParameters >
			KSFieldFMElectrostaticParametersBuilder;

template< >
inline bool KSFieldFMElectrostaticParametersBuilder::
	AddAttribute(KContainer* aContainer)
{
	if( aContainer->GetName() == "strategy")
	{
		int val = KFMSubdivisionStrategy::Aggressive;
		string strategy_name = aContainer->AsReference< string >();
		if(strategy_name == string("aggressive"))
		{
			val = KFMSubdivisionStrategy::Aggressive;
		}
		if(strategy_name == string("balanced"))
		{
			val = KFMSubdivisionStrategy::Balanced;
		}
		if(strategy_name == string("guided"))
		{
			val = KFMSubdivisionStrategy::Guided;
		}
		fObject->strategy = val;
		return true;
	}
	if( aContainer->GetName() == "top_level_divisions"){
		aContainer->CopyTo( fObject->top_level_divisions);
		return true;
	}
	if( aContainer->GetName() == "tree_level_divisions" ){
		aContainer->CopyTo( fObject->divisions );
		return true;
	}
	if( aContainer->GetName() == "expansion_degree" ){
		aContainer->CopyTo( fObject->degree);
		return true;
	}
	if( aContainer->GetName() == "neighbor_order" ){
		aContainer->CopyTo( fObject->zeromask);
		return true;
	}
	if( aContainer->GetName() == "maximum_tree_depth" ){
		aContainer->CopyTo(fObject->maximum_tree_depth);
		return true;
	}
	if( aContainer->GetName() == "region_expansion_factor" ){
		aContainer->CopyTo(	fObject->region_expansion_factor);
		return true;
	}
	if( aContainer->GetName() == "use_region_size_estimation" )	{
		aContainer->CopyTo( fObject->use_region_estimation);
		return true;
	}
	if( aContainer->GetName() == "world_cube_center_x" ){
		aContainer->CopyTo( fObject->world_center_x);
		return true;
	}
	if( aContainer->GetName() == "world_cube_center_y" ){
		aContainer->CopyTo(	fObject->world_center_y);
		return true;
	}
	if( aContainer->GetName() == "world_cube_center_z" ){
		aContainer->CopyTo( fObject->world_center_z);
		return true;
	}
	if( aContainer->GetName() == "world_cube_length" ){
		aContainer->CopyTo(	fObject->world_length);
		return true;
	}
	if( aContainer->GetName() == "use_caching" ){
		aContainer->CopyTo(	fObject->use_caching);
		return true;
	}
	if( aContainer->GetName() == "verbosity" ){
		aContainer->CopyTo( fObject->verbosity);
		return true;
	}
	if( aContainer->GetName() == "insertion_ratio" ){
		aContainer->CopyTo(	fObject->insertion_ratio);
		return true;
	}
	if( aContainer->GetName() == "bias_degree" ){
		aContainer->CopyTo(	fObject->bias_degree);
		return true;
	}
	if( aContainer->GetName() == "allowed_number" ){
		aContainer->CopyTo(	fObject->allowed_number);
		return true;
	}
	if( aContainer->GetName() == "allowed_fraction" ){
		aContainer->CopyTo(	fObject->allowed_fraction);
		return true;
	}
	return false;
}

}



#endif /* KSFIELDPASTMULTIPOLEBEMSOLVERBUILDER_H_ */
